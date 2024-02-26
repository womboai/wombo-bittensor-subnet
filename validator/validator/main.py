# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 WOMBO

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import random
import base64
from typing import List, Tuple
from PIL import Image
from io import BytesIO

# Bittensor
import bittensor as bt
from aiohttp import ClientSession
import torch
from fastapi import HTTPException
from starlette import status

from image_generation_protocol.io_protocol import ImageGenerationInputs
from tensor.protocol import ImageGenerationSynapse, ImageGenerationClientSynapse, NeuronInfoSynapse
from neuron_selector.uids import get_random_uids
from tensor.timeouts import CLIENT_REQUEST_TIMEOUT, AXON_REQUEST_TIMEOUT, KEEP_ALIVE_TIMEOUT

# import base validator class which takes care of most of the boilerplate
from validator.validator import BaseValidatorNeuron
from validator.reward import get_rewards, select_endpoint

WATERMARK = Image.open("w_watermark.png")


def watermark_image(image: Image.Image) -> Image.Image:
    image_copy = image.copy()
    wm = WATERMARK.resize((image_copy.size[0], int(image_copy.size[0] * WATERMARK.size[1] / WATERMARK.size[0])))
    wm, alpha = wm.convert("RGB"), wm.split()[3]
    image_copy.paste(wm, (0, image_copy.size[1] - wm.size[1]), alpha)
    return image_copy


def add_watermarks(images: List[Image.Image]) -> List[bytes]:
    """
    Add watermarks to the images.
    """
    def save_image(image: Image.Image) -> bytes:
        image = watermark_image(image)
        with BytesIO() as image_bytes:
            image.save(image_bytes, format="JPEG")
            return base64.b64encode(image_bytes.getvalue())

    return [save_image(image) for image in images]


def validator_forward_info(synapse: NeuronInfoSynapse):
    synapse.is_validator = True

    return synapse


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        self.axon.attach(forward_fn=validator_forward_info)

        self.axon.attach(
            forward_fn=self.forward_image,
            blacklist_fn=self.blacklist_image,
        )

        self.axon.fast_config.timeout_keep_alive = KEEP_ALIVE_TIMEOUT
        self.axon.fast_config.timeout_notify = AXON_REQUEST_TIMEOUT

        bt.logging.info(f"Axon created: {self.axon}")

        bt.logging.info("load_state()")
        self.load_state()

    async def check_miners(self):
        """
        Validator forward pass, called by the validator every time step. Consists of:
        - Generating the query
        - Querying the network miners
        - Getting the responses
        - Rewarding the miners based on their responses
        - Updating the scores
        """

        miner_uids = get_random_uids(self, k=self.config.neuron.sample_size, validators=False)

        if not len(miner_uids):
            return

        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        max_seed = 2 ** 32
        random_int = random.randint(0, max_seed)
        seed = (self.step * random_int) % max_seed

        base_prompt = str(self.step * random_int)
        selection = random.randint(0, 3)

        if selection == 1:
            prompt = base_prompt.encode("utf-8").hex()
        elif selection == 2:
            prompt = base64.b64encode(base_prompt.encode("utf-8")).decode("ascii")
        else:
            prompt = base_prompt

        input_parameters = {
            "prompt": prompt,
            "seed": seed,
            "width": 512,
            "height": 512,
            "steps": 15,
        }

        inputs = ImageGenerationInputs(**input_parameters)

        bt.logging.info(f"Sending request {input_parameters} to {miner_uids} which have axons {axons}")

        async with self.dendrite as dendrite:
            # The dendrite client queries the network.
            responses: List[ImageGenerationSynapse] = await dendrite.forward(
                # Send the query to selected miner axons in the network.
                axons=axons,
                synapse=ImageGenerationSynapse(inputs=inputs),
                # All responses have the deserialize function called on them before returning.
                # You are encouraged to define your own deserialization function.
                deserialize=False,
                timeout=CLIENT_REQUEST_TIMEOUT,
            )

        working_miner_uids = []
        finished_responses = []

        for response in responses:
            if not response.output or not response.axon or not response.axon.hotkey:
                continue

            uid = [uid for uid, axon in zip(miner_uids, axons) if axon.hotkey == response.axon.hotkey][0]
            working_miner_uids.append(uid)
            finished_responses.append(response)

        # Log the results for monitoring purposes.
        bt.logging.info(f"Received {len(finished_responses)} responses")

        if not len(finished_responses):
            return

        try:
            # Adjust the scores based on responses from miners.
            rewards = await get_rewards(
                self,
                query=inputs,
                uids=[uid.item() for uid in working_miner_uids],
                responses=finished_responses,
            )
        except Exception as e:
            bt.logging.error("Failed to get rewards for responses", e)
            return

        bt.logging.info(f"Scored responses: {rewards}")
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        self.update_scores(rewards, working_miner_uids)

        # punish bad miners
        bad_miner_uids = [uid for uid in miner_uids if uid not in working_miner_uids]
        self.update_scores(torch.FloatTensor([0.0] * len(bad_miner_uids)), bad_miner_uids)

    async def forward_image(self, synapse: ImageGenerationClientSynapse) -> ImageGenerationClientSynapse:
        miner_uids = get_random_uids(self, k=1, validators=False)

        if not len(miner_uids):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No suitable miners found",
            )

        miner_uid = miner_uids[0]

        # Grab the axon you're serving
        axon = self.metagraph.axons[miner_uid]

        async with self.dendrite as dendrite:
            response: ImageGenerationSynapse = (await dendrite.forward(
                axons=[axon],
                synapse=ImageGenerationSynapse(inputs=synapse.inputs),
                deserialize=False,
                timeout=CLIENT_REQUEST_TIMEOUT,
            ))[0]

        if response.output:
            synapse.images = response.output.images
            synapse.images = add_watermarks(synapse.deserialize())

            if random.randint(0, 10) != 0:
                return synapse

            uids = [miner_uid.item()]

            try:
                # Adjust the scores based on responses from miners.
                rewards = await get_rewards(
                    self,
                    query=synapse.inputs,
                    uids=uids,
                    responses=[response],
                )
            except Exception as e:
                bt.logging.error("Failed to get rewards for responses", e)
                return synapse

            self.update_scores(rewards, uids)
        else:
            bt.logging.error(f"Failed to query miner with {synapse.inputs} and axon {axon}, {response.dendrite}")

            raise HTTPException(
                status_code=response.dendrite.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.dendrite.status_message or "Failed to query miner",
            )

    async def blacklist_image(
        self,
        synapse: ImageGenerationClientSynapse,
    ) -> Tuple[bool, str]:
        is_hotkey_allowed_endpoint = select_endpoint(
            self.config.is_hotkey_allowed_endpoint,
            self.config.subtensor.network,
            "https://dev-neuron-identifier.api.wombo.ai/api/is_hotkey_allowed",
            "https://neuron-identifier.api.wombo.ai/api/is_hotkey_allowed",
        )

        async with ClientSession() as session:
            response = await session.get(
                f"{is_hotkey_allowed_endpoint}?hotkey={synapse.dendrite.hotkey}",
                headers={"Content-Type": "application/json"},
            )

            response.raise_for_status()

            is_hotkey_allowed = await response.json()

        if not is_hotkey_allowed:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )

        return False, "Hotkey recognized!"


def main():
    Validator().run()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    main()
