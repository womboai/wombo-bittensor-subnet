# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 WOMBO
import asyncio
import os
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
from asyncio import Future, Lock
from typing import List, Tuple, Optional, AsyncGenerator

import torch
from PIL import Image
from io import BytesIO

# Bittensor
import bittensor as bt
from aiohttp import ClientSession
from bittensor import AxonInfo, TerminalInfo
from fastapi import HTTPException
from starlette import status
from torch import tensor

from image_generation_protocol.io_protocol import ImageGenerationInputs
from tensor.protocol import ImageGenerationSynapse, ImageGenerationClientSynapse, NeuronInfoSynapse
from neuron_selector.uids import get_best_uids
from tensor.timeouts import CLIENT_REQUEST_TIMEOUT, AXON_REQUEST_TIMEOUT, KEEP_ALIVE_TIMEOUT

# import base validator class which takes care of most of the boilerplate
from validator.validator import BaseValidatorNeuron, get_oldest_uids
from validator.reward import get_rewards, select_endpoint

WATERMARK = Image.open("w_watermark.png")


RANDOM_VALIDATION_CHANCE = float(os.getenv("RANDOM_VALIDATION_CHANCE", str(0.25)))


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


class NoMinersAvailableException(Exception):
    def __init__(self, dendrite: TerminalInfo | None):
        super().__init__(f"No miners available for {dendrite} query")
        self.dendrite = dendrite


class GetMinerResponseException(Exception):
    def __init__(self, dendrites: list[TerminalInfo], axons: list[TerminalInfo]):
        super().__init__(f"Failed to query miners, dendrites: {dendrites}")

        self.dendrites = dendrites
        self.axons = axons


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

        self.pending_validation_lock = Lock()
        self.pending_validation_requests: list[Future[None]] = []

    async def score_responses(
        self,
        inputs: ImageGenerationInputs,
        miner_uids: torch.LongTensor,
        axons: List[AxonInfo],
        responses: List[ImageGenerationSynapse],
    ):
        working_miner_uids: List[int] = []
        finished_responses: List[ImageGenerationSynapse] = []

        axon_uids = {
            axon.hotkey: uid.item()
            for uid, axon in zip(miner_uids, axons)
        }

        for response in responses:
            if not response.output or not response.axon or not response.axon.hotkey:
                continue

            working_miner_uids.append(axon_uids[response.axon.hotkey])
            finished_responses.append(response)

        # Log the results for monitoring purposes.
        bt.logging.info(f"Received {len(finished_responses)} responses")

        if not len(finished_responses):
            return None

        try:
            # Adjust the scores based on responses from miners.
            rewards = await get_rewards(
                self,
                query=inputs,
                uids=working_miner_uids,
                responses=finished_responses,
            )
        except Exception as e:
            bt.logging.error("Failed to get rewards for responses", e)
            return

        bt.logging.info(f"Scored responses: {rewards}")
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        self.update_scores(rewards, working_miner_uids)

        bad_miner_uids = [uid.item() for uid in miner_uids if uid.item() not in working_miner_uids]

        self.update_scores(self.scores[tensor(bad_miner_uids, dtype=torch.int64)] * 0.5, bad_miner_uids)

    async def check_miners(self):
        """
        Validator forward pass, called by the validator every time step. Consists of:
        - Generating the query
        - Querying the network miners
        - Getting the responses
        - Rewarding the miners based on their responses
        - Updating the scores
        """

        async with self.pending_validation_lock:
            pending_validation_requests = self.pending_validation_requests.copy()

        if len(pending_validation_requests):
            pending_validation_requests[0].get_loop().run_until_complete(asyncio.gather(*pending_validation_requests))

        miner_uids = get_oldest_uids(self, k=self.config.neuron.sample_size)

        if not len(miner_uids):
            return

        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        max_seed = 2 ** 32
        random_int = random.randint(0, max_seed)
        seed = (self.step * random_int) % max_seed

        base_prompt = str(self.step * random_int)
        selection = random.randint(0, 2)

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

        # The dendrite client queries the network.
        responses: List[ImageGenerationSynapse] = await self.periodic_check_dendrite(
            # Send the query to selected miner axons in the network.
            axons=axons,
            synapse=ImageGenerationSynapse(inputs=inputs),
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
            timeout=CLIENT_REQUEST_TIMEOUT,
        )

        await self.score_responses(inputs, miner_uids, axons, responses)

    async def get_forward_responses(
        self,
        axons: list[AxonInfo],
        synapse: ImageGenerationSynapse,
    ) -> AsyncGenerator[ImageGenerationSynapse, None]:
        responses = asyncio.as_completed([
            self.forward_dendrite(
                axons=axon,
                synapse=synapse,
                deserialize=False,
                timeout=CLIENT_REQUEST_TIMEOUT,
            )
            for axon in axons
        ])

        for response in responses:
            yield await response

    async def validate_user_request_responses(
        self,
        inputs: ImageGenerationInputs,
        finished_response: ImageGenerationSynapse,
        miner_uids: torch.LongTensor,
        axons: list[AxonInfo],
        bad_responses: list[ImageGenerationSynapse],
        response_generator: AsyncGenerator[ImageGenerationSynapse, None],
    ):
        axon_uids = {
            axon.hotkey: uid.item()
            for uid, axon in zip(miner_uids, axons)
        }

        working_miner_uids: List[int] = [axon_uids[finished_response.axon.hotkey]]
        finished_responses: List[ImageGenerationSynapse] = [finished_response]

        async for response in response_generator:
            if not response.output:
                bad_responses.append(response)
                continue

            working_miner_uids.append(axon_uids[response.axon.hotkey])
            finished_responses.append(response)

        if len(bad_responses):
            bad_axons = [response.axon for response in bad_responses]
            bad_dendrites = [response.dendrite for response in bad_responses]
            bad_miner_uids = [axon_uids[axon.hotkey] for axon in bad_axons]

            # Some failed to response, punish them
            self.update_scores(self.scores[tensor(bad_miner_uids, dtype=torch.int64)] * 0.25, bad_miner_uids)

            bt.logging.error(f"Failed to query some miners with {inputs} for axons {bad_axons}, {bad_dendrites}")

        if random.random() < RANDOM_VALIDATION_CHANCE:
            working_axons = [self.metagraph.axons[uid] for uid in working_miner_uids]

            await self.score_responses(
                inputs,
                tensor(working_miner_uids),
                working_axons,
                finished_responses,
            )

    async def forward_image(self, synapse: ImageGenerationClientSynapse) -> ImageGenerationClientSynapse:
        miner_uids = get_best_uids(self, validators=False)

        if not len(miner_uids):
            raise NoMinersAvailableException(synapse.dendrite)

        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        response_generator = self.get_forward_responses(axons, ImageGenerationSynapse(inputs=synapse.inputs))

        bad_responses: list[ImageGenerationSynapse] = []

        async for response in response_generator:
            if response.output:
                synapse.images = response.output.images
                synapse.images = add_watermarks(synapse.deserialize())

                validation_coroutine = self.validate_user_request_responses(
                    synapse.inputs,
                    response,
                    miner_uids,
                    axons,
                    bad_responses,
                    response_generator,
                )

                async with self.pending_validation_lock:
                    self.pending_validation_requests.append(asyncio.ensure_future(validation_coroutine))

                return synapse

            bad_responses.append(response)

        axon_uids = {
            axon.hotkey: uid.item()
            for uid, axon in zip(miner_uids, axons)
        }

        bad_axons = [response.axon for response in bad_responses]
        bad_dendrites = [response.dendrite for response in bad_responses]
        bad_miner_uids = [axon_uids[axon.hotkey] for axon in bad_axons]

        # Some failed to response, punish them
        self.update_scores(self.scores[tensor(bad_miner_uids, dtype=torch.int64)] * 0.25, bad_miner_uids)

        raise GetMinerResponseException(bad_dendrites, bad_axons)

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
