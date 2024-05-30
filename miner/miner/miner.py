#  The MIT License (MIT)
#  Copyright © 2023 Yuma Rao
#  Copyright © 2024 WOMBO
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#
#

import asyncio
import traceback
from asyncio import Semaphore
from typing import Tuple

import bittensor as bt
from aiohttp import ClientSession
from diffusers import StableDiffusionXLControlNetPipeline
from grpc import ServicerContext
from image_generation_protocol.io_protocol import ImageGenerationInputs

from gpu_pipeline.pipeline import get_pipeline
from miner.image_generator import generate
from miner.protos.miner_pb2_grpc import MinerServicer
from neuron.api_handler import ApiHandler
from neuron.neuron import BaseNeuron
from tensor.config import add_args, check_config
from tensor.protocol import ImageGenerationSynapse, NeuronInfo, NeuronCapability, NEURON_INFO_ENDPOINT
from tensor.protos.inputs_pb2 import GenerationRequestInputs


def miner_forward_info():
    return NeuronInfo(capabilities={NeuronCapability.MINER})


class MinerService(MinerServicer):
    def __init__(self, gpu_semaphore: Semaphore, pipeline: StableDiffusionXLControlNetPipeline):
        super().__init__()

        self.gpu_semaphore = gpu_semaphore
        self.pipeline = pipeline

    async def Generate(self, request: GenerationRequestInputs, context: ServicerContext):
        await generate(self.gpu_semaphore, self.pipeline, request)


class Miner(BaseNeuron):
    nonces: dict[str, set[int]]
    nonce_lock: asyncio.Lock
    last_metagraph_sync: int

    def __init__(self):
        super().__init__()

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )

        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # The axon handles request processing, allowing validators to send this miner requests.
        self.api = ApiHandler(self.wallet.hotkey.ss58_address)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to miner axon.")

        self.api.app.get(f"/{NEURON_INFO_ENDPOINT}")(miner_forward_info)

        self.api.attach(
            forward_fn=self.forward_image,
            blacklist_fn=self.blacklist_image,
            priority_fn=self.priority_image,
        )

        self.last_metagraph_sync = self.block

        self.gpu_semaphore, self.pipeline = get_pipeline(self.device)

        self.pipeline.vae = None

        self.image_generator_session = None

    @classmethod
    def check_config(cls, config: bt.config):
        check_config(config, "miner")

    @classmethod
    def add_args(cls, parser):
        add_args(parser)

        parser.add_argument(
            "--blacklist.force_validator_permit",
            action="store_true",
            help="If set, we will force incoming requests to have a permit.",
            default=False,
        )

        parser.add_argument(
            "--blacklist.allow_non_registered",
            action="store_true",
            help="If set, miners will accept queries from non registered entities. (Dangerous!)",
            default=False,
        )

        parser.add_argument(
            "--blacklist.validator_minimum_tao",
            type=int,
            help="The minimum number of TAO needed for a validator's queries to be accepted.",
            default=4096,
        )

    async def run(self):
        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving miner axon {self.api} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.api.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start the miner's axon, making it active on the network.
        self.api.start()

        bt.logging.info(f"Miner starting at block: {self.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while True:
                block = self.block

                if block - self.last_metagraph_sync <= self.config.neuron.epoch_length:
                    await asyncio.sleep(max(self.config.neuron.epoch_length - block + self.last_metagraph_sync, 0) * 12)
                    continue

                # Sync metagraph and potentially set weights.
                self.sync()

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.api.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as _:
            bt.logging.error(traceback.format_exc())

    def sync(self):
        # Ensure miner hotkey is still registered on the network.
        self.check_registered()

        try:
            self.resync_metagraph()
        except Exception as _:
            bt.logging.error("Failed to resync metagraph, ", traceback.format_exc())

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        self.last_metagraph_sync = self.block

    async def forward_image(
        self,
        inputs: ImageGenerationInputs,
    ) -> bytes:
        return

    async def blacklist_image(self, synapse: ImageGenerationSynapse) -> Tuple[bool, str]:
        if not self.image_generator_session:
            self.image_generator_session = ClientSession()

        async with self.image_generator_session.get(
            f"{self.is_whitelisted_endpoint}?hotkey={synapse.dendrite.hotkey}",
            headers={"Content-Type": "application/json"},
        ) as response:
            response.raise_for_status()

            is_hotkey_allowed = await response.json()

        if is_hotkey_allowed:
            return False, "Whitelisted hotkey"

        if not self.config.blacklist.allow_non_registered:
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                # Ignore requests from unrecognized entities.
                bt.logging.trace(
                    f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey"

            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

            if self.config.blacklist.force_validator_permit and not self.metagraph.validator_permit[uid]:
                bt.logging.trace(
                    f"No validator permit for hotkey {synapse.dendrite.hotkey}"
                )
                return True, "No validator permit"

            if self.metagraph.total_stake[uid] < self.config.blacklist.validator_minimum_tao:
                # Ignore requests from unrecognized entities.
                bt.logging.trace(
                    f"Not enough stake for hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Insufficient stake"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority_image(self, synapse: ImageGenerationSynapse) -> float:
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority
