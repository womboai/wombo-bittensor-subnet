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
from asyncio import Semaphore
from datetime import timedelta
from hashlib import sha256
from uuid import uuid4, UUID

import bittensor as bt
import grpc
from diffusers import StableDiffusionXLControlNetPipeline
from google.protobuf.empty_pb2 import Empty
from grpc import StatusCode, HandlerCallDetails
from grpc.aio import Metadata
from redis.asyncio import Redis

from gpu_pipeline.pipeline import get_pipeline, get_tao_img
from miner.image_generator import generate
from neuron.api_handler import request_error, RequestVerifier, HOTKEY_HEADER, serve_ip, WhitelistChecker
from neuron.neuron import BaseNeuron
from neuron.protos.neuron_pb2 import MinerGenerationResponse, MinerGenerationIdentifier, MinerGenerationResult
from neuron.protos.neuron_pb2_grpc import MinerServicer, add_MinerServicer_to_server
from tensor.config import add_args, check_config, SPEC_VERSION
from tensor.input_sanitization import DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_STEPS, DEFAULT_GUIDANCE
from tensor.interceptors import LoggingInterceptor
from tensor.protos.inputs_pb2 import GenerationRequestInputs, InfoResponse, NeuronCapabilities
from tensor.protos.inputs_pb2_grpc import NeuronServicer, add_NeuronServicer_to_server
from tensor.response import axon_address


class MinerInfoService(NeuronServicer):
    def Info(self, request: Empty, context: HandlerCallDetails):
        return InfoResponse(
            spec_version=SPEC_VERSION,
            capabilities=[NeuronCapabilities.MINER],
        )


class MinerGenerationService(MinerServicer):
    def __init__(
        self,
        config: bt.config,
        metagraph: bt.metagraph,
        hotkey: str,
        is_whitelisted_endpoint: str,
        redis: Redis,
        gpu_semaphore: Semaphore,
        pipeline: StableDiffusionXLControlNetPipeline,
    ):
        super().__init__()

        self.config = config
        self.metagraph = metagraph
        self.verifier = RequestVerifier(hotkey)
        self.whitelist_checker = WhitelistChecker(is_whitelisted_endpoint)
        self.redis = redis
        self.gpu_semaphore = gpu_semaphore
        self.pipeline = pipeline

    async def Generate(self, request: GenerationRequestInputs, context: HandlerCallDetails) -> MinerGenerationResponse:
        invocation_metadata = Metadata.from_tuple(context.invocation_metadata())
        verification_failure = await self.verifier.verify(context, invocation_metadata)

        if verification_failure:
            return verification_failure

        hotkey = invocation_metadata[HOTKEY_HEADER]

        if not self.config.blacklist.allow_non_registered and not await self.whitelist_checker.check(hotkey):
            if hotkey not in self.metagraph.hotkeys:
                # Ignore requests from unrecognized entities.
                bt.logging.trace(f"Blacklisting unrecognized hotkey {hotkey}")

                return await request_error(context, StatusCode.PERMISSION_DENIED, "Unrecognized hotkey")

            uid = self.metagraph.hotkeys.index(hotkey)

            if self.config.blacklist.force_validator_permit and not self.metagraph.validator_permit[uid]:
                bt.logging.trace(f"No validator permit for hotkey {hotkey}")

                return await request_error(context, StatusCode.PERMISSION_DENIED, "No validator permit")

            if self.metagraph.stake[uid] < self.config.blacklist.validator_minimum_tao:
                # Ignore requests from unrecognized entities.
                bt.logging.trace(f"Not enough stake for hotkey {hotkey}")

                return await request_error(context, StatusCode.PERMISSION_DENIED, "Insufficient stake")

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {hotkey}"
        )

        frames = await generate(self.gpu_semaphore, self.pipeline, request)

        generation_id = uuid4()
        frames_hash = sha256(frames).digest()

        await self.redis.set(generation_id.hex, frames, ex=timedelta(minutes=15))

        return MinerGenerationResponse(
            id=MinerGenerationIdentifier(id=generation_id.bytes),
            hash=frames_hash,
        )

    async def Download(self, request: MinerGenerationIdentifier, context) -> MinerGenerationResult:
        return MinerGenerationResult(frames=await self.redis.getdel(UUID(bytes=request.id).hex))

    async def Delete(self, request: MinerGenerationIdentifier, context):
        await self.redis.delete(UUID(bytes=request.id).hex)

        return Empty()


class Miner(BaseNeuron):
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

        bt.logging.info("Loading pipeline")
        self.gpu_semaphore, self.pipeline = get_pipeline(self.device)

        self.pipeline.vae = None

        bt.logging.info("Running warmup for pipeline")
        self.pipeline(
            prompt="Warmup",
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            num_inference_steps=DEFAULT_STEPS,
            image=get_tao_img(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            guidance_scale=DEFAULT_GUIDANCE,
            output_type="latent",
        )

        self.server = grpc.aio.server(interceptors=[LoggingInterceptor()])

        add_MinerServicer_to_server(
            MinerGenerationService(
                self.config,
                self.metagraph,
                self.wallet.hotkey.ss58_address,
                self.is_whitelisted_endpoint,
                self.redis,
                self.gpu_semaphore,
                self.pipeline,
            ),
            self.server
        )

        add_NeuronServicer_to_server(MinerInfoService(), self.server)

        self.server.add_insecure_port(axon_address(self.config.axon))

        self.last_metagraph_sync = self.block

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
            f"Serving miner on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        serve_ip(self.config, self.subtensor, self.wallet)

        # Start the miner's gRPC server, making it active on the network.
        await self.server.start()

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
            await self.server.stop()
            await self.server.wait_for_termination()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as exception:
            bt.logging.error("Caught exception in miner loop", exc_info=exception)

    def sync(self):
        # Ensure miner hotkey is still registered on the network.
        self.check_registered()

        try:
            self.resync_metagraph()
        except Exception as exception:
            bt.logging.error("Failed to resync metagraph", exc_info=exception)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        self.last_metagraph_sync = self.block
