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
import os
import traceback
from asyncio import Lock, Future
from typing import AsyncGenerator, Tuple

import bittensor as bt
import torch
from PIL.Image import Image
from aiohttp import ClientSession
from bittensor import TerminalInfo, AxonInfo
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from fastapi.security import HTTPBasic
from image_generation_protocol.io_protocol import ImageGenerationInputs
from torch import Tensor, tensor
from transformers import CLIPImageProcessor, CLIPConfig

from gpu_pipeline.pipeline import get_pipeline
from neuron_selector.uids import get_best_uids
from tensor.config import add_args
from tensor.protocol import (
    NeuronInfo, ImageGenerationSynapse, ImageGenerationClientRequest,
    MinerGenerationOutput, NeuronCapability,
)
from tensor.timeouts import KEEP_ALIVE_TIMEOUT, AXON_REQUEST_TIMEOUT, CLIENT_REQUEST_TIMEOUT
from user_requests_validator.miner_metrics import MinerUserRequestMetricManager
from user_requests_validator.similarity_score_pipeline import score_similarity
from user_requests_validator.watermark import add_watermarks
from validator.score_protocol import ScoreOutputSynapse, OutputScoreRequest
from validator.validator import BaseValidator

RANDOM_VALIDATION_CHANCE = float(os.getenv("RANDOM_VALIDATION_CHANCE", str(0.25)))


def validator_forward_info():
    return NeuronInfo(capabilities={NeuronCapability.FORWARDING_VALIDATOR})


class NoMinersAvailableException(Exception):
    def __init__(self, dendrite: TerminalInfo | None):
        super().__init__(f"No miners available for {dendrite} query")
        self.dendrite = dendrite


def query_failure_error_message(
    inputs: ImageGenerationInputs,
    bad_axons: list[TerminalInfo],
    bad_dendrites: list[TerminalInfo],
):
    axon_text = "\n".join([repr(axon) for axon in bad_axons])
    dendrite_text = "\n".join([repr(dendrite) for dendrite in bad_dendrites])

    return (
        f"Failed to query some miners with {repr(inputs)} for:\n"
        f"\taxons: {axon_text}\n"
        f"\tdendrites: {dendrite_text}"
    )


class GetMinerResponseException(Exception):
    def __init__(self, inputs: ImageGenerationInputs, dendrites: list[TerminalInfo], axons: list[TerminalInfo]):
        super().__init__(query_failure_error_message(inputs, axons, dendrites))

        self.dendrites = dendrites
        self.axons = axons


class BadImagesDetected(Exception):
    def __init__(self, inputs: ImageGenerationInputs, dendrite: TerminalInfo, axon: TerminalInfo):
        super().__init__(f"Bad/NSFW images have been detected for inputs: {inputs} with dendrite: {dendrite}")

        self.inputs = inputs
        self.dendrite = dendrite
        self.axon = axon


class UserRequestValidator(BaseValidator):
    axon: bt.axon

    pending_requests_lock: Lock
    pending_request_futures: list[Future[None]]

    security = HTTPBasic()

    def __init__(self):
        super().__init__()

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.metric_manager = MinerUserRequestMetricManager(self)

        # Serve axon to enable external connections.
        self.serve_axon()

        self.axon.attach(forward_fn=validator_forward_info)

        self.axon.attach(
            forward_fn=self.forward_image,
            blacklist_fn=self.blacklist_image,
        )

        self.axon.attach(
            forward_fn=self.score_stress_test_output,
            blacklist_fn=self.blacklist_score_request,
        )

        self.axon.fast_config.timeout_keep_alive = KEEP_ALIVE_TIMEOUT
        self.axon.fast_config.timeout_notify = AXON_REQUEST_TIMEOUT

        bt.logging.info(f"Axon created: {self.axon}")

        self.neuron_info = {}

        self.last_neuron_info_block = self.block
        self.last_metagraph_sync = self.block

        self.pending_requests_lock = Lock()
        self.pending_request_futures = []

        self.gpu_semaphore, self.pipeline = get_pipeline(self.device)

        self.image_processor = self.pipeline.feature_extractor or CLIPImageProcessor()
        self.safety_checker = StableDiffusionSafetyChecker(CLIPConfig()).to(self.device)

    async def score_stress_test_output(self, request: OutputScoreRequest) -> float:
        return await score_similarity(
            self.gpu_semaphore,
            self.pipeline,
            request.frames,
            request.inputs,
        )

    def blacklist_score_request(self, synapse: ScoreOutputSynapse) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey != self.wallet.hotkey.ss58_address:
            return True, "Mismatching hotkey"

        return False, "Correct hotkey"

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass

    async def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        await self.sync_neuron_info()

        self.axon.start()

        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"block({self.block})")

                try:
                    neuron_refresh_blocks = 25

                    blocks_since_neuron_refresh = self.block - self.last_neuron_info_block
                    blocks_since_sync = self.block - self.last_metagraph_sync

                    sleep = True

                    if blocks_since_neuron_refresh > neuron_refresh_blocks:
                        await self.sync_neuron_info()
                        sleep = False

                    if blocks_since_sync > self.config.neuron.epoch_length:
                        self.metagraph.sync(subtensor=self.subtensor)
                        self.last_metagraph_sync = self.block
                        sleep = False

                    if sleep:
                        neuron_refresh_in = neuron_refresh_blocks - blocks_since_neuron_refresh
                        sync_in = self.config.neuron.epoch_length - blocks_since_sync

                        await asyncio.sleep(max(min(neuron_refresh_in, sync_in), 1) * 12)

                    async with self.pending_requests_lock:
                        remaining_futures = []
                        for future in self.pending_request_futures:
                            if not future.done():
                                remaining_futures.append(future)
                                continue
                            try:
                                future.result()
                            except Exception as e:
                                error_traceback = traceback.format_exc()
                                bt.logging.error(f"Error in validation coroutine: {e}\n{error_traceback}")

                        self.pending_request_futures = remaining_futures
                except Exception as _:
                    bt.logging.error("Failed to forward to miners, ", traceback.format_exc())

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(
                traceback.print_exception(type(err), err, err.__traceback__)
            )

    @classmethod
    def add_args(cls, parser):
        add_args(parser, "cuda")

        super().add_args(parser)

    async def get_forward_responses(
        self,
        axons: list[AxonInfo],
        synapse: ImageGenerationSynapse,
    ) -> AsyncGenerator[ImageGenerationSynapse, None]:
        responses = asyncio.as_completed(
            [
                self.dendrite(
                    axons=axon,
                    synapse=synapse,
                    deserialize=False,
                    timeout=CLIENT_REQUEST_TIMEOUT,
                )
                for axon in axons
            ]
        )

        for response in responses:
            yield await response

    async def validate_user_request_responses(
        self,
        inputs: ImageGenerationInputs,
        finished_response: ImageGenerationSynapse,
        miner_uids: Tensor,
        axons: list[AxonInfo],
        bad_responses: list[tuple[ImageGenerationSynapse, float | None]],
        response_generator: AsyncGenerator[ImageGenerationSynapse, None],
    ):
        axon_uids = {
            axon.hotkey: uid.item()
            for uid, axon in zip(miner_uids, axons)
        }

        working_miner_uids: list[int] = [axon_uids[finished_response.axon.hotkey]]
        finished_responses: list[ImageGenerationSynapse] = [finished_response]

        async for response in response_generator:
            if not response.output:
                bad_responses.append((response, None))
                continue

            similarity_score = await self.score_output(inputs, response)

            if similarity_score < 0.85:
                bad_responses.append((response, similarity_score))
                continue

            working_miner_uids.append(axon_uids[response.axon.hotkey])
            finished_responses.append(response)

        if len(bad_responses):
            bad_axons = [(response.axon, similarity_score) for response, similarity_score in bad_responses]
            bad_dendrites = [response.dendrite for response, _ in bad_responses]
            bad_miner_uids = [(axon_uids[axon.hotkey], similarity_score) for axon, similarity_score in bad_axons]

            # Some failed to response, punish them
            await asyncio.gather(
                *[
                    self.metric_manager.failed_user_request(uid, similarity_score)
                    for uid, similarity_score in bad_miner_uids
                ]
            )

            await self.redis.sadd("stress_test_queue", *[uid for uid, _ in bad_miner_uids])

            bt.logging.info(
                query_failure_error_message(
                    inputs,
                    [axon for axon, _ in bad_axons],
                    bad_dendrites,
                )
            )

        async def rank_response(uid: int, uid_response: ImageGenerationSynapse):
            score = await self.score_output(inputs, uid_response)
            await self.metric_manager.successful_user_request(uid, score)

        # Some failed to response, punish them
        await asyncio.gather(
            *[
                rank_response(uid, response)
                for response, uid in zip(finished_responses, working_miner_uids)
            ]
        )

        if (os.urandom(1)[0] / 255) >= RANDOM_VALIDATION_CHANCE:
            return

        await self.redis.sadd("stress_test_queue", *working_miner_uids)

    async def score_output(self, inputs: ImageGenerationInputs, response: ImageGenerationSynapse):
        return await score_similarity(
            self.gpu_semaphore,
            self.pipeline,
            response,
            inputs,
        )

    def is_unsafe_image(self, image: Image) -> bool:
        safety_checker_input = self.image_processor(image, return_tensors="pt")

        with self.gpu_semaphore:
            safety_checker_input = safety_checker_input.to(self.device)

            _, has_nsfw_concept = self.safety_checker(
                images=[image],
                clip_input=safety_checker_input.pixel_values.to(torch.float16),
            )

        return has_nsfw_concept[0]

    async def forward_image(self, synapse: ImageGenerationClientRequest) -> ImageGenerationClientRequest:
        miner_uids = (
            get_best_uids(
                self.config.blacklist,
                self.metagraph,
                self.neuron_info,
                (await self.metric_manager.get_rps()).nan_to_num(0.0),
                lambda _, info: info.is_validator is False,
            )
            if synapse.miner_uid is None
            else tensor([synapse.miner_uid])
        )

        if not len(miner_uids):
            raise NoMinersAvailableException(synapse.dendrite)

        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        response_generator = self.get_forward_responses(
            axons,
            ImageGenerationSynapse(inputs=synapse.inputs),
        )

        bad_responses: list[tuple[ImageGenerationSynapse, float | None]] = []

        axon_uids = {
            axon.hotkey: uid.item()
            for uid, axon in zip(miner_uids, axons)
        }

        async for response in response_generator:
            if not response.output:
                bad_responses.append((response, None))
                continue

            similarity_score = await self.score_output(synapse.inputs, response)

            if similarity_score < 0.85:
                bad_responses.append((response, similarity_score))
                continue

            synapse.output = MinerGenerationOutput(
                images=response.output.images,
                process_time=response.dendrite.process_time,
                miner_uid=axon_uids[response.axon.hotkey],
                miner_hotkey=response.axon.hotkey,
            )

            images = synapse.deserialize()

            if any([self.is_unsafe_image(image) for image in images]):
                raise BadImagesDetected(synapse.inputs, response.dendrite, response.axon)

            if synapse.watermark:
                synapse.output.images = add_watermarks(images)

            validation_coroutine = self.validate_user_request_responses(
                synapse.inputs,
                response,
                miner_uids,
                axons,
                bad_responses,
                response_generator,
            )

            async with self.pending_requests_lock:
                self.pending_request_futures.append(asyncio.ensure_future(validation_coroutine))

            return synapse

        bad_axons = [(response.axon, similarity_score) for response, similarity_score in bad_responses]
        bad_dendrites = [response.dendrite for response, _ in bad_responses]
        bad_miner_uids = [(axon_uids[axon.hotkey], similarity_score) for axon, similarity_score in bad_axons]

        # All failed to response, punish them
        await asyncio.gather(
            *[
                self.metric_manager.failed_user_request(uid, similarity_score)
                for uid, similarity_score in bad_miner_uids
            ]
        )

        raise GetMinerResponseException(synapse.inputs, bad_dendrites, [axon for axon, _ in bad_axons])

    async def blacklist_image(self, synapse: ImageGenerationClientRequest) -> Tuple[bool, str]:
        if not self.session:
            self.session = ClientSession()

        async with self.session.get(
            f"{self.is_whitelisted_endpoint}?hotkey={synapse.dendrite.hotkey}",
            headers={"Content-Type": "application/json"},
        ) as response:
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
