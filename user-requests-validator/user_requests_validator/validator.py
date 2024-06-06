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
import sys
import traceback
from asyncio import Semaphore, Lock
from io import BytesIO
from typing import AsyncGenerator, TypeAlias

import bittensor as bt
import grpc
import torch
from bittensor import AxonInfo
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from fastapi.security import HTTPBasic
from google.protobuf.empty_pb2 import Empty
from grpc import StatusCode, HandlerCallDetails
from grpc.aio import Channel, Metadata
from torch import Tensor, tensor
from transformers import CLIPConfig

from base_validator.protos.scoring_pb2 import OutputScoreRequest
from base_validator.protos.scoring_pb2_grpc import OutputScorerServicer, add_OutputScorerServicer_to_server
from base_validator.validator import (
    BaseValidator,
    get_miner_response,
    SuccessfulGenerationResponseInfo, is_cheater,
)
from gpu_pipeline.pipeline import get_pipeline
from gpu_pipeline.tensor import load_tensor
from neuron.api_handler import HOTKEY_HEADER, request_error, RequestVerifier, serve_ip, WhitelistChecker
from neuron.protos.neuron_pb2 import MinerGenerationResponse, MinerGenerationResult
from neuron.protos.neuron_pb2_grpc import MinerStub
from neuron_selector.protos.forwarding_validator_pb2 import ValidatorUserRequest, ValidatorGenerationResponse
from neuron_selector.protos.forwarding_validator_pb2_grpc import (
    ForwardingValidatorServicer,
    add_ForwardingValidatorServicer_to_server,
)
from neuron_selector.uids import get_best_uids
from tensor.config import add_args, SPEC_VERSION
from tensor.input_sanitization import sanitize_inputs
from tensor.interceptors import LoggingInterceptor
from tensor.protos.inputs_pb2 import GenerationRequestInputs, InfoResponse, NeuronCapabilities
from tensor.protos.inputs_pb2_grpc import NeuronServicer, add_NeuronServicer_to_server
from tensor.response import (
    Response, axon_address, axon_channel, Channels, FailedResponseInfo, SuccessfulResponse,
    call_request,
)
from user_requests_validator.miner_metrics import MinerUserRequestMetricManager
from user_requests_validator.similarity_score_pipeline import score_similarity
from user_requests_validator.watermark import apply_watermark

RANDOM_VALIDATION_CHANCE = float(os.getenv("RANDOM_VALIDATION_CHANCE", str(0.35)))

MinerResponseFailureInfo: TypeAlias = SuccessfulGenerationResponseInfo | FailedResponseInfo


def _random():
    return os.urandom(1)[0] / 255


async def get_miner_response_with_channel(
    inputs: GenerationRequestInputs,
    axon: AxonInfo,
    channel: Channel,
    wallet: bt.wallet,
):
    return await get_miner_response(inputs, axon, channel, wallet), channel


async def get_forward_responses(
    channels: list[tuple[Channel, AxonInfo]],
    inputs: GenerationRequestInputs,
    wallet: bt.wallet,
) -> AsyncGenerator[tuple[Response[MinerGenerationResponse], Channel], None]:
    responses = asyncio.as_completed(
        [
            get_miner_response_with_channel(inputs, axon, channel, wallet)
            for channel, axon in channels
        ]
    )

    for response, channel in responses:
        yield await response, channel


class ValidatorInfoService(NeuronServicer):
    def Info(self, request: Empty, context: HandlerCallDetails):
        return InfoResponse(
            spec_version=SPEC_VERSION,
            capabilities=[NeuronCapabilities.MINER]
        )


class OutputScoreService(OutputScorerServicer):
    def __init__(
        self,
        hotkey: str,
        gpu_semaphore: Semaphore,
        pipeline: StableDiffusionXLControlNetPipeline,
    ):
        super().__init__()

        self.hotkey = hotkey
        self.verifier = RequestVerifier(hotkey)
        self.gpu_semaphore = gpu_semaphore
        self.pipeline = pipeline

    async def ScoreOutput(self, request: OutputScoreRequest, context: HandlerCallDetails):
        invocation_metadata = Metadata.from_tuple(context.invocation_metadata())
        verification_failure = await self.verifier.verify(context, invocation_metadata)

        if verification_failure:
            return verification_failure

        hotkey = invocation_metadata[HOTKEY_HEADER]

        if hotkey != self.hotkey:
            return True, "Mismatching hotkey"

        async with self.gpu_semaphore:
            return await score_similarity(
                self.pipeline,
                load_tensor(request.frames),
                request.inputs,
            )


class ValidatorGenerationService(ForwardingValidatorServicer):
    def __init__(
        self,
        validator: "UserRequestValidator",
        gpu_semaphore: Semaphore,
        pipeline: StableDiffusionXLControlNetPipeline,
    ):
        super().__init__()

        self.validator = validator
        self.whitelist_checker = WhitelistChecker(validator.is_whitelisted_endpoint)
        self.verifier = RequestVerifier(validator.wallet.hotkey.ss58_address)
        self.gpu_semaphore = gpu_semaphore
        self.pipeline = pipeline

        self.safety_checker = StableDiffusionSafetyChecker(CLIPConfig()).to(validator.device)

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.metric_manager = MinerUserRequestMetricManager(validator)
        self.redis = validator.redis

    async def Generate(self, request: ValidatorUserRequest, context: HandlerCallDetails):
        invocation_metadata = Metadata.from_tuple(context.invocation_metadata())
        verification_failure = await self.verifier.verify(context, invocation_metadata)

        if verification_failure:
            return verification_failure

        hotkey = invocation_metadata[HOTKEY_HEADER]

        if not await self.whitelist_checker.check(hotkey):
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {hotkey}"
            )

            return await request_error(context, StatusCode.PERMISSION_DENIED, "Unrecognized hotkey")

        sanitize_inputs(request.inputs)

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {hotkey}"
        )

        miner_uids = (
            get_best_uids(
                self.validator.config.blacklist,
                self.validator.metagraph,
                self.validator.neuron_info,
                (await self.metric_manager.get_rps()).nan_to_num(0.0),
                lambda _, info: NeuronCapabilities.MINER in info.capabilities,
            )
            if request.miner_uid is None
            else tensor([request.miner_uid])
        )

        if not len(miner_uids):
            raise NoMinersAvailableException(request, hotkey)

        axons = [self.validator.metagraph.axons[uid] for uid in miner_uids]

        channels = Channels([axon_channel(axon) for axon in axons])

        try:
            response_generator = get_forward_responses(
                zip(channels.channels, axons),
                request.inputs,
                self.validator.wallet,
            )

            bad_responses: list[MinerResponseFailureInfo] = []

            axon_uids = {
                axon.hotkey: uid.item()
                for uid, axon in zip(miner_uids, axons)
            }

            async for response, channel in response_generator:
                if not response.successful:
                    bad_responses.append(response.info())
                    continue

                download_result: Response[MinerGenerationResult] = (
                    await call_request(response.axon, response.data.id, MinerStub(channel).Download)
                )

                if is_cheater(axon_uids[response.axon.hotkey], download_result.data.frames, response.data.hash):
                    bad_responses.append(
                        SuccessfulGenerationResponseInfo.of(
                            response.info,
                            0.0,
                            True,
                        )
                    )

                    continue

                async with self.gpu_semaphore:
                    frames_tensor = load_tensor(download_result.data.frames)

                    if _random() < RANDOM_VALIDATION_CHANCE:
                        similarity_score = await score_similarity(
                            self.pipeline,
                            frames_tensor,
                            request.inputs,
                        )

                        if similarity_score < 0.85:
                            bad_responses.append(
                                SuccessfulGenerationResponseInfo.of(
                                    response.info,
                                    similarity_score,
                                    False,
                                )
                            )

                            continue

                    latents = frames_tensor[-1].to(self.pipeline.unet.device, self.pipeline.unet.dtype)

                    # make sure the VAE is in float32 mode, as it overflows in float16
                    needs_upcasting = self.pipeline.vae.dtype == torch.float16 and self.pipeline.vae.config.force_upcast

                    if needs_upcasting:
                        self.pipeline.upcast_vae()
                        latents = latents.to(next(iter(self.pipeline.vae.post_quant_conv.parameters())).dtype)

                    image = self.pipeline.vae.decode(
                        latents / self.pipeline.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]

                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.pipeline.vae.to(dtype=torch.float16)

                    pt_image = self.pipeline.image_processor(image, return_tensors="pt")

                    _, has_nsfw_concept = self.safety_checker(
                        images=[image],
                        clip_input=pt_image.pixel_values.to(torch.float16),
                    )

                    if has_nsfw_concept[0]:
                        raise BadImagesDetected(request.inputs, response.axon)

                    nd_image = self.pipeline.image_processor.pt_to_numpy(pt_image)
                    image = self.pipeline.image_processor.numpy_to_pil(nd_image)

                if request.watermark:
                    image = apply_watermark(image)

                with BytesIO() as buf:
                    image.save(buf)
                    image_bytes = buf.getvalue()

                validation_coroutine = self.validate_user_request_responses(
                    request.inputs,
                    response,
                    miner_uids,
                    axons,
                    bad_responses,
                    response_generator,
                    channels,
                )

                async with self.validator.pending_requests_lock:
                    self.validator.pending_request_futures.append(asyncio.ensure_future(validation_coroutine))

                return ValidatorGenerationResponse(
                    image=image_bytes,
                    miner_uid=axon_uids[response.axon.hotkey],
                    generation_time=response.process_time,
                )
        except:
            channels.__aexit__(*sys.exc_info())

            raise

        async with channels:
            # All failed to response, punish them
            await asyncio.gather(
                *[
                    self.metric_manager.failed_user_request(
                        axon_uids[info.axon.hotkey],
                        info.similarity_score,
                        info.cheater,
                    )
                    if isinstance(info, SuccessfulGenerationResponseInfo)
                    else self.metric_manager.failed_user_request(axon_uids[info.axon.hotkey])
                    for info in bad_responses
                ]
            )

            raise GetMinerResponseException(request.inputs, bad_responses)

    async def validate_user_request_responses(
        self,
        inputs: GenerationRequestInputs,
        finished_response: SuccessfulResponse[MinerGenerationResponse],
        miner_uids: Tensor,
        axons: list[AxonInfo],
        bad_responses: list[MinerResponseFailureInfo],
        response_generator: AsyncGenerator[tuple[Response[MinerGenerationResponse], Channel], None],
        channels: Channels,
    ):
        axon_uids = {
            axon.hotkey: uid.item()
            for uid, axon in zip(miner_uids, axons)
        }

        working_miner_uids: list[int] = [axon_uids[finished_response.axon.hotkey]]
        finished_responses: list[SuccessfulResponse[MinerGenerationResponse]] = [finished_response]

        async with channels:
            async for response, channel in response_generator:
                if not response.successful:
                    bad_responses.append(response.info)
                    continue

                if _random() < RANDOM_VALIDATION_CHANCE:
                    download_result = await call_request(response.axon, response.data.id, MinerStub(channel).Download)

                    if download_result.successful:
                        if is_cheater(axon_uids[response.axon.hotkey], download_result.data.frames, response.data.hash):
                            bad_responses.append(
                                SuccessfulGenerationResponseInfo.of(
                                    response.info,
                                    0.0,
                                    True,
                                )
                            )

                            continue

                        async with self.gpu_semaphore:
                            similarity_score = await score_similarity(
                                self.pipeline,
                                load_tensor(download_result.data.frames),
                                inputs,
                            )
                    else:
                        similarity_score = 0.0

                    if similarity_score < 0.85:
                        bad_responses.append(
                            SuccessfulGenerationResponseInfo.of(
                                response.info,
                                similarity_score,
                                False,
                            )
                        )
                        continue
                else:
                    await call_request(response.axon, response.data.id, MinerStub(channel).Delete)

                working_miner_uids.append(axon_uids[response.axon.hotkey])
                finished_responses.append(response)

            if len(bad_responses):
                # Some failed to response, punish them
                await asyncio.gather(
                    *[
                        self.metric_manager.failed_user_request(
                            axon_uids[info.axon.hotkey],
                            info.similarity_score,
                            info.cheater,
                        )
                        if isinstance(info, SuccessfulGenerationResponseInfo)
                        else self.metric_manager.failed_user_request(axon_uids[info.axon.hotkey])
                        for info in bad_responses
                    ]
                )

                await self.redis.sadd(
                    "stress_test_queue", *[axon_uids[response.axon.hotkey] for response in bad_responses]
                )

                bt.logging.info(query_failure_error_message(inputs, bad_responses))

        async def rank_response(uid: int, frames: bytes):
            async with self.gpu_semaphore:
                score = await score_similarity(
                    self.pipeline,
                    load_tensor(frames),
                    inputs,
                )

            await self.metric_manager.successful_user_request(uid, score)

        # Some failed to response, punish them
        await asyncio.gather(
            *[
                rank_response(uid, response.frames)
                for response, uid in zip(finished_responses, working_miner_uids)
            ]
        )

        if _random() < RANDOM_VALIDATION_CHANCE:
            return

        await self.redis.sadd("stress_test_queue", *working_miner_uids)


def validator_forward_info():
    return InfoResponse(capabilities={NeuronCapabilities.FORWARDING_VALIDATOR})


class NoMinersAvailableException(Exception):
    def __init__(self, inputs: ValidatorUserRequest, hotkey: str):
        super().__init__(f"No miners available for {hotkey}'s query, input: {inputs}")

        self.inputs = inputs
        self.hotkey = hotkey


def query_failure_error_message(
    inputs: GenerationRequestInputs,
    responses: list[MinerResponseFailureInfo],
):
    response_text = "\n\t".join([repr(response) for response in responses])

    return (
        f"Failed to query some miners with {repr(inputs)}:\n"
        f"\t{response_text}\n"
    )


class GetMinerResponseException(Exception):
    def __init__(self, inputs: GenerationRequestInputs, responses: list[MinerResponseFailureInfo]):
        super().__init__(query_failure_error_message(inputs, responses))

        self.responses = responses


class BadImagesDetected(Exception):
    def __init__(self, inputs: GenerationRequestInputs, axon: AxonInfo):
        super().__init__(f"Bad/NSFW images have been detected for inputs: {inputs} with axon: {axon}")

        self.inputs = inputs
        self.axon = axon


class UserRequestValidator(BaseValidator):
    axon: bt.axon

    security = HTTPBasic()

    def __init__(self):
        super().__init__()

        self.neuron_info = {}

        self.last_neuron_info_block = self.block
        self.last_metagraph_sync = self.block

        gpu_semaphore, pipeline = get_pipeline(self.device)

        self.server = grpc.aio.server(interceptors=[LoggingInterceptor()])

        add_ForwardingValidatorServicer_to_server(
            ValidatorGenerationService(
                self,
                gpu_semaphore,
                pipeline,
            ),
            self.server,
        )

        add_OutputScorerServicer_to_server(
            OutputScoreService(
                self.wallet.hotkey.ss58_address,
                gpu_semaphore,
                pipeline
            ),
            self.server,
        )

        add_NeuronServicer_to_server(ValidatorInfoService(), self.server)

        self.server.add_insecure_port(axon_address(self.config.axon))

        self.pending_requests_lock = Lock()
        self.pending_request_futures = []

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

        # Serve IP to enable external connections.
        serve_ip(self.config, self.subtensor, self.wallet)

        await self.server.start()

        bt.logging.info(
            f"Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
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
                except Exception as exception:
                    bt.logging.error("Failed to forward to miners", exc_info=exception)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            await self.server.stop()
            await self.server.wait_for_termination()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", exc_info=err)

    @classmethod
    def add_args(cls, parser):
        add_args(parser)

        super().add_args(parser)
