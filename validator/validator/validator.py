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
import copy
import os
import random
import traceback
from asyncio import Future, Lock
from threading import Semaphore
from typing import AsyncGenerator, Tuple

import bittensor as bt
import heapdict
import torch
from PIL.Image import Image
from aiohttp import ClientSession
from bittensor import AxonInfo, TerminalInfo
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch import tensor, Tensor
from transformers import CLIPConfig, CLIPImageProcessor

from gpu_pipeline.pipeline import get_pipeline
from image_generation_protocol.io_protocol import ImageGenerationInputs
from neuron.neuron import BaseNeuron
from neuron_selector.uids import get_best_uids, sync_neuron_info, DEFAULT_NEURON_INFO, weighted_sample
from tensor.config import add_args, check_config
from tensor.protocol import (
    ImageGenerationSynapse, ImageGenerationClientSynapse, NeuronInfoSynapse,
    MinerGenerationOutput,
)
from tensor.timeouts import CLIENT_REQUEST_TIMEOUT, AXON_REQUEST_TIMEOUT, KEEP_ALIVE_TIMEOUT
from validator.miner_metrics import MinerMetricManager, set_miner_metrics
from validator.reward import select_endpoint, reward
from validator.watermark import add_watermarks

RANDOM_VALIDATION_CHANCE = float(os.getenv("RANDOM_VALIDATION_CHANCE", str(0.25)))


def validator_forward_info(synapse: NeuronInfoSynapse):
    synapse.is_validator = True

    return synapse


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


class Validator(BaseNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    spec_version: int = 19
    neuron_info: dict[int, NeuronInfoSynapse]

    pending_requests_lock: Lock
    pending_request_futures: list[Future[None]]

    periodic_validation_queue_lock: Lock
    periodic_validation_queue: set[int]

    def __init__(self):
        super().__init__()

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.periodic_check_dendrite = bt.dendrite(wallet=self.wallet)
        self.forward_dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.periodic_check_dendrite}")

        self.stress_test_session = None
        self.user_request_session = None

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.metric_manager = MinerMetricManager(self)

        self.step = 0

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Serve axon to enable external connections.
        self.serve_axon()

        self.miner_heap = heapdict.heapdict()

        self.axon.attach(forward_fn=validator_forward_info)

        self.axon.attach(
            forward_fn=self.forward_image,
            blacklist_fn=self.blacklist_image,
        )

        self.axon.fast_config.timeout_keep_alive = KEEP_ALIVE_TIMEOUT
        self.axon.fast_config.timeout_notify = AXON_REQUEST_TIMEOUT

        bt.logging.info(f"Axon created: {self.axon}")

        self.neuron_info = {}

        self.last_neuron_info_block = self.block
        self.last_miner_check = self.block

        bt.logging.info("load_state()")
        self.load_state()

        self.pending_requests_lock = Lock()
        self.pending_request_futures = []

        self.periodic_validation_queue_lock = Lock()
        self.periodic_validation_queue = set()

        concurrency, self.pipeline = get_pipeline(self.device)
        self.gpu_semaphore = Semaphore(concurrency)

        self.image_processor = self.pipeline.feature_extractor or CLIPImageProcessor()
        self.safety_checker = StableDiffusionSafetyChecker(CLIPConfig()).to(self.device)

    @classmethod
    def check_config(cls, config: bt.config):
        check_config(config, "validator")

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

    @classmethod
    def add_args(cls, parser):
        add_args(parser, "cuda")

        parser.add_argument(
            "--validation_endpoint",
            type=str,
            help="The endpoint to call for validator scoring",
            default="",
        )

        parser.add_argument(
            "--is_hotkey_allowed_endpoint",
            type=str,
            help="The endpoint called when checking if the hotkey is accepted by validators",
            default="",
        )

        parser.add_argument(
            "--data_endpoint",
            type=str,
            help="The endpoint to send metrics to if enabled",
            default="",
        )

        parser.set_defaults(send_metrics=True)
        parser.add_argument(
            "--no_metrics",
            action="store_false",
            dest="send_metrics",
            help="Disables sending metrics.",
        )

        parser.add_argument(
            "--blacklist.hotkeys",
            action='append',
            help="The hotkeys to block when sending requests",
            default=[],
        )

        parser.add_argument(
            "--blacklist.coldkeys",
            action='append',
            help="The coldkeys to block when sending requests",
            default=["5DhPDjLR4YNAixDLNFNP2pTiCpkDQ5A5vm5fyQ3Q52rYcEaw"],
        )

        parser.add_argument(
            "--neuron.disable_set_weights",
            action="store_true",
            help="Disables setting weights.",
            default=False,
        )

        parser.add_argument(
            "--neuron.moving_average_alpha",
            type=float,
            help="Moving average alpha parameter, how much to add of the new observation.",
            default=0.5,
        )

    async def sync_neuron_info(self):
        await sync_neuron_info(self, self.periodic_check_dendrite)

        self.last_neuron_info_block = self.block

    async def sync(self):
        await super().sync()

        try:
            if self.should_set_weights():
                self.set_weights()
        except Exception as _:
            bt.logging.error("Failed to set validator weights, ", traceback.format_exc())

        # Always save state.
        self.save_state()

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

    def get_next_uid(self) -> tuple[int, str]:
        miners = {
            self.metagraph.axons[uid].hotkey: uid
            for uid in range(self.metagraph.n.item())
            if self.metagraph.axons[uid].is_serving
        }

        infos = {
            uid: self.neuron_info.get(uid, DEFAULT_NEURON_INFO)
            for uid in miners.values()
        }

        bt.logging.info(f"Neuron info found: {infos}")

        invalid_miner_list = [
            hotkey
            for hotkey, uid in miners.items()
            if infos[uid].is_validator is not False
        ]

        for hotkey in invalid_miner_list:
            miners.pop(hotkey)

        for hotkey in miners.keys():
            if hotkey in self.miner_heap:
                continue

            # Push new miners to be near the start of the queue to give them a base score
            value = random.random()

            block_count = max(self.miner_heap.values()) - min(self.miner_heap.values())
            self.miner_heap[hotkey] = int(block_count * value * value * 0.25)

        disconnected_miner_list = [
            hotkey
            for hotkey in self.miner_heap.keys()
            if hotkey not in miners.keys()
        ]

        for hotkey in disconnected_miner_list:
            self.miner_heap.pop(hotkey)

        last_block = max(self.miner_heap.values())

        weighted_choices = [(last_block - block, hotkey) for hotkey, block in self.miner_heap.items()]

        if sum([block for block, _ in weighted_choices]) > 0:
            hotkey = weighted_sample(weighted_choices, k=1)[0]
        else:
            hotkey = random.choice(list(self.miner_heap.keys()))

        self.miner_heap.pop(hotkey)

        return miners[hotkey], hotkey

    async def reset_idle_miner_incentives(self):
        for uid, info in self.neuron_info.items():
            if info.is_validator is False:
                # Working miner, skip
                continue

            self.metric_manager.reset(uid)

    async def check_next_miner(self):
        """
        Validator forward pass, called by the validator every time step. Consists of:
        - Generating the query
        - Querying the network miners
        - Getting the responses
        - Rewarding the miners based on their responses
        - Updating the scores
        """

        if self.step % 2 == 0:
            async with self.periodic_validation_queue_lock:
                if len(self.periodic_validation_queue):
                    hotkey = None
                    miner_uid = self.periodic_validation_queue.pop()
                else:
                    miner_uid, hotkey = self.get_next_uid()
        else:
            miner_uid, hotkey = self.get_next_uid()

            async with self.periodic_validation_queue_lock:
                if miner_uid in self.periodic_validation_queue:
                    self.periodic_validation_queue.remove(miner_uid)

        await set_miner_metrics(
            self,
            miner_uid,
        )

        self.last_miner_check = self.block

        if hotkey:
            # Checked miners(with base scores) are pushed to the end of the queue
            # to leave room for ones that have not been checked in a while
            self.miner_heap[hotkey] = self.block

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
                bt.logging.info(f"step({self.step}) block({self.block})")

                try:
                    neuron_refresh_blocks = 25
                    check_blocks = 5

                    blocks_since_neuron_refresh = self.block - self.last_neuron_info_block
                    blocks_since_check = self.block - self.last_miner_check

                    sleep = True

                    if blocks_since_neuron_refresh > neuron_refresh_blocks:
                        await self.sync_neuron_info()
                        sleep = False

                    if blocks_since_check > check_blocks:
                        await self.reset_idle_miner_incentives()
                        await self.check_next_miner()

                        sleep = False

                    # Sync metagraph and potentially set weights.
                    await self.sync()

                    if sleep:
                        neuron_refresh_in = neuron_refresh_blocks - blocks_since_neuron_refresh
                        check_in = check_blocks - blocks_since_check

                        await asyncio.sleep(max(min(neuron_refresh_in, check_in), 1) * 12)

                    self.step += 1
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

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        metrics = [self.metric_manager[uid] for uid in range(self.metagraph.n.item())]

        scores = tensor(
            [
                miner_metrics.get_weight() if miner_metrics else 0.0
                for miner_metrics in metrics
            ]
        )

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = torch.nn.functional.normalize(
            scores, p=1, dim=0
        )

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", self.metagraph.uids.to("cpu"))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids.to("cpu"),
            weights=raw_weights.to("cpu"),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, message = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )

        if result:
            bt.logging.info(f"set_weights on chain successfully! {message}")
        else:
            bt.logging.error(f"set_weights failed. {message}")

    async def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )

        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                # hotkey has been replaced
                self.metric_manager.reset(uid)

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            self.metric_manager.resize()

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Define appropriate logic for when set weights.
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        torch.save(
            {
                "step": self.step,
                "miner_data": self.metric_manager.miner_data,
                "hotkeys": self.hotkeys,
                "miner_heap": self.miner_heap,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        path = self.config.neuron.full_path + "/state.pt"

        if not os.path.isfile(path):
            return

        # Load the state of the validator from file.
        state = torch.load(path)
        self.step = state["step"]
        self.metric_manager.load_data(state.get("miner_data", self.metric_manager.miner_data))
        self.hotkeys = state["hotkeys"]
        self.miner_heap = state.get("miner_heap", self.miner_heap)

    async def get_forward_responses(
        self,
        axons: list[AxonInfo],
        synapse: ImageGenerationSynapse,
    ) -> AsyncGenerator[ImageGenerationSynapse, None]:
        responses = asyncio.as_completed(
            [
                self.forward_dendrite(
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

            similarity_score = self.score_output(inputs, response)

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

            async with self.periodic_validation_queue_lock:
                self.periodic_validation_queue.update({uid for uid, _ in bad_miner_uids})

            bt.logging.info(
                query_failure_error_message(
                    inputs,
                    [axon for axon, _ in bad_axons],
                    bad_dendrites,
                )
            )

        async def rank_response(uid: int, uid_response: ImageGenerationSynapse):
            score = self.score_output(inputs, uid_response)
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

        async with self.periodic_validation_queue_lock:
            self.periodic_validation_queue.update({uid for uid in working_miner_uids})

    def score_output(self, inputs: ImageGenerationInputs, response: ImageGenerationSynapse):
        return reward(
            self.gpu_semaphore,
            self.pipeline,
            inputs,
            response,
        )

    def is_unsafe_image(self, image: Image) -> bool:
        safety_checker_input = self.image_processor(image, return_tensors="pt").to(self.device)

        _, has_nsfw_concept = self.safety_checker(
            images=[image],
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )

        return has_nsfw_concept[0]

    async def forward_image(self, synapse: ImageGenerationClientSynapse) -> ImageGenerationClientSynapse:
        miner_uids = (
            get_best_uids(
                self.config.blacklist,
                self.metagraph,
                self.neuron_info,
                (
                    self.metric_manager.miner_data.generation_counts / self.metric_manager.miner_data.generation_times).nan_to_num(
                    0.0
                ),
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

            similarity_score = self.score_output(synapse.inputs, response)

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

    async def blacklist_image(self, synapse: ImageGenerationClientSynapse) -> Tuple[bool, str]:
        is_hotkey_allowed_endpoint = select_endpoint(
            self.config.is_hotkey_allowed_endpoint,
            self.config.subtensor.network,
            "https://dev-neuron-identifier.api.wombo.ai/api/is_hotkey_allowed",
            "https://neuron-identifier.api.wombo.ai/api/is_hotkey_allowed",
        )

        if not self.user_request_session:
            self.user_request_session = ClientSession()

        async with self.user_request_session.get(
            f"{is_hotkey_allowed_endpoint}?hotkey={synapse.dendrite.hotkey}",
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
