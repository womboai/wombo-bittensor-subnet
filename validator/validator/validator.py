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

import asyncio
import base64
import copy
import os
import random
import time
import traceback
from asyncio import Future, Lock
from typing import AsyncGenerator, Tuple, Any

import bittensor as bt
import heapdict
import torch
from aiohttp import ClientSession, BasicAuth
from bittensor import AxonInfo, TerminalInfo
from substrateinterface import Keypair
from torch import tensor, Tensor

from image_generation_protocol.io_protocol import ImageGenerationInputs
from neuron.neuron import BaseNeuron
from neuron_selector.uids import get_best_uids, sync_neuron_info, DEFAULT_NEURON_INFO
from tensor.config import add_args, check_config
from tensor.protocol import ImageGenerationSynapse, ImageGenerationClientSynapse, NeuronInfoSynapse, \
    MinerGenerationOutput
from tensor.timeouts import CLIENT_REQUEST_TIMEOUT, AXON_REQUEST_TIMEOUT, KEEP_ALIVE_TIMEOUT
from validator.get_base_weights import get_base_weight
from validator.reward import select_endpoint
from validator.watermark import add_watermarks

RANDOM_VALIDATION_CHANCE = float(os.getenv("RANDOM_VALIDATION_CHANCE", str(0.25)))


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


class Validator(BaseNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    spec_version: int = 8
    neuron_info: dict[int, NeuronInfoSynapse]

    pending_requests_lock: Lock
    pending_request_futures: list[Future[None]]

    periodic_validation_queue_lock: Lock
    periodic_validation_queue: dict[int, ImageGenerationInputs]

    def __init__(self):
        super().__init__()

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.periodic_check_dendrite = bt.dendrite(wallet=self.wallet)
        self.forward_dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.periodic_check_dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.base_scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
        self.scores_bonuses = torch.ones_like(self.metagraph.S, dtype=torch.float32)

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

        bt.logging.info("load_state()")
        self.load_state()

        self.pending_requests_lock = Lock()
        self.pending_request_futures = []

        self.periodic_validation_queue_lock = Lock()
        self.periodic_validation_queue = {}

        self.data_endpoint = select_endpoint(
            self.config.data_endpoint,
            self.config.subtensor.network,
            "https://dev-neuron-identifier.api.wombo.ai/api/data",
            "https://neuron-identifier.api.wombo.ai/api/data",
        )

    async def send_metrics(
        self,
        endpoint: str,
        data: Any,
    ):
        if not self.config.send_metrics:
            return

        keypair: Keypair = self.periodic_check_dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"

        bt.logging.info(f"Sending {endpoint} metrics {data}")

        async with ClientSession() as session:
            await session.post(
                    f"{self.data_endpoint}/{endpoint}",
                    auth=BasicAuth(hotkey, signature),
                    json=data,
            )

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
        add_args(parser)

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
            pending_validation_requests = self.pending_request_futures.copy()
            self.pending_request_futures.clear()

        if len(pending_validation_requests):
            pending_validation_requests[0].get_loop().run_until_complete(asyncio.gather(*pending_validation_requests))

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
            # Push new miners to the start of the queue to give them a base score
            if hotkey not in self.miner_heap:
                self.miner_heap[hotkey] = 0

        disconnected_miner_list = [
            hotkey
            for hotkey in self.miner_heap.keys()
            if hotkey not in miners.keys()
        ]

        for hotkey in disconnected_miner_list:
            self.miner_heap.pop(hotkey)

        last_block = max(self.miner_heap.values())

        random_choice_weights = [last_block - block for block in self.miner_heap.values()]

        if sum(random_choice_weights) > 0:
            hotkey = random.choices(
                list(self.miner_heap.keys()),
                weights=random_choice_weights,
            )[0]
        else:
            hotkey = random.choice(list(self.miner_heap.keys()))

        self.miner_heap.pop(hotkey)

        return miners[hotkey], hotkey

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
            hotkey = None

            async with self.periodic_validation_queue_lock:
                if not len(self.periodic_validation_queue):
                    return

                miner_uid, inputs = self.periodic_validation_queue.popitem()
        else:
            miner_uid, hotkey = self.get_next_uid()

            async with self.periodic_validation_queue_lock:
                if miner_uid in self.periodic_validation_queue:
                    inputs = self.periodic_validation_queue.pop(miner_uid)
                else:
                    # TODO Get prompt from prompt bank
                    randomness = base64.b64encode(str(self.step * time.monotonic_ns()).encode("ascii")).decode("ascii")

                    input_parameters = {
                        "prompt": f"Tao, scenic, mountain, night, moon, {randomness}, (deep blue)",
                        "negative_prompt": "blurry, nude, (out of focus), JPEG artifacts",
                        "width": 1024,
                        "height": 1024,
                        "steps": 30,
                    }

                    inputs = ImageGenerationInputs(**input_parameters)

        base_weight = await get_base_weight(
            self,
            miner_uid,
            inputs,
        )

        self.update_base_scores(base_weight, miner_uid)

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

        await sync_neuron_info(self, self.periodic_check_dendrite)

        self.axon.start()

        bt.logging.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                await sync_neuron_info(self, self.periodic_check_dendrite)

                try:
                    await self.check_next_miner()
                except Exception as _:
                    bt.logging.error("Failed to forward to miners, ", traceback.format_exc())

                # Sync metagraph and potentially set weights.
                await self.sync()

                await asyncio.sleep(300)

                self.step += 1

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

        scores = self.base_scores * self.scores_bonuses

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
                self.base_scores[uid] = 0.0
                self.scores_bonuses[uid] = 1.0

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_bases = torch.zeros(self.metagraph.n).to(self.device)
            new_bonuses = torch.ones(self.metagraph.n).to(self.device)

            min_len = min(len(self.hotkeys), len(self.base_scores))

            new_bases[:min_len] = self.base_scores[:min_len]
            new_bonuses[:min_len] = self.scores_bonuses[:min_len]

            self.base_scores = new_bases
            self.scores_bonuses = new_bonuses

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

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

    def update_base_scores(self, score: float, uid: int):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        bt.logging.debug(f"Base score: {score} for UID {uid}")

        alpha: float = self.config.neuron.moving_average_alpha
        self.base_scores[uid] = score * alpha + self.base_scores[uid] * (1 - alpha)

        bt.logging.debug(f"Updated base scores: {self.base_scores}")

    async def update_score_bonuses(self, rewards: Tensor, uids: list[int]):
        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards = self.scores_bonuses.to(self.device).scatter(
            0,
            torch.tensor(uids).to(self.device),
            rewards.to(self.device),
        )
        bt.logging.debug(f"Scattered score bonuses: {rewards}")

        # Update scores with rewards for handling user requests.
        # shape: [ metagraph.n ]
        self.scores_bonuses = scattered_rewards

        await self.send_metrics("bonus_scores", [[uid, rewards[index].item()] for index, uid in enumerate(uids)])

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        torch.save(
            {
                "step": self.step,
                "base_scores": self.base_scores,
                "score_bonuses": self.scores_bonuses,
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
        self.base_scores = state.get("base_scores", self.base_scores)
        self.scores_bonuses = state.get("scores_bonuses", self.scores_bonuses)
        self.hotkeys = state["hotkeys"]
        self.miner_heap = state.get("miner_heap", self.miner_heap)

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
            miner_uids: Tensor,
            axons: list[AxonInfo],
            bad_responses: list[ImageGenerationSynapse],
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
                bad_responses.append(response)
                continue

            working_miner_uids.append(axon_uids[response.axon.hotkey])
            finished_responses.append(response)

        if len(bad_responses):
            bad_axons = [response.axon for response in bad_responses]
            bad_dendrites = [response.dendrite for response in bad_responses]
            bad_miner_uids = [axon_uids[axon.hotkey] for axon in bad_axons]

            # Some failed to response, punish them
            await self.update_score_bonuses(
                self.scores_bonuses[tensor(bad_miner_uids, dtype=torch.int64)] * 0.85,
                bad_miner_uids,
            )

            async with self.periodic_validation_queue_lock:
                self.periodic_validation_queue.update({uid: inputs for uid in bad_miner_uids})

            bt.logging.error(f"Failed to query some miners with {inputs} for axons {bad_axons}, {bad_dendrites}")

        working_miner_tensor = tensor(working_miner_uids, dtype=torch.int64)
        base = self.base_scores[working_miner_tensor]
        bonus = self.scores_bonuses[working_miner_tensor]

        await self.update_score_bonuses((base * bonus + 0.125) / base, working_miner_uids)

        if random.random() >= RANDOM_VALIDATION_CHANCE:
            return

        async with self.periodic_validation_queue_lock:
            self.periodic_validation_queue.update({uid: inputs for uid in working_miner_uids})

    async def forward_image(self, synapse: ImageGenerationClientSynapse) -> ImageGenerationClientSynapse:
        miner_uids = (
            get_best_uids(self.metagraph, self.neuron_info, validators=False)
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

        bad_responses: list[ImageGenerationSynapse] = []

        axon_uids = {
            axon.hotkey: uid.item()
            for uid, axon in zip(miner_uids, axons)
        }

        async for response in response_generator:
            if not response.output:
                bad_responses.append(response)
                continue

            synapse.output = MinerGenerationOutput(
                images=response.output.images,
                process_time=response.dendrite.process_time,
                miner_uid=axon_uids[response.axon.hotkey],
                miner_hotkey=response.axon.hotkey,
            )

            if synapse.watermark:
                synapse.output.images = add_watermarks(synapse.deserialize())

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

        bad_axons = [response.axon for response in bad_responses]
        bad_dendrites = [response.dendrite for response in bad_responses]
        bad_miner_uids = [axon_uids[axon.hotkey] for axon in bad_axons]

        # Some failed to response, punish them
        await self.update_score_bonuses(
            self.scores_bonuses[tensor(bad_miner_uids, dtype=torch.int64)] * 0.85,
            bad_miner_uids,
        )

        raise GetMinerResponseException(bad_dendrites, bad_axons)

    async def blacklist_image(self, synapse: ImageGenerationClientSynapse) -> Tuple[bool, str]:
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
