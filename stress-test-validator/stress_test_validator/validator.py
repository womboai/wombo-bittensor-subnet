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
import pickle
import random

import bittensor as bt
import numpy
from bittensor.utils.weight_utils import process_weights_for_netuid, convert_weights_and_uids_for_emit
from heapdict import heapdict

from base_validator.validator import BaseValidator
from stress_test_validator.miner_metrics import MinerStressTestMetricManager, stress_test_miner
from tensor.config import add_args, SPEC_VERSION
from tensor.protos.inputs_pb2 import NeuronCapabilities
from tensor.sample import weighted_sample


class StressTestValidator(BaseValidator):
    def __init__(self):
        super().__init__()

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.metric_manager = MinerStressTestMetricManager(self)

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        self.miner_heap = heapdict()

        self.last_miner_check = self.block

        bt.logging.info("load_state()")
        self.load_state()

        self.step = 0

    @classmethod
    def add_args(cls, parser):
        add_args(parser, "cpu")

        super().add_args(parser)

        parser.add_argument(
            "--neuron.disable_set_weights",
            action="store_true",
            help="Disables setting weights.",
            default=False,
        )

    async def sync(self):
        # Ensure validator hotkey is still registered on the network.
        self.check_registered()

        if not self.should_sync_metagraph():
            return

        try:
            self.resync_metagraph()
        except Exception as exception:
            bt.logging.error("Failed to resync metagraph", exc_info=exception)

        try:
            if self.should_set_weights():
                await self.set_weights()
        except Exception as exception:
            bt.logging.error("Failed to set validator weights", exc_info=exception)

        # Always save state.
        self.save_state()

    def get_next_uid(self) -> tuple[int, str] | None:
        miners = {
            self.metagraph.axons[uid].hotkey: uid
            for uid in range(self.metagraph.n.item())
            if self.metagraph.axons[uid].is_serving
        }

        infos = {
            uid: self.neuron_info.get(uid)
            for uid in miners.values()
        }

        bt.logging.info(f"Neuron info found: {infos}")

        invalid_miner_list = [
            hotkey
            for hotkey, uid in miners.items()
            if (
                not infos[uid] or
                infos[uid].spec_version != SPEC_VERSION or
                NeuronCapabilities.MINER not in infos[uid].capabilities
            )
        ]

        for hotkey in invalid_miner_list:
            miners.pop(hotkey)

        for hotkey in miners.keys():
            if hotkey in self.miner_heap:
                continue

            # Push new miners to be near the start of the queue to give them a base score
            value = random.random()

            if len(self.miner_heap):
                block_count = max(self.miner_heap.values()) - min(self.miner_heap.values())
                self.miner_heap[hotkey] = int(block_count * value * value * 0.25)
            else:
                self.miner_heap[hotkey] = 0

        disconnected_miner_list = [
            hotkey
            for hotkey in self.miner_heap.keys()
            if hotkey not in miners.keys()
        ]

        for hotkey in disconnected_miner_list:
            self.miner_heap.pop(hotkey)

        if sum(self.miner_heap.values()):
            last_block = max(self.miner_heap.values())

            weighted_choices = [(last_block - block, hotkey) for hotkey, block in self.miner_heap.items()]

            if sum([block for block, _ in weighted_choices]) > 0:
                hotkey = weighted_sample(weighted_choices, k=1)[0]
            else:
                hotkey = random.choice(list(self.miner_heap.keys()))
        elif len(self.miner_heap):
            hotkey = random.choice(list(self.miner_heap.keys()))
        else:
            return None

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
            miner_uid = await self.redis.spop("stress_test_queue")

            if miner_uid is None:
                return
            else:
                hotkey = None
        else:
            next_miner = self.get_next_uid()

            if not next_miner:
                return

            miner_uid, hotkey = next_miner

            await self.redis.srem("stress_test_queue", miner_uid)

        await stress_test_miner(self, miner_uid)

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

                    if blocks_since_neuron_refresh >= neuron_refresh_blocks:
                        await self.sync_neuron_info()
                        sleep = False

                    if blocks_since_check >= check_blocks:
                        await self.check_next_miner()
                        sleep = False

                    # Sync metagraph and potentially set weights.
                    await self.sync()

                    if sleep:
                        neuron_refresh_in = neuron_refresh_blocks - blocks_since_neuron_refresh
                        check_in = check_blocks - blocks_since_check

                        sleep_time = max(min(neuron_refresh_in, check_in), 0)

                        bt.logging.info(f"Nothing to do, sleeping for {sleep_time} blocks")

                        await asyncio.sleep(sleep_time * 12)

                    self.step += 1
                except Exception as exception:
                    bt.logging.error("Failed to forward to miners", exc_info=exception)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", exc_info=err)

    async def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        metrics = await asyncio.gather(*[self.metric_manager.get(uid) for uid in range(self.metagraph.n.item())])

        scores = numpy.array(
            [
                miner_metrics.get_weight() if miner_metrics else 0.0
                for miner_metrics in metrics
            ]
        )

        # Check if self.scores contains any NaN values and log a warning if it does.
        if numpy.isnan(scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        if numpy.sum(scores) == 0.0:
            return

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0
        raw_weights = scores / numpy.linalg.norm(scores, ord=1, axis=0, keepdims=True)

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", self.metagraph.uids)
        # Process the raw weights to final_weights via subtensor limitations.

        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
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
        ) = convert_weights_and_uids_for_emit(
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
            version_key=SPEC_VERSION,
        )

        if result:
            bt.logging.info(f"set_weights on chain successfully! {message}")
        else:
            bt.logging.error(f"set_weights failed. {message}")

    def resync_metagraph(self):
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
            "Metagraph updated, re-syncing hotkeys and moving averages"
        )

        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                # hotkey has been replaced
                self.metric_manager.reset(uid)

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
        with open(self.config.neuron.full_path + "/state.bin", "wb") as state:
            pickle.dump(
                {
                    "step": self.step,
                    "hotkeys": self.hotkeys,
                    "miner_heap": self.miner_heap,
                },
                state,
            )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        path = self.config.neuron.full_path + "/state.bin"

        if not os.path.isfile(path):
            return

        # Load the state of the validator from file.
        with open(path, "rb") as state_file:
            state = pickle.load(state_file)

        self.step = state["step"]
        self.hotkeys = state["hotkeys"]
        self.miner_heap = state.get("miner_heap", self.miner_heap)
