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


import copy
import os.path
import random
import traceback
from abc import abstractmethod

import heapdict
import torch
import asyncio
import bittensor as bt

from typing import List
from traceback import print_exception

from neuron.neuron import BaseNeuron
from tensor.config import add_args
from neuron_selector.uids import sync_neuron_info, DEFAULT_NEURON_INFO


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    spec_version: int = 5

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)

        # Serve axon to enable external connections.
        self.serve_axon()

        self.loop = asyncio.get_event_loop()

        self.loop.run_until_complete(sync_neuron_info(self))

        self.miner_heap = heapdict.heapdict()

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

    @abstractmethod
    async def check_miners(self):
        ...

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

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
            "--are_wombo_neurons_endpoint",
            type=str,
            help="The endpoint to call to check if the uids are WOMBO owned, which we give a reward advantage for",
            default="",
        )

        parser.add_argument(
            "--neuron.sample_size",
            type=int,
            help="The number of miners to query in a single step.",
            default=10,
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
            default=0.05,
        )

    def run(self):
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

        # Check that validator is registered on the network.
        self.sync()
        self.resync_metagraph()

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
                    self.loop.run_until_complete(self.check_miners())
                except Exception as _:
                    bt.logging.error("Failed to forward to miners, ", traceback.format_exc())

                # Sync metagraph and potentially set weights.
                self.sync()

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
                print_exception(type(err), err, err.__traceback__)
            )

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = torch.nn.functional.normalize(
            self.scores, p=1, dim=0
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

    def resync_metagraph(self):
        if not self.should_sync_metagraph():
            if self.step == 0:
                self.loop.run_until_complete(sync_neuron_info(self))

            return

        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            self.loop.run_until_complete(sync_neuron_info(self))

            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros(self.metagraph.n).to(
                self.device
            )
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        self.loop.run_until_complete(sync_neuron_info(self))

    def sync(self):
        super().sync()

        try:
            if self.should_set_weights():
                self.set_weights()
        except Exception as _:
            bt.logging.error("Failed to set validator weights, ", traceback.format_exc())

        # Always save state.
        self.save_state()

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

    def update_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: torch.FloatTensor = self.scores.to(self.device).scatter(
            0, torch.tensor(uids).to(self.device), rewards.to(self.device)
        )
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * scattered_rewards + (
            1 - alpha
        ) * self.scores.to(self.device)
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
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
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
        self.miner_heap = state.get("miner_heap", heapdict.heapdict())


def get_oldest_uids(
    self: BaseValidatorNeuron,
    k: int,
) -> torch.LongTensor:
    all_uids_and_hotkeys_dict = {
        self.metagraph.axons[uid].hotkey: uid
        for uid in range(self.metagraph.n.item())
        if self.metagraph.axons[uid].is_serving
    }

    hotkeys = list(all_uids_and_hotkeys_dict.keys())
    random.shuffle(hotkeys)
    shuffled_miner_dict = {hotkey: all_uids_and_hotkeys_dict[hotkey] for hotkey in hotkeys}
    # if this is not randomized, every new validator will have the same mining order in their heap upon first launch,
    # which would likely perpetuate the problem this function solves

    infos = {
        uid: self.neuron_info.get(uid, DEFAULT_NEURON_INFO)
        for uid in shuffled_miner_dict.values()
    }

    bt.logging.info(f"Neuron info found: {infos}")

    invalid_miner_list = [
        hotkey
        for hotkey, uid in shuffled_miner_dict.items()
        if infos[uid].is_validator is not False
    ]

    for hotkey in invalid_miner_list:
        shuffled_miner_dict.pop(hotkey)

    for hotkey in shuffled_miner_dict.keys():
        if hotkey not in self.miner_heap:
            self.miner_heap[hotkey] = self.block

    disconnected_miner_list = [
        hotkey
        for hotkey in self.miner_heap.keys()
        if hotkey not in shuffled_miner_dict.keys()
    ]
    for hotkey in disconnected_miner_list:
        self.miner_heap.pop(hotkey)

    uids = torch.tensor(
        [shuffled_miner_dict[hotkey] for hotkey in get_n_lowest_values(self, k)]
    )

    return uids


def get_n_lowest_values(self, n):
    lowest_values = []

    for _ in range(min(n, len(self.miner_heap))):
        hotkey, ts = self.miner_heap.popitem()
        lowest_values.append(hotkey)
        self.miner_heap[hotkey] = self.block

    return lowest_values
