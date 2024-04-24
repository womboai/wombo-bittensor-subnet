# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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

import traceback
from abc import ABC, abstractmethod

import bittensor as bt

from neuron.misc import ttl_get_block
# Sync calls set weights and also resyncs the metagraph.
from tensor.config import config


class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    @classmethod
    @abstractmethod
    def check_config(cls, config: bt.config):
        ...

    @classmethod
    @abstractmethod
    def add_args(cls, parser):
        ...

    subtensor: bt.subtensor
    wallet: bt.wallet
    metagraph: bt.metagraph
    config: bt.config
    axon: bt.axon

    @property
    def block(self):
        return ttl_get_block(self.subtensor)

    def __init__(self):
        self.config = config(self.add_args)
        self.check_config(self.config)

        # Set up logging with the provided configuration and directory.
        bt.logging(config=self.config, logging_dir=self.config.full_path)

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        bt.logging.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the miner.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        # The subtensor is our connection to the Bittensor blockchain.
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # The metagraph holds the state of the network, letting us know about other validators and miners.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Check if the neuron is registered on the Bittensor network before proceeding further.
        self.check_registered()

        bt.logging.info(
            f"Running neuron on subnet: {self.config.netuid} with using network: {self.subtensor.chain_endpoint}"
        )

        self.log_dashboard_info()

    def log_dashboard_info(self):
        hotkey = self.wallet.hotkey.ss58_address
        uid = self.metagraph.hotkeys.index(hotkey)
        axon = self.metagraph.axons[uid]

        message = f"0x{hotkey}:{axon.ip}:{axon.port}:{uid}"
        signature = self.wallet.hotkey.sign(message)
        verification_code = f"{hotkey}:{signature}"

        if self.subtensor.network == "test":
            grafana_link = "https://test-bittensor.w.ai/d/bdi11ys29q5moe/miner-stats"
        else:
            grafana_link = "https://bittensor.w.ai/d/bdi11ys29q5moe/miner-stats"

        bt.logging.info(
            f"You can view your stats and metrics on the dashboard at {grafana_link}, "
            f"using the verification code {verification_code}"
        )

    @abstractmethod
    async def resync_metagraph(self):
        ...

    async def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if not self.should_sync_metagraph():
            return

        try:
            await self.resync_metagraph()
        except Exception as _:
            bt.logging.error("Failed to resync metagraph, ", traceback.format_exc())

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    @abstractmethod
    def should_sync_metagraph(self):
        ...
