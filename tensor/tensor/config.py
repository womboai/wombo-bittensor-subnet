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

import argparse
import logging
import os
from typing import Callable

import bittensor as bt

SPEC_VERSION = 20


def check_config(config: bt.config, name: str):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    # Add custom event logger for the events.
    logging.addLevelName(38, "EVENTS")


def add_args(parser: argparse.ArgumentParser, default_device: str = "cuda"):
    """
    Adds relevant arguments to the parser for operation.
    """
    # Netuid Arg: The netuid of the subnet to connect to.
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default=default_device,
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we pull the metagraph, measured in 12 second blocks).",
        default=100,
    )

    parser.add_argument(
        "--neuron.redis_url",
        type=str,
        help="The URL to connect to Redis at",
        default="redis://localhost:6379/",
    )

    parser.add_argument(
        "--blacklist.hotkeys",
        action='append',
        help="The hotkeys to block when sending or receiving requests",
        default=[],
    )

    parser.add_argument(
        "--blacklist.coldkeys",
        action='append',
        help="The coldkeys to block when sending or receiving requests",
        default=["5DhPDjLR4YNAixDLNFNP2pTiCpkDQ5A5vm5fyQ3Q52rYcEaw"],
    )

    parser.add_argument(
        "--is_hotkey_allowed_endpoint",
        type=str,
        help="The endpoint called when checking if the hotkey should be whitelisted for requests",
        default="",
    )


def config(add_args_fn: Callable[[argparse.ArgumentParser], None]):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    add_args_fn(parser)
    return bt.config(parser)
