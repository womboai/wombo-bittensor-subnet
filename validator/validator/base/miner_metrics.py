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

import traceback
from typing import Any

import bittensor as bt
from aiohttp import ClientSession, BasicAuth
from substrateinterface import Keypair

from validator.reward import select_endpoint


class MinerMetricManager:
    def __init__(self, validator):
        self.validator = validator

        self.data_endpoint = select_endpoint(
            validator.config.data_endpoint,
            validator.config.subtensor.network,
            "https://dev-neuron-identifier.api.wombo.ai/api/data",
            "https://neuron-identifier.api.wombo.ai/api/data",
        )

    async def send_metrics(
        self,
        session: ClientSession,
        dendrite: bt.dendrite,
        endpoint: str,
        data: Any,
    ):
        if not self.validator.config.send_metrics:
            return

        keypair: Keypair = dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"

        bt.logging.info(f"Sending {endpoint} metrics {data}")

        try:
            async with session.post(
                f"{self.data_endpoint}/{endpoint}",
                auth=BasicAuth(hotkey, signature),
                json=data,
            ):
                pass
        except Exception as _:
            bt.logging.warning("Failed to export metrics, ", traceback.format_exc())
