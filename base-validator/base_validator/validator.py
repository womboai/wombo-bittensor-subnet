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

import copy
from hashlib import sha256
from typing import TypeVar

import bittensor as bt
from bittensor import AxonInfo
from grpc.aio import Channel

from neuron.neuron import BaseNeuron
from neuron.protos.neuron_pb2 import MinerGenerationResponse
from neuron.protos.neuron_pb2_grpc import MinerStub
from tensor.config import check_config
from tensor.neuron_info import sync_neuron_info
from tensor.protos.inputs_pb2 import InfoResponse, GenerationRequestInputs
from tensor.response import SuccessfulResponseInfo, call_request, Response

T = TypeVar("T")


class SuccessfulGenerationResponseInfo(SuccessfulResponseInfo):
    similarity_score: float
    cheater: bool

    @classmethod
    def of(cls, info: SuccessfulResponseInfo, similarity_score: float, cheater: bool):
        return cls(
            axon=info.axon,
            process_time=info.process_time,
            similarity_score=similarity_score,
            cheater=cheater,
        )


async def get_miner_response(
    inputs: GenerationRequestInputs,
    axon: AxonInfo,
    channel: Channel,
    wallet: bt.wallet,
) -> Response[MinerGenerationResponse]:
    return await call_request(axon, inputs, MinerStub(channel).Generate, wallet)


def is_cheater(uid: int, frames: bytes, expected_hash: bytes):
    detected_hash = sha256(frames).digest()

    if expected_hash != detected_hash:
        bt.logging.info(
            f"Miner {uid} has been detected as a cheater, "
            f"as they declared the hash as {expected_hash} while it was {detected_hash}"
        )

        return True

    return False


class BaseValidator(BaseNeuron):
    neuron_info: dict[int, InfoResponse]

    def __init__(self):
        super().__init__()

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        self.session = None

        self.neuron_info = {}

        self.last_neuron_info_block = self.block

    @classmethod
    def check_config(cls, config: bt.config):
        check_config(config, "validator")

    @classmethod
    def add_args(cls, parser):
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

    async def sync_neuron_info(self):
        self.neuron_info = await sync_neuron_info(self.metagraph, self.wallet)

        self.last_neuron_info_block = self.block
