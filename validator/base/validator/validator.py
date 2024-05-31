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
from time import perf_counter

import bittensor as bt
from bittensor import AxonInfo
from grpc.aio import insecure_channel

from neuron.neuron import BaseNeuron
from neuron.protos.neuron_pb2 import MinerGenerationResponse
from neuron.protos.neuron_pb2_grpc import MinerStub
from neuron_selector.uids import sync_neuron_info
from tensor.config import check_config
from tensor.protos.inputs_pb2 import InfoResponse, GenerationRequestInputs


async def generate_miner_response(inputs: GenerationRequestInputs, axon: AxonInfo) -> tuple[bytes, float] | None:
    # TODO Return None if error
    async with insecure_channel(f"{axon.ip}:{axon.port}") as channel:
        stub = MinerStub(channel)
        generation_call = stub.Generate(inputs)

        start = perf_counter()
        response: MinerGenerationResponse = await generation_call

        return response.frames, perf_counter() - start


class BaseValidator(BaseNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

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
        await sync_neuron_info(self, self.dendrite)

        self.last_neuron_info_block = self.block
