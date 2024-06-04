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

import bittensor as bt
from bittensor import AxonInfo
from google.protobuf.empty_pb2 import Empty
from tensor.protos.inputs_pb2 import InfoResponse
from tensor.protos.inputs_pb2_grpc import NeuronStub

from tensor.config import SPEC_VERSION
from tensor.response import Response, create_request

DEFAULT_NEURON_INFO = InfoResponse(spec_version=SPEC_VERSION, capabilities=set())


async def get_neuron_info(axon: AxonInfo) -> Response[InfoResponse]:
    return await create_request(axon, Empty(), lambda channel: NeuronStub(channel).Info)


async def sync_neuron_info(metagraph: bt.metagraph, wallet: bt.wallet):
    uids: list[int] = [
        uid
        for uid in range(metagraph.n.item())
        if metagraph.axons[uid].is_serving
    ]

    uid_by_hotkey: dict[str, int] = {
        metagraph.axons[uid].hotkey: uid
        for uid in uids
        if metagraph.axons[uid].hotkey != wallet.hotkey.ss58_address
    }

    axon_by_hotkey: dict[str, AxonInfo] = {
        metagraph.axons[uid].hotkey: metagraph.axons[uid]
        for uid in uids
    }

    axons = [axon_by_hotkey[hotkey] for hotkey in uid_by_hotkey.keys()]

    neuron_info: list[Response[InfoResponse]] = list(
        await asyncio.gather(
            *[
                get_neuron_info(axon)
                for axon in axons
            ]
        )
    )

    info_by_hotkey = {
        info.axon.hotkey: info
        for info in neuron_info
    }

    return {
        uid_by_hotkey[hotkey]: info.data
        for hotkey, info in info_by_hotkey.items()
    }
