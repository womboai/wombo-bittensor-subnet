import torch
import random
import bittensor as bt
from typing import Callable, List

from bittensor import AxonInfo

from tensor.protocol import NeuronInfoSynapse


def is_validator(metagraph: "bt.metagraph.Metagraph", uid: int, info: NeuronInfoSynapse) -> bool:
    return metagraph.validator_permit[uid] and info.validator


def is_miner(metagraph: "bt.metagraph.Metagraph", uid: int, info: NeuronInfoSynapse) -> bool:
    return not is_validator(metagraph, uid, info)


async def get_random_uids(
    self,
    k: int,
    availability_checker: Callable[["bt.metagraph.Metagraph", int, NeuronInfoSynapse], bool],
) -> torch.LongTensor:
    active_uids = {}

    for neuron in self.metagraph.neurons:
        active_uids[neuron.uid] = neuron.active

    available_uids = [
        uid
        for uid in range(self.metagraph.n.item())
        if active_uids[uid] and self.metagraph.axons[uid].is_serving
    ]

    axons = [self.metagraph.axons[uid] for uid in available_uids]

    async with self.dendrite as dendrite:
        responses: List[NeuronInfoSynapse] = (await dendrite.forward(
            # Send the query to selected miner axon in the network.
            axons=axons,
            synapse=NeuronInfoSynapse(),
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
        ))

    available_uids = [
        uid
        for uid, info in zip(available_uids, responses)
        if availability_checker(self.metagraph, uid, info)
    ]

    uids = torch.tensor(random.sample(available_uids, min(k, len(available_uids))))
    return uids
