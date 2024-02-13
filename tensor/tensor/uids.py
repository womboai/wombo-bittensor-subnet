import torch
import random
import bittensor as bt
from typing import List, Callable, Awaitable

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
    uids = range(self.metagraph.n.item())
    axons = [self.metagraph.axons[uid] for uid in uids]

    async with self.dendrite as dendrite:
        infos = await dendrite.forward(
            axons=axons,
            synapse=NeuronInfoSynapse(),
            deserialize=False,
        )

    neuron_infos = zip(uids, infos)

    candidate_uids = [
        uid
        for uid, info in neuron_infos
        if (
            self.metagraph.axons[uid].is_serving and
            availability_checker(self.metagraph, uid, info)
        )
    ]

    return torch.tensor(random.sample(candidate_uids, k))
