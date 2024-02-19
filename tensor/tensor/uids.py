import torch
import random
import bittensor as bt
from typing import Callable

from bittensor import AxonInfo

from tensor.protocol import NeuronInfoSynapse


def is_validator(metagraph: "bt.metagraph.Metagraph", uid: int, info: NeuronInfoSynapse) -> bool:
    return metagraph.validator_permit[uid] and info.validator


def is_miner(metagraph: "bt.metagraph.Metagraph", uid: int, info: NeuronInfoSynapse) -> bool:
    return not is_validator(metagraph, uid, info)


def get_random_uids(
    self,
    k: int,
    availability_checker: Callable[["bt.metagraph.Metagraph", int, NeuronInfoSynapse], bool],
) -> torch.LongTensor:
    available_uids = []

    for uid in range(self.metagraph.n.item()):
        axon: AxonInfo = self.metagraph.axons[uid]

        metagraph: bt.metagraph = self.metagraph

        active = False

        for neuron in metagraph.neurons:
            if neuron.uid == uid:
                active = neuron.active
                break

        uid_is_available = (
            active and
            axon.is_serving and
            availability_checker(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
        )

        if uid_is_available:
            available_uids.append(uid)

    uids = torch.tensor(random.sample(available_uids, min(k, len(available_uids))))
    return uids
