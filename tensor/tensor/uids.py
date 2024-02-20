import torch
import random
import bittensor as bt
from typing import Callable, List

from tensor.protocol import NeuronInfoSynapse


async def get_random_uids(
    self,
    k: int,
    validators: bool,
) -> torch.LongTensor:
    available_uids = [
        uid
        for uid in range(self.metagraph.n.item())
        if (
                self.metagraph.active[uid] and
                self.metagraph.axons[uid].is_serving
                and (not validators or self.metagraph.validator_permit[uid])
        )
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
        if info.is_validator == validators
    ]

    uids = torch.tensor(random.sample(available_uids, min(k, len(available_uids))))
    return uids
