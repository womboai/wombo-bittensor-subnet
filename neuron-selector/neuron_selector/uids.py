import torch
import random
import bittensor
from typing import List

from tensor.protocol import NeuronInfoSynapse


DEFAULT_NEURON_INFO = NeuronInfoSynapse()


async def sync_neuron_info(self):
    uids = [
        uid
        for uid in range(self.metagraph.n.item())
        if self.metagraph.axons[uid].is_serving and self.metagraph.active[uid]
    ]

    axons = [
        self.metagraph.axons[uid]
        for uid in uids
        if self.metagraph.axons[uid].hotkey != self.wallet.hotkey.ss58_address
    ]

    uids_by_hotkey = {axon.hotkey: uid for uid, axon in zip(uids, axons)}

    async with self.dendrite as dendrite:
        neuron_info: List[NeuronInfoSynapse] = await dendrite.forward(
            axons=axons,
            synapse=NeuronInfoSynapse(),
            deserialize=False,
        )

    info_by_hotkey = {
        info.axon.hotkey: info
        for info in neuron_info
    }

    self.neuron_info = {
        uids_by_hotkey[hotkey]: info
        for hotkey, info in info_by_hotkey.items()
    }


def get_random_uids(
    self,
    k: int,
    validators: bool,
) -> torch.LongTensor:
    if validators:
        def validator_condition(uid: int, info: NeuronInfoSynapse) -> bool:
            return info.is_validator and self.metagraph.validator_permit[uid]
    else:
        def validator_condition(_uid: int, info: NeuronInfoSynapse) -> bool:
            return info.is_validator is False

    available_uids = [
        uid
        for uid in range(self.metagraph.n.item())
        if (
                self.metagraph.active[uid] and
                self.metagraph.axons[uid].is_serving
        )
    ]

    infos = {
        uid: self.neuron_info.get(uid, DEFAULT_NEURON_INFO)
        for uid in available_uids
    }

    bittensor.logging.info(f"Neuron info found: {infos}")

    available_uids = [
        uid
        for uid in available_uids
        if validator_condition(uid, infos[uid])
    ]

    uids = torch.tensor(random.sample(available_uids, min(k, len(available_uids))))
    return uids
