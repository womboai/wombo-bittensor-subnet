import heapdict
import torch
import random
import bittensor
from typing import List
from datetime import datetime


from tensor.protocol import NeuronInfoSynapse


DEFAULT_NEURON_INFO = NeuronInfoSynapse()

async def sync_neuron_info(self):
    uids = [
        uid
        for uid in range(self.metagraph.n.item())
        if self.metagraph.axons[uid].is_serving
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
        if self.metagraph.axons[uid].is_serving
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


def get_oldest_uids(
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

    all_uids = [
        uid
        for uid in range(self.metagraph.n.item())
        if self.metagraph.axons[uid].is_serving
    ]

    random.shuffle(all_uids)
    # if this is not randomized, every new validator will have the same mining order in their heap upon first launch,
    # which would likely perpetuate the problem this function solves

    infos = {
        uid: self.neuron_info.get(uid, DEFAULT_NEURON_INFO)
        for uid in all_uids
    }

    all_uids = [
        uid
        for uid in all_uids
        if validator_condition(uid, infos[uid])
    ]

    for uid in all_uids:
        if uid not in self.miner_heap:
            self.miner_heap[uid] = 0

    uids = torch.tensor(
        get_n_lowest_values(self.miner_heap, k)
    )
    return uids


def get_n_lowest_values(heapdict: heapdict.heapdict, n):
    lowest_values = []
    for _ in range(min(n, len(heapdict))):
        key, value = heapdict.popitem()
        lowest_values.append((key, value))
        heapdict[key] = int(datetime.utcnow().timestamp())
    return lowest_values
