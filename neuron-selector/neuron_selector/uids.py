import random
import bittensor
from typing import List, Optional

import torch
from bittensor import AxonInfo

from tensor.protocol import NeuronInfoSynapse

DEFAULT_NEURON_INFO = NeuronInfoSynapse()


async def sync_neuron_info(self, dendrite: bittensor.dendrite):
    uids: list[int] = [
        uid
        for uid in range(self.metagraph.n.item())
        if self.metagraph.axons[uid].is_serving
    ]

    uid_by_hotkey: dict[str, int] = {
        self.metagraph.axons[uid].hotkey: uid
        for uid in uids
        if self.metagraph.axons[uid].hotkey != self.wallet.hotkey.ss58_address
    }

    axon_by_hotkey: dict[str, AxonInfo] = {
        self.metagraph.axons[uid].hotkey: self.metagraph.axons[uid]
        for uid in uids
    }

    axons = [axon_by_hotkey[hotkey] for hotkey in uid_by_hotkey.keys()]

    neuron_info: List[NeuronInfoSynapse] = await dendrite(
        axons=axons,
        synapse=NeuronInfoSynapse(),
        deserialize=False,
    )

    info_by_hotkey = {
        info.axon.hotkey: info
        for info in neuron_info
    }

    self.neuron_info = {
        uid_by_hotkey[hotkey]: info
        for hotkey, info in info_by_hotkey.items()
    }


def get_best_uids(
    self,
    validators: bool,
    k: int = 3,
) -> torch.LongTensor:
    if validators:
        trust = self.metagraph.validator_trust

        def validator_condition(uid: int, info: NeuronInfoSynapse) -> bool:
            return info.is_validator and self.metagraph.validator_permit[uid]
    else:
        trust = self.metagraph.trust

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

    uids = torch.tensor(
        random.choices(
            available_uids,
            weights=[trust[uid] for uid in available_uids],
            k=k,
        )
    )

    return uids
