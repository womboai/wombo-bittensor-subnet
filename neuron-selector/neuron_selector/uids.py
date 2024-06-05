from typing import Any, Callable

import bittensor as bt
import torch
from torch import Tensor

from tensor.config import SPEC_VERSION
from tensor.protos.inputs_pb2 import InfoResponse
from tensor.sample import weighted_sample


def get_best_uids(
    blacklist: Any,
    metagraph: bt.metagraph,
    neuron_info: dict[int, InfoResponse],
    rank: Tensor,
    condition: Callable[[int, InfoResponse], bool],
    k: int = 3,
) -> Tensor:
    available_uids = [
        uid
        for uid in range(metagraph.n.item())
        if (
            metagraph.axons[uid].is_serving and
            (not blacklist or
             (
                 metagraph.axons[uid].hotkey not in blacklist.hotkeys and
                 metagraph.axons[uid].coldkey not in blacklist.coldkeys
             ))
        )
    ]

    infos = {
        uid: neuron_info.get(uid)
        for uid in available_uids
    }

    bt.logging.info(f"Neuron info found: {infos}")

    available_uids = [
        uid
        for uid in available_uids
        if infos[uid] and infos[uid].spec_version == SPEC_VERSION and condition(uid, infos[uid])
    ]

    if not len(available_uids):
        return torch.tensor([], dtype=torch.int64)

    uids = torch.tensor(
        weighted_sample(
            [(rank[uid].item(), uid) for uid in available_uids],
            k=k,
        ),
    )

    return uids
