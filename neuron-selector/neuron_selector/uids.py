from typing import Any, Callable

import bittensor as bt
from numpy import ndarray

from tensor.config import SPEC_VERSION
from tensor.protos.inputs_pb2 import InfoResponse
from tensor.sample import weighted_sample


def get_best_uids(
    blacklist: Any,
    metagraph: bt.metagraph,
    neuron_info: dict[int, InfoResponse],
    rank: ndarray,
    condition: Callable[[int, InfoResponse], bool],
    k: int = 3,
) -> list[int]:
    available_uids = [
        uid
        for uid in range(metagraph.n)
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
        return []

    return weighted_sample(
        [(rank[uid], uid) for uid in available_uids],
        k=k,
    )
