import torch
import random
import bittensor as bt
from typing import List, Callable


def is_validator(metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int) -> bool:
    return metagraph.validator_permit[uid] and metagraph.S[uid] > vpermit_tao_limit


def is_miner(metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int) -> bool:
    return not is_validator(metagraph, uid, vpermit_tao_limit)


def get_random_uids(
    self,
    k: int,
    availability_checker: Callable[["bt.metagraph.Metagraph", int, int], bool],
    exclude: List[int] = None,
) -> torch.LongTensor:
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = (
            self.metagraph.axons[uid].is_serving and
            availability_checker(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
        )

        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    uids = torch.tensor(random.sample(available_uids, k))
    return uids
