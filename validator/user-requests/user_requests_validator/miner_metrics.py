#  The MIT License (MIT)
#  Copyright © 2023 Yuma Rao
#  Copyright © 2024 WOMBO
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#
#

from typing import Optional

import torch
from aiohttp import ClientSession
from torch import tensor

from neuron.redis import parse_redis_value
from validator.miner_metrics import MinerMetricManager


class MinerUserRequestMetricManager(MinerMetricManager):
    def send_user_request_metric(self, uid: int, successful: int, failed: int, similarity_score: float | None):
        if not self.validator.session:
            self.validator.session = ClientSession()

        return self.send_metrics(
            self.validator.session,
            self.validator.wallet.hotkey,
            "user_requests",
            {
                "miner_uid": uid,
                "successful": successful,
                "failed": failed,
                "similarity_score": similarity_score,
            },
        )

    async def get_rps(self):
        count = self.validator.metagraph.n.item()

        keys = [
            *[f"generation_count_{uid}" for uid in range(count)],
            *[f"generation_time_{uid}" for uid in range(count)],
            *[f"cheater_{uid}" for uid in range(count)],
        ]

        values = await self.validator.redis.mget(keys)

        counts = tensor(
            [
                parse_redis_value(request_count, int)
                for request_count in values[0:count]
            ],
            dtype=torch.int32,
        )

        times = tensor([parse_redis_value(time, float) for time in values[count:count * 2]])

        cheaters = [parse_redis_value(cheater, bool) for cheater in values[count * 2:]]

        return tensor([0.0 if cheaters[index] else rps for index, rps in enumerate((counts / times).tolist())])

    async def successful_user_request(self, uid: int, similarity_score: float):
        async with self.validator.redis.pipeline() as pipeline:
            successful = await pipeline.incr(f"successful_user_requests_{uid}")
            failed = await pipeline.get(f"failed_user_requests_{uid}"),
            old_similarity_score = await pipeline.set(f"similarity_score_{uid}", similarity_score, get=True)

            await pipeline.execute()

            similarity_score = min(old_similarity_score, similarity_score)

            await self.send_user_request_metric(uid, successful, failed, similarity_score)

    async def failed_user_request(self, uid: int, similarity_score: Optional[float]):
        async with self.validator.redis.pipeline() as pipeline:
            successful = await pipeline.get(f"successful_user_requests_{uid}")
            failed = await pipeline.incr(f"failed_user_requests_{uid}"),

            if similarity_score:
                old_similarity_score = await pipeline.set(f"similarity_score_{uid}", similarity_score, get=True)
                similarity_score = min(old_similarity_score, similarity_score)

            await pipeline.execute()

            await self.send_user_request_metric(uid, successful, failed, similarity_score)
