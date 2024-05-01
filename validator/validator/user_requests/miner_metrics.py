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

from aiohttp import ClientSession

from validator.base.miner_metrics import MinerMetricManager


class MinerUserRequestMetricManager(MinerMetricManager):
    def send_user_request_metric(self, uid: int):
        if not self.validator.user_request_session:
            self.validator.user_request_session = ClientSession()

        return self.send_metrics(
            self.validator.user_request_session,
            self.validator.dendrite,
            "user_requests",
            {
                "miner_uid": uid,
                "successful": self.miner_data.successful_user_requests[uid].item(),
                "failed": self.miner_data.failed_user_requests[uid].item(),
                "similarity_score": self.miner_data.similarity_scores[uid].item(),
            },
        )

    async def successful_user_request(self, uid: int, similarity_score: float):
        self.successful_user_requests[uid] += 1
        self.similarity_scores[uid] = min(self.similarity_scores[uid].item(), similarity_score)

        await self.send_user_request_metric(uid)

    async def failed_user_request(self, uid: int, similarity_score: Optional[float]):
        self.failed_user_requests[uid] += 1

        if similarity_score:
            self.similarity_scores[uid] = min(self.similarity_scores[uid].item(), similarity_score)
        else:
            self.similarity_scores[uid] = self.similarity_scores[uid] * 0.75

        await self.send_user_request_metric(uid)
