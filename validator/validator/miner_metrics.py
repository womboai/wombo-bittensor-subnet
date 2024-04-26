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

import asyncio
import os
import traceback
from random import shuffle
from typing import Any, TypeAlias, Annotated, Optional

import bittensor as bt
import nltk
import torch
from aiohttp import ClientSession, BasicAuth
from pydantic import BaseModel, Field
from substrateinterface import Keypair
from torch import Tensor

from image_generation_protocol.cryptographic_sample import cryptographic_sample
from image_generation_protocol.io_protocol import ImageGenerationInputs
from tensor.protocol import ImageGenerationSynapse
from tensor.timeouts import CLIENT_REQUEST_TIMEOUT
from validator.reward import select_endpoint, reward

nltk.download('words')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import words
from nltk import pos_tag

TIME_CONSTRAINT = 30.0
"""
The time constraint to test the RPS under,
 if requests start taking longer than this value then we have reached the maximum viable RPS
"""

MAX_ERROR_RATE = 0.25
"""
The max percentage of failures acceptable before stopping
 and assuming we have reached the maximum viable RPS 
"""

ValidatableResponse: TypeAlias = tuple[ImageGenerationSynapse, ImageGenerationInputs]

WORDS = [word for word, tag in pos_tag(words.words(), tagset='universal') if tag == "ADJ" or tag == "NOUN"]
REQUEST_INCENTIVE = 0.0001


def generate_random_prompt():
    words = cryptographic_sample(WORDS, k=min(len(WORDS), min(os.urandom(1)[0] % 32, 8))) + ["tao"]
    shuffle(words)

    return ", ".join(words)


class MinerMetrics(BaseModel):
    generated_count: Annotated[int, Field(ge=0)]
    generation_time: Annotated[float, Field(gt=0)]
    similarity_score: Annotated[float, Field(ge=0, le=1)]
    error_rate: Annotated[float, Field(ge=0, le=1)]
    successful_user_requests: Annotated[int, Field(ge=0)]
    failed_user_requests: Annotated[int, Field(ge=0)]

    def get_weight(self):
        concurrency_factor = pow((self.generated_count / self.generation_time * CLIENT_REQUEST_TIMEOUT) / 128, 1.125)
        similarity_factor = pow(self.similarity_score, 8)
        success_factor = pow(1 - self.error_rate, 2)

        return (
            (concurrency_factor * similarity_factor * success_factor) +
            (self.successful_user_requests * REQUEST_INCENTIVE) -
            (self.failed_user_requests * REQUEST_INCENTIVE * 8)
        )


class MinerData:
    generation_counts: Tensor
    generation_times: Tensor
    similarity_scores: Tensor
    error_rates: Tensor
    successful_user_requests: Tensor
    failed_user_requests: Tensor

    def __init__(self, validator):
        metagraph = validator.metagraph
        device = validator.device

        self.generation_counts = torch.zeros_like(metagraph.S, dtype=torch.int16, device=device)
        self.generation_times = torch.zeros_like(metagraph.S, dtype=torch.float32, device=device)
        self.similarity_scores = torch.zeros_like(metagraph.S, dtype=torch.float16, device=device)
        self.error_rates = torch.zeros_like(metagraph.S, dtype=torch.float16, device=device)
        self.successful_user_requests = torch.zeros_like(metagraph.S, dtype=torch.int64, device=device)
        self.failed_user_requests = torch.zeros_like(metagraph.S, dtype=torch.int64, device=device)

    def __getitem__(self, uid: int):
        if not self.generation_counts[uid]:
            return None

        return MinerMetrics(
            generated_count=max(0, self.generation_counts[uid].item()),
            generation_time=max(0.0, self.generation_times[uid].item()),
            similarity_score=max(0.0, min(1.0, self.similarity_scores[uid].item())),
            error_rate=max(0.0, min(1.0, self.error_rates[uid].item())),
            successful_user_requests=max(0, self.successful_user_requests[uid].item()),
            failed_user_requests=max(0, self.failed_user_requests[uid].item()),
        )

    def failed_miner(self, uid: int):
        self.generation_counts[uid] = 0
        self.generation_times[uid] = 0.0
        self.similarity_scores[uid] = 0.0
        self.error_rates[uid] = 1.0

    def reset(self, uid: int):
        self.generation_counts[uid] = 0
        self.generation_times[uid] = 0.0
        self.similarity_scores[uid] = 0.0
        self.error_rates[uid] = 0.0
        self.successful_user_requests[uid] = 0
        self.failed_user_requests[uid] = 0

    def successful_stress_test(
        self,
        uid: int,
        generated_count: int,
        generation_time: float,
        similarity_score: float,
        error_rate: float,
    ):
        self.generation_counts[uid] = generated_count
        self.generation_times[uid] = generation_time
        self.similarity_scores[uid] = similarity_score
        self.error_rates[uid] = error_rate

    def successful_user_request(self, uid: int, similarity_score: float):
        self.successful_user_requests[uid] += 1
        self.similarity_scores[uid] = min(self.similarity_scores[uid].item(), similarity_score)

    def failed_user_request(self, uid: int, similarity_score: Optional[float]):
        self.failed_user_requests[uid] += 1

        if similarity_score:
            self.similarity_scores[uid] = min(self.similarity_scores[uid].item(), similarity_score)
        else:
            self.similarity_scores[uid] = self.similarity_scores[uid] * 0.75


class MinerMetricManager:
    miner_data: MinerData

    def __init__(self, validator):
        self.validator = validator

        self.miner_data = MinerData(validator)

        self.data_endpoint = select_endpoint(
            validator.config.data_endpoint,
            validator.config.subtensor.network,
            "https://dev-neuron-identifier.api.wombo.ai/api/data",
            "https://neuron-identifier.api.wombo.ai/api/data",
        )

    def __getitem__(self, uid: int):
        return self.miner_data[uid]

    def failed_miner(self, uid: int):
        self.miner_data.failed_miner(uid)

    def reset(self, uid: int):
        self.miner_data.reset(uid)

    def resize(self):
        new_data = MinerData(self.validator)
        existing_data = self.miner_data

        length = len(self.validator.hotkeys)

        new_data.generation_counts[:length] = existing_data.generation_counts[:length]
        new_data.generation_times[:length] = existing_data.generation_times[:length]
        new_data.similarity_scores[:length] = existing_data.similarity_scores[:length]
        new_data.error_rates[:length] = existing_data.error_rates[:length]
        new_data.successful_user_requests[:length] = existing_data.successful_user_requests[:length]
        new_data.failed_user_requests[:length] = existing_data.failed_user_requests[:length]

        self.miner_data = new_data

    def load_data(self, data: MinerData):
        self.miner_data = data

    async def send_metrics(
        self,
        session: ClientSession,
        dendrite: bt.dendrite,
        endpoint: str,
        data: Any,
    ):
        if not self.validator.config.send_metrics:
            return

        keypair: Keypair = dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"

        bt.logging.info(f"Sending {endpoint} metrics {data}")

        try:
            async with session.post(
                f"{self.data_endpoint}/{endpoint}",
                auth=BasicAuth(hotkey, signature),
                json=data,
            ):
                pass
        except Exception as _:
            bt.logging.warning("Failed to export metrics, ", traceback.format_exc())

    def send_user_request_metric(self, uid: int):
        if not self.validator.user_request_session:
            self.validator.user_request_session = ClientSession()

        return self.send_metrics(
            self.validator.user_request_session,
            self.validator.forward_dendrite,
            "user_requests",
            {
                "miner_uid": uid,
                "successful": self.miner_data.successful_user_requests[uid].item(),
                "failed": self.miner_data.failed_user_requests[uid].item(),
                "similarity_score": self.miner_data.similarity_scores[uid].item(),
            },
        )

    async def successful_stress_test(
        self,
        uid: int,
        generated_count: int,
        generation_time: float,
        similarity_score: float,
        error_rate: float,
    ):
        self.miner_data.successful_stress_test(uid, generated_count, generation_time, similarity_score, error_rate)

        await self.send_metrics(
            self.validator.stress_test_session,
            self.validator.periodic_check_dendrite,
            "success",
            {
                "miner_uid": uid,
                "similarity_score": similarity_score,
                "generation_time": generation_time,
                "concurrent_requests_processed": generated_count,
                "error_rate": error_rate,
            },
        )

    async def failed_stress_test(self, uid: int):
        self.failed_miner(uid)

        await self.send_metrics(
            self.validator.stress_test_session,
            self.validator.periodic_check_dendrite,
            "failure",
            uid,
        )

    async def successful_user_request(self, uid: int, similarity_score: float):
        self.miner_data.successful_user_request(uid, similarity_score)

        await self.send_user_request_metric(uid)

    async def failed_user_request(self, uid: int, similarity_score: Optional[float]):
        self.miner_data.failed_user_request(uid, similarity_score)

        await self.send_user_request_metric(uid)


async def set_miner_metrics(validator, uid: int):
    blacklist = validator.config.blacklist
    axon = validator.metagraph.axons[uid]

    if blacklist and (axon.hotkey in blacklist.hotkeys or axon.coldkey in blacklist.coldkeys):
        validator.metric_manager.failed_miner(uid)
        return

    bt.logging.info(f"Measuring RPS of UID {uid} with Axon {axon}")

    count = 4
    error_rate: float | None = None

    finished_responses: list[ValidatableResponse] = []

    while True:
        bt.logging.info(f"\tTesting {count} requests")

        def get_inputs():
            return ImageGenerationInputs(
                prompt=generate_random_prompt(),
                negative_prompt="blurry, nude, (out of focus), JPEG artifacts",
                width=1024,
                height=1024,
                num_inference_steps=30,
                controlnet_conditioning_scale=0.5,
                seed=int.from_bytes(os.urandom(4), "little"),
            )

        request_inputs = [
            get_inputs()
            for _ in range(count)
        ]

        responses: list[ImageGenerationSynapse] = list(
            await asyncio.gather(
                *[
                    validator.periodic_check_dendrite(
                        axons=axon,
                        synapse=ImageGenerationSynapse(inputs=inputs),
                        deserialize=False,
                        timeout=CLIENT_REQUEST_TIMEOUT,
                    )
                    for inputs in request_inputs
                ]
            )
        )

        slowest_response = max(
            responses,
            key=lambda response: (
                response.dendrite.process_time
                if response.dendrite and response.dendrite.process_time
                else -1
            ),
        )

        finished_responses.extend(
            [
                (response, inputs)
                for response, inputs in zip(responses, request_inputs)
                if response.output
            ]
        )

        error_count = [bool(response.output) for response in responses].count(False)

        if error_count == len(responses):
            if error_rate is not None:
                # Failed this pass, but succeeded a previous pass. We use the last pass's result
                break
            else:
                # Failed, mark as such
                await validator.metric_manager.failed_stress_test(uid)

                return

        error_rate = error_count / len(responses)

        response_time = slowest_response.dendrite.process_time

        if not response_time:
            break

        bt.logging.info(f"\t{count} requests generated in {response_time} with an error rate of {error_rate * 100}%")

        if error_rate >= MAX_ERROR_RATE or response_time > TIME_CONSTRAINT:
            break

        count *= 2

    check_count = max(1, int(len(finished_responses) * 0.125))

    scores = []

    for response, inputs in cryptographic_sample(finished_responses, check_count):
        score = reward(
            validator.gpu_semaphore,
            validator.pipeline,
            inputs,
            response,
        )

        scores.append(score)

    if len(scores):
        score = torch.tensor(scores).mean().item()
    else:
        score = 0.0

    await validator.metric_manager.successful_stress_test(
        uid,
        count,
        response_time,
        score,
        error_rate,
    )
