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
from random import shuffle
from typing import TypeAlias, Annotated

import bittensor as bt
import nltk
import torch
from aiohttp import ClientSession, TCPConnector
from pydantic import BaseModel, Field

from image_generation_protocol.cryptographic_sample import cryptographic_sample
from image_generation_protocol.io_protocol import ImageGenerationInputs
from tensor.timeouts import CLIENT_REQUEST_TIMEOUT
from validator.miner_metrics import MinerMetricManager, parse_redis_value
from validator.score_protocol import OutputScoreRequest

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

ValidatableResponse: TypeAlias = tuple[bytes, ImageGenerationInputs]

WORDS = [word for word, tag in pos_tag(words.words(), tagset='universal') if tag == "ADJ" or tag == "NOUN"]
REQUEST_INCENTIVE = 0.0001


def generate_random_prompt():
    words = cryptographic_sample(WORDS, k=min(len(WORDS), min(os.urandom(1)[0] % 32, 8))) + ["tao"]
    shuffle(words)

    return ", ".join(words)


async def score_output(
    validator,
    inputs: ImageGenerationInputs,
    frames: bytes,
) -> float:
    axon = validator.metagraph.axons[validator.uid]
    return await validator.dendrite(
        axons=axon,
        synapse=OutputScoreRequest(inputs=inputs, frames=frames),
    )


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


class MinerStressTestMetricManager(MinerMetricManager):
    async def get(self, uid: int):
        (
            generated_count,
            generation_time,
            similarity_score,
            error_rate,
            successful_user_requests,
            failed_user_requests,
        ) = await self.validator.redis.mget(
            [
                f"generation_count_{uid}",
                f"generation_time_{uid}",
                f"similarity_score_{uid}",
                f"error_rate_{uid}",
                f"successful_user_requests_{uid}",
                f"failed_user_requests_{uid}",
            ]
        )

        generated_count = parse_redis_value(generated_count, int)

        if not generated_count:
            return None

        generation_time = parse_redis_value(generation_time, float)
        similarity_score = parse_redis_value(similarity_score, float)
        error_rate = parse_redis_value(error_rate, float)
        successful_user_requests = parse_redis_value(successful_user_requests, int)
        failed_user_requests = parse_redis_value(failed_user_requests, int)

        return MinerMetrics(
            generated_count=max(0, generated_count),
            generation_time=max(0.0, generation_time),
            similarity_score=max(0.0, min(1.0, similarity_score)),
            error_rate=max(0.0, min(1.0, error_rate)),
            successful_user_requests=max(0, successful_user_requests),
            failed_user_requests=max(0, failed_user_requests),
        )

    async def failed_miner(self, uid: int):
        await self.validator.redis.mset(
            {
                f"generation_count_{uid}": 0,
                f"generation_time_{uid}": 0.0,
                f"similarity_score_{uid}": 0.0,
                f"error_rate_{uid}": 0.0,
            }
        )

    async def reset(self, uid: int):
        await self.validator.redis.mset(
            {
                f"generation_count_{uid}": 0,
                f"generation_time_{uid}": 0.0,
                f"similarity_score_{uid}": 0.0,
                f"error_rate_{uid}": 0.0,
                f"successful_user_requests_{uid}": 0,
                f"failed_user_requests_{uid}": 0,
            }
        )

    async def successful_stress_test(
        self,
        uid: int,
        generated_count: int,
        generation_time: float,
        similarity_score: float,
        error_rate: float,
    ):
        await self.validator.redis.mset(
            {
                f"generation_count_{uid}": generated_count,
                f"generation_time_{uid}": generation_time,
                f"similarity_score_{uid}": similarity_score,
                f"error_rate_{uid}": error_rate,
            }
        )

        await self.send_metrics(
            self.validator.session,
            self.validator.dendrite,
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
        await self.failed_miner(uid)

        await self.send_metrics(
            self.validator.session,
            self.validator.dendrite,
            "failure",
            uid,
        )


async def stress_test_miner(validator, uid: int):
    if not validator.session:
        validator.session = ClientSession()

    blacklist = validator.config.blacklist
    axon = validator.metagraph.axons[uid]

    if blacklist and (axon.hotkey in blacklist.hotkeys or axon.coldkey in blacklist.coldkeys):
        validator.metric_manager.failed_miner(uid)
        return

    bt.logging.info(f"Measuring RPS of UID {uid} with Axon {axon}")

    count = 4
    error_rate: float | None = None

    finished_responses: list[ValidatableResponse] = []

    dendrite: bt.dendrite = validator.dendrite
    session = await dendrite.session

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
            )

        request_inputs = [
            get_inputs()
            for _ in range(count)
        ]

        if count > session.connector.limit:
            # Accessing private variables but can't find another way to do this
            await session.close()
            session._connector = TCPConnector(limit=count)

        responses: list[ImageGenerationSynapse] = list(
            await asyncio.gather(
                *[
                    validator.dendrite(
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

    check_count = max(1, int(len(finished_responses) * 0.0625))

    scores = await asyncio.gather(
        *[
            score_output(validator, inputs, response.output.frames)
            for response, inputs in cryptographic_sample(finished_responses, check_count)
        ]
    )

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
