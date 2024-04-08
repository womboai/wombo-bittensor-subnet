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
import asyncio
import random
from typing import Any, TypeAlias

import bittensor as bt
import torch
from substrateinterface import Keypair

from image_generation_protocol.cryptographic_sample import cryptographic_sample
from image_generation_protocol.io_protocol import ImageGenerationInputs
from tensor.protocol import ImageGenerationSynapse
from tensor.timeouts import CLIENT_REQUEST_TIMEOUT
from validator.reward import select_endpoint, reward

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


async def get_base_weight(
    validator,
    uid: int,
    base_inputs: ImageGenerationInputs,
) -> float:
    axon = validator.metagraph.axons[uid]

    bt.logging.info(f"Measuring RPS of UID {uid} with Axon {axon}")

    count = 8
    rps: float | None = None

    finished_responses: list[ValidatableResponse] = []

    while True:
        bt.logging.info(f"\tTesting {count} requests")

        seed = random.randint(0, 2 ** 32)

        input_dict: dict[str, Any] = base_inputs.dict()

        input_dict.pop("seed")

        inputs = ImageGenerationInputs(**input_dict, seed=seed)

        responses: list[ImageGenerationSynapse] = await validator.periodic_check_dendrite(
            axons=[axon] * count,
            synapse=ImageGenerationSynapse(inputs=inputs),
            deserialize=False,
            timeout=CLIENT_REQUEST_TIMEOUT,
        )

        slowest_response = max(
            responses,
            key=lambda response: (
                response.dendrite.process_time
                if response.dendrite and response.dendrite.process_time
                else -1
            ),
        )

        finished_responses.extend([
            (response, inputs)
            for response in responses
            if response.output
        ])

        error_count = [bool(response.output) for response in responses].count(False)

        error_percentage = error_count / len(responses)

        response_time = slowest_response.dendrite.process_time

        if not response_time:
            break

        bt.logging.info(f"\t{count} requests generated in {response_time} with an error rate of {error_percentage * 100}%")

        rps = count / response_time

        if error_percentage >= MAX_ERROR_RATE or response_time > TIME_CONSTRAINT:
            break

        count *= 2

    if error_count == len(responses):
        await validator.send_metrics(
            "failure",
            {
                "miner_uid": uid,
                "generation_time": response_time,
                "concurrent_requests_processed": count,
            },
        )

        return 0.0

    validation_endpoint = select_endpoint(
        validator.config.validation_endpoint,
        validator.config.subtensor.network,
        "https://dev-validate.api.wombo.ai/api/validate",
        "https://validate.api.wombo.ai/api/validate",
    )

    keypair: Keypair = validator.periodic_check_dendrite.keypair
    hotkey = keypair.ss58_address
    signature = f"0x{keypair.sign(hotkey).hex()}"

    check_count = min(1, int(len(finished_responses) * 0.125))

    scores = await asyncio.gather(*[
        reward(
            validation_endpoint,
            hotkey,
            signature,
            inputs,
            response,
        )
        for response, inputs in cryptographic_sample(finished_responses, check_count)
    ])

    score = torch.tensor(scores).mean().item()

    await validator.send_metrics(
        "success",
        {
            "miner_uid": uid,
            "similarity_score": score,
            "generation_time": response_time,
            "concurrent_requests_processed": count,
            "error_rate": error_percentage,
        },
    )

    return score * rps * (1 - error_percentage)
