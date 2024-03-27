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

import random
import sys

import bittensor as bt
from substrateinterface import Keypair

from image_generation_protocol.io_protocol import ImageGenerationInputs, ImageGenerationRequest
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


async def get_base_weight(
    uid: int,
    base_inputs: ImageGenerationInputs,
    metagraph: bt.metagraph,
    dendrite: bt.dendrite,
    config: bt.config,
) -> float:
    axon = metagraph.axons[uid]

    bt.logging.info(f"Measuring RPS of UID {uid} with Axon {axon}")

    count = 8
    rps: float | None = None

    while True:
        seed = random.randint(0, 2 ** 32)

        inputs = ImageGenerationInputs(**base_inputs.dict(), seed=seed)

        num_random_indices = 3
        step_indices = sorted(random.sample(
            range(base_inputs.num_inference_steps),
            k=num_random_indices,
        ))

        responses: list[ImageGenerationSynapse] = await dendrite(
            axons=[axon] * count,
            synapse=ImageGenerationSynapse(
                inputs=inputs,
                step_indices=step_indices,
            ),
            deserialize=False,
            timeout=CLIENT_REQUEST_TIMEOUT,
        )

        slowest_response = max(
            responses,
            key=lambda response: (
                response.dendrite.process_time
                if response.dendrite
                else -1
            ),
        )

        fastest_response = min(
            responses,
            key=lambda response: (
                response.dendrite.process_time
                if response.dendrite
                else sys.float_info.max
            ),
        )

        error_count = [bool(response.output) for response in responses].count(False)

        error_rate = error_count / len(responses)

        if error_rate >= MAX_ERROR_RATE:
            break

        if not slowest_response.dendrite:
            break

        response_time = slowest_response.dendrite.process_time

        if response_time > TIME_CONSTRAINT:
            break

        rps = count / response_time

        count *= 2

    if not rps:
        return 0.0

    validation_endpoint = select_endpoint(
        config.validation_endpoint,
        config.subtensor.network,
        "https://dev-validate.api.wombo.ai/api/validate",
        "https://validate.api.wombo.ai/api/validate",
    )

    keypair: Keypair = dendrite.keypair
    hotkey = keypair.ss58_address
    signature = f"0x{keypair.sign(hotkey).hex()}"

    score = await reward(
        validation_endpoint,
        hotkey,
        signature,
        ImageGenerationRequest(
            inputs=inputs,
            step_indices=step_indices,
        ),
        fastest_response,
    )

    return score * rps
