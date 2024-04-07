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

import base64
from typing import cast

import bittensor
from aiohttp import ClientSession, FormData, BasicAuth

from image_generation_protocol.io_protocol import ImageGenerationRequest, ImageGenerationOutput
from tensor.protocol import ImageGenerationSynapse


def select_endpoint(config: str, network: str, dev: str, prod: str) -> str:
    if config:
        return config
    elif network == "test":
        return dev
    else:
        return prod


async def reward(
    validation_endpoint: str,
    hotkey: str,
    signature: str,
    query: ImageGenerationRequest,
    synapse: ImageGenerationSynapse,
) -> float:
    """
    Reward the miner response to the generation request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """

    frames = cast(ImageGenerationOutput, synapse.output).frames

    if not frames:
        return 0.0

    async with ClientSession() as session:
        data = FormData()

        data.add_field(
            "input_parameters",
            query.json(),
            content_type="application/json",
        )

        data.add_field(
            "frames",
            base64.b64decode(frames),
            content_type="application/octet-stream",
        )

        async with session.post(
            validation_endpoint,
            auth=BasicAuth(hotkey, signature),
            data=data,
        ) as response:
            if response.status != 200:
                bittensor.logging.error(f"Failed to validate one output, error code {response.status} with error {await response.text()}")

                return 0.0

            score = await response.json()

    return score
