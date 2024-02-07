# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 WOMBO
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
from typing import List

import torch
from aiohttp import ClientSession

from image_generation_protocol.io_protocol import ImageGenerationInputs, ValidationInputs
from tensor.protocol import ImageGenerationOutputSynapse


async def reward(scoring_endpoint: str, query: ImageGenerationInputs, synapse: ImageGenerationOutputSynapse) -> float:
    """
    Reward the miner response to the generation request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """

    target_time = 0.09375
    time_reward = target_time / synapse.dendrite.process_time

    async with ClientSession() as session:
        data = ValidationInputs(
            input_parameters=query,
            frames=synapse.frames,
        )

        response = await session.post(
            scoring_endpoint,
            headers={"Content-Type": "application/json"},
            data=data.model_dump(),
        )

        score = await response.json()

    return score + time_reward


async def get_rewards(
    self,
    query: ImageGenerationInputs,
    responses: List[ImageGenerationOutputSynapse],
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.
    return torch.FloatTensor(
        await asyncio.gather(*[reward(self.config.scoring_endpoint, query, response) for response in responses])
    ).to(self.device)
