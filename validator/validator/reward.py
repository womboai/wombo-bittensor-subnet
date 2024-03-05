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
import base64
from typing import List, Tuple

import torch
from aiohttp import ClientSession, FormData, BasicAuth
from substrateinterface import Keypair

from image_generation_protocol.io_protocol import ImageGenerationInputs

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
    query: ImageGenerationInputs,
    synapse: ImageGenerationSynapse,
) -> float:
    """
    Reward the miner response to the generation request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """

    target_time = 5
    worst_time = 12
    time_difference = synapse.dendrite.process_time - target_time
    time_penalty = max(0.0, min(0.5, (time_difference * 4.0) / (worst_time * target_time * 3.0)))

    async with ClientSession() as session:
        data = FormData()

        data.add_field(
            "input_parameters",
            query.json(),
            content_type="application/json",
        )

        data.add_field(
            "frames",
            base64.b64decode(synapse.output.frames),
            content_type="application/octet-stream",
        )

        async with session.post(
            validation_endpoint,
            auth=BasicAuth(hotkey, signature),
            data=data,
        ) as response:
            response.raise_for_status()

            score = await response.json()

    return max(0.0, score - time_penalty)


async def get_rewards(
    self,
    query: ImageGenerationInputs,
    uids: List[int],
    responses: List[ImageGenerationSynapse],
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """

    validation_endpoint = select_endpoint(
        self.config.validation_endpoint,
        self.config.subtensor.network,
        "https://dev-validate.api.wombo.ai/api/validate",
        "https://validate.api.wombo.ai/api/validate",
    )

    are_wombo_neurons_endpoint = select_endpoint(
        self.config.are_wombo_neurons_endpoint,
        self.config.subtensor.network,
        "https://dev-neuron-identifier.api.wombo.ai/api/are_wombo_neurons",
        "https://neuron-identifier.api.wombo.ai/api/are_wombo_neurons",
    )

    keypair: Keypair = self.dendrite.keypair
    hotkey = keypair.ss58_address
    signature = f"0x{keypair.sign(hotkey).hex()}"

    async with ClientSession() as session:
        uids_query = ",".join([str(uid) for uid in uids])

        async with session.get(
            f"{are_wombo_neurons_endpoint}?uids={uids_query}",
            headers={"Content-Type": "application/json"},
        ) as response:
            response.raise_for_status()

            wombo_advantages = [(int(is_wombo_neuron) - 1) * 0.1 for is_wombo_neuron in await response.json()]

    # Get all the reward results by iteratively calling your reward() function.
    rewards = await asyncio.gather(*[
        reward(
            validation_endpoint,
            hotkey,
            signature,
            query,
            response,
        )
        for response in responses
    ])

    scores = [
        max(0.0, uid_reward + wombo_advantage)
        for uid_reward, wombo_advantage in zip(rewards, wombo_advantages)
    ]

    return torch.FloatTensor(scores).to(self.device)
