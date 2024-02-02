# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 WOMBO
import asyncio
import random
import torch

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

import bittensor as bt

from base.protocol import ImageGenerationSynapse
from base.validator.reward import get_rewards
from base.utils.uids import get_random_uids


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    axons = [self.metagraph.axons[uid] for uid in miner_uids]

    max_seed = 2**32
    random_int = random.randint(0, max_seed)
    seed = (self.step * random_int) % max_seed

    input_parameters = {
        "prompt": f"Test Apples {self.step} * {random_int}",
        "seed": seed,
        "width": 512,
        "height": 512,
        "steps": 15,
    }

    bt.logging.info(f"Sending request {input_parameters} to {miner_uids} which have axons {axons}")

    # The dendrite client queries the network.
    responses = self.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=axons,
        synapse=ImageGenerationSynapse(input_parameters=input_parameters),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        deserialize=True,
    )

    if responses is None:
        bt.logging.error(f"Received {None} response when querying for {input_parameters}")

        await asyncio.sleep(12)

        return

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")

    # Adjust the scores based on responses from miners.
    rewards = get_rewards(
        self,
        query={
            **input_parameters,
            "generator": torch.Generator().manual_seed(seed),
        },
        responses=responses
    )

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)
