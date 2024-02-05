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


import random
import torch
import time
from typing import List

# Bittensor
import bittensor as bt

from tensor.protocol import ImageGenerationSynapse
from tensor.uids import get_random_uids

# import base validator class which takes care of most of the boilerplate
from validator.validator import BaseValidatorNeuron
from validator.reward import get_rewards


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

    async def forward(self):
        """
        Validator forward pass, called by the validator every time step. Consists of:
        - Generating the query
        - Querying the network miners
        - Getting the responses
        - Rewarding the miners based on their responses
        - Updating the scores
        """

        # TODO(developer): Define how the validator selects a miner to query, how often, etc.
        # get_random_uids is an example method, but you can replace it with your own.
        miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

        if not len(miner_uids):
            return

        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        max_seed = 2 ** 32
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
        responses: List[ImageGenerationSynapse] = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=axons,
            synapse=ImageGenerationSynapse(input_parameters=input_parameters),
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
        )

        working_miner_uids = []
        finished_responses = []

        for uid, response in zip(miner_uids, responses):
            if not response.output_data:
                continue

            working_miner_uids.append(uid)
            finished_responses.append(response)

        # Log the results for monitoring purposes.
        bt.logging.info(f"Received responses: {finished_responses}")

        if not len(finished_responses):
            return

        # Adjust the scores based on responses from miners.
        rewards = await get_rewards(
            self,
            query={
                **input_parameters,
                "generator": torch.Generator().manual_seed(seed),
            },
            responses=finished_responses
        )

        bt.logging.info(f"Scored responses: {rewards}")
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        self.update_scores(rewards, miner_uids)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)
