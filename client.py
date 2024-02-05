import argparse
from typing import Dict, Any, List, Tuple

import bittensor as bt
import torch
import uvicorn
from PIL.Image import Image
from fastapi import FastAPI, APIRouter

from base.protocol import ImageGenerationSynapse
from base.utils.uids import get_random_uids


class Client:
    def __init__(self):
        parser = argparse.ArgumentParser()

        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)

        # Netuid Arg: The netuid of the subnet to connect to.
        parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

        self.config = bt.config(parser)

        # The wallet holds the cryptographic key pairs for the miner.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # The metagraph holds the state of the network, letting us know about other validators and miners.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")


if __name__ == "__main__":
    app = FastAPI()
    client = Client()

    @app.post("/api/generate")
    def query_synapse(
        input_parameters: Dict[str, Any],
    ) -> List[Image]:
        uid = get_random_uids(client, k=1)[0]

        network = client.config.subtensor.network

        # instantiate the metagraph with provided network and netuid
        metagraph = bt.metagraph(
            netuid=client.config.netuid, network=network, sync=True, lite=False
        )

        # Grab the axon you're serving
        axon = metagraph.axons[uid]

        resp: Tuple[torch.Tensor, List[Image]] = client.dendrite.query(
            # Send the query to selected miner axon in the network.
            axons=[axon],
            synapse=ImageGenerationSynapse(input_parameters=input_parameters),
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=True,
        )[0]

        return resp[1]

    uvicorn.run(app, host="0.0.0.0", port=8080)
