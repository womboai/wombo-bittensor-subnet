import argparse
from typing import Dict, Any, List

import bittensor as bt
import uvicorn
from fastapi import FastAPI, Body, HTTPException
from starlette import status

from utils.config import add_args
from utils.protocol import ImageGenerationSynapse
from utils.uids import get_random_uids


class Client:
    def __init__(self):
        parser = argparse.ArgumentParser()

        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        add_args(Client, parser)

        self.config = bt.config(parser)

        # The wallet holds the cryptographic key pairs for the miner.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # The metagraph holds the state of the network, letting us know about other validators and miners.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

    def generate(
        self, input_parameters: Dict[str, Any],
    ) -> List[bytes]:
        uid = get_random_uids(self, k=1)[0]

        # Grab the axon you're serving
        axon = self.metagraph.axons[uid]

        resp: ImageGenerationSynapse = self.dendrite.query(
            # Send the query to selected miner axon in the network.
            axons=[axon],
            synapse=ImageGenerationSynapse(input_parameters=input_parameters),
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
        )[0]

        if not resp.output_data:
            bt.logging.error(f"Failed to query subnetwork with {input_parameters} and axon {axon}")

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to query subnetwork",
            )

        _, images = resp.output_data

        return images


if __name__ == "__main__":
    app = FastAPI()
    client = Client()

    @app.post("/api/generate")
    def generate(input_parameters: Dict[str, Any] = Body()) -> List[bytes]:
        return client.generate(input_parameters)

    uvicorn.run(app)
