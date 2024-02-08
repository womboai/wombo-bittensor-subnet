from datetime import datetime
from typing import List, Optional, Annotated

import bittensor as bt
import uvicorn
from fastapi import FastAPI, Body, HTTPException
from starlette import status

from image_generation_protocol.io_protocol import ImageGenerationInputs
from tensor.config import config, check_config, add_args
from tensor.protocol import ImageGenerationRequestSynapse, ImageGenerationOutputSynapse
from tensor.uids import get_random_uids, is_validator


class Client:
    def __init__(self):
        client_config = self.client_config()

        self.config = client_config
        self.check_config(client_config)

        # Set up logging with the provided configuration and directory.
        bt.logging(config=client_config, logging_dir=client_config.full_path)

        # Log the configuration for reference.
        bt.logging.info(client_config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects.")

        self.subtensor = bt.subtensor(config=client_config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # The metagraph holds the state of the network, letting us know about other validators and miners.
        self.metagraph = self.subtensor.metagraph(client_config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite()
        bt.logging.info(f"Dendrite: {self.dendrite}")

    @classmethod
    def check_config(cls, client_config: "bt.Config"):
        check_config(cls, client_config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def client_config(cls):
        return config(cls)

    async def generate(
        self,
        input_parameters: ImageGenerationInputs,
    ) -> List[bytes]:
        validator_uid = get_random_uids(self, k=1, availability_checker=is_validator)[0]

        # Grab the axon you're serving
        axon = self.metagraph.axons[validator_uid]

        bt.logging.info(f"Sending request {input_parameters} to validator {validator_uid}, axon {axon}")

        async with self.dendrite as dendrite:
            resp: Optional[ImageGenerationOutputSynapse] = (await dendrite.forward(
                # Send the query to selected miner axon in the network.
                axons=[axon],
                synapse=ImageGenerationRequestSynapse(**input_parameters.model_dump()),
                # All responses have the deserialize function called on them before returning.
                # You are encouraged to define your own deserialization function.
                deserialize=False,
            ))[0]

        if not resp:
            bt.logging.error(f"Failed to query subnetwork with {input_parameters} and axon {axon}")

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to query subnetwork",
            )

        return resp.images


def main():
    app = FastAPI()
    client = Client()

    @app.post("/api/generate")
    async def generate(input_parameters: Annotated[ImageGenerationInputs, Body()]) -> List[bytes]:
        return await client.generate(input_parameters)

    @app.get("/")
    def healthcheck():
        return datetime.utcnow()

    uvicorn.run(app, host="0.0.0.0")


if __name__ == "__main__":
    main()
