import asyncio
import traceback
from asyncio import Task
from datetime import datetime
from typing import List, Optional, Annotated

import bittensor as bt
import uvicorn
from fastapi import FastAPI, Body, HTTPException
from image_generation_protocol.io_protocol import ImageGenerationInputs
from starlette import status

from tensor.config import config, check_config, add_args
from tensor.protocol import ImageGenerationClientSynapse
from neuron_selector.uids import get_random_uids, sync_neuron_info
from tensor.timeouts import CLIENT_REQUEST_TIMEOUT


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

        # The wallet holds the cryptographic key pairs for the miner.
        self.wallet = bt.wallet(config=client_config)
        bt.logging.info(f"Wallet: {self.wallet}")

        self.subtensor = bt.subtensor(config=client_config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # The metagraph holds the state of the network, letting us know about other validators and miners.
        self.metagraph = self.subtensor.metagraph(client_config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        self.metagraph.sync(subtensor=self.subtensor)

        self.periodic_metagraph_resync: Task
        self.neuron_info = {}

    async def __aenter__(self):
        async def resync_metagraph():
            while True:
                """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
                bt.logging.info("resync_metagraph()")

                try:
                    # Sync the metagraph.
                    self.metagraph.sync(subtensor=self.subtensor)
                    await sync_neuron_info(self)
                except Exception as _:
                    bt.logging.error("Failed to sync validator metagraph, ", traceback.format_exc())

                await asyncio.sleep(90)

        self.periodic_metagraph_resync = asyncio.get_event_loop().create_task(resync_metagraph())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.periodic_metagraph_resync.cancel()

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
        validator_uids = get_random_uids(self, k=1, validators=True)

        if not len(validator_uids):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No suitable validators found",
            )

        validator_uid = validator_uids[0]

        # Grab the axon you're serving
        axon = self.metagraph.axons[validator_uid]

        bt.logging.info(f"Sending request {input_parameters} to validator {validator_uid}, axon {axon}")

        async with self.dendrite as dendrite:
            response: Optional[ImageGenerationClientSynapse] = (await dendrite.forward(
                # Send the query to selected miner axon in the network.
                axons=[axon],
                synapse=ImageGenerationClientSynapse(inputs=input_parameters),
                # All responses have the deserialize function called on them before returning.
                # You are encouraged to define your own deserialization function.
                deserialize=False,
                timeout=CLIENT_REQUEST_TIMEOUT,
            ))[0]

        if not response.images:
            bt.logging.error(f"Failed to query subnetwork with {input_parameters} and axon {axon}")

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to query subnetwork",
            )

        return response.images


async def main():
    app = FastAPI()

    async with Client() as client:
        @app.post("/api/generate")
        async def generate(input_parameters: Annotated[ImageGenerationInputs, Body()]) -> List[bytes]:
            return await client.generate(input_parameters)

        @app.get("/")
        def healthcheck():
            return datetime.utcnow()

        uvicorn.run(app, host="0.0.0.0")


if __name__ == "__main__":
    asyncio.run(main())
