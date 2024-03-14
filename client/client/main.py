import asyncio
import os
import random
import traceback
from asyncio import Task
from datetime import datetime
from typing import List, Annotated

import bittensor as bt
import uvicorn
from bittensor import SubnetsAPI, TerminalInfo
from fastapi import FastAPI, Body, HTTPException, status
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from image_generation_protocol.io_protocol import ImageGenerationInputs

from tensor.config import config, check_config, add_args
from tensor.protocol import ImageGenerationClientSynapse
from neuron_selector.uids import get_best_uids, sync_neuron_info
from tensor.timeouts import CLIENT_REQUEST_TIMEOUT


class ValidatorQueryException(Exception):
    def __init__(self, queried_axons: list[TerminalInfo], dendrite_responses: list[TerminalInfo]):
        super().__init__(f"Failed to query subnetwork, dendrites {dendrite_responses}")
        self.queried_axons = queried_axons
        self.dendrite_responses = dendrite_responses


class WomboSubnetAPI(SubnetsAPI):
    def __init__(self):
        client_config = self.client_config()

        self.config = client_config
        self.check_config(client_config)

        # Set up logging with the provided configuration and directory.
        bt.logging(config=client_config, logging_dir=client_config.full_path)

        # Log the configuration for reference.
        bt.logging.info(client_config)

        super().__init__(bt.wallet(config=client_config))

        self.subtensor = bt.subtensor(config=client_config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # The metagraph holds the state of the network, letting us know about other validators and miners.
        self.metagraph = self.subtensor.metagraph(client_config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        self.metagraph.sync(subtensor=self.subtensor)

        self.periodic_metagraph_resync: Task
        self.neuron_info = {}

    def prepare_synapse(self, inputs: ImageGenerationInputs) -> ImageGenerationClientSynapse:
        return ImageGenerationClientSynapse(inputs=inputs)

    def process_responses(self, responses: List[ImageGenerationClientSynapse]) -> List[bytes]:
        bad_responses = []
        finished_responses = []

        for response in responses:
            if response.images:
                finished_responses.append(response)
            else:
                bad_responses.append(response)

        bad_axons = [response.axon for response in bad_responses]
        bad_dendrites = [response.dendrite for response in bad_responses]

        bt.logging.error(f"Failed to query validators with axons {bad_axons}, {bad_dendrites}")

        if not len(finished_responses):
            raise ValidatorQueryException(bad_axons, bad_dendrites)

        return finished_responses[random.randint(0, len(finished_responses) - 1)].images

    async def __aenter__(self):
        async def resync_metagraph():
            while True:
                """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
                bt.logging.info("resync_metagraph()")

                try:
                    # Sync the metagraph.
                    self.metagraph.sync(subtensor=self.subtensor)
                    await sync_neuron_info(self, self.dendrite)
                except Exception as _:
                    bt.logging.error("Failed to sync client metagraph, ", traceback.format_exc())

                await asyncio.sleep(30)

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
        validator_uids = get_best_uids(self, validators=True)

        if not len(validator_uids):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No suitable validators found",
            )

        axons = [self.metagraph.axons[uid] for uid in validator_uids]

        response = await self.query_api(
            axons,
            deserialize=False,
            timeout=CLIENT_REQUEST_TIMEOUT,
            inputs=input_parameters,
        )

        return response


async def main():
    app = FastAPI()

    @app.exception_handler(ValidatorQueryException)
    async def validator_query_handler(_: Request, exception: ValidatorQueryException) -> JSONResponse:
        return JSONResponse(
            content={
                "detail": str(exception),
                "axons": [axon.dict() for axon in exception.queried_axons],
                "dendrites": [dendrite.dict() for dendrite in exception.dendrite_responses],
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    async with WomboSubnetAPI() as client:
        @app.post("/api/generate")
        async def generate(input_parameters: Annotated[ImageGenerationInputs, Body()]) -> List[bytes]:
            return await client.generate(input_parameters)

        @app.get("/")
        def healthcheck():
            return datetime.utcnow()

        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", str(8000))))


if __name__ == "__main__":
    asyncio.run(main())
