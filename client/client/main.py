import asyncio
import logging
import os
import traceback
from asyncio import Task
from datetime import datetime
from typing import Annotated, AsyncGenerator, cast, TypeAlias

import bittensor as bt
import uvicorn
from bittensor import SubnetsAPI, TerminalInfo, AxonInfo
from fastapi import FastAPI, Body, HTTPException, status
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torch import tensor

from image_generation_protocol.io_protocol import ImageGenerationInputs
from neuron_selector.uids import get_best_uids, sync_neuron_info
from tensor.config import config, add_args
from tensor.protocol import ImageGenerationClientSynapse, MinerGenerationOutput
from tensor.timeouts import CLIENT_REQUEST_TIMEOUT

Axon: TypeAlias = AxonInfo | bt.axon


class ValidatorQueryException(Exception):
    def __init__(self, queried_axons: list[TerminalInfo], dendrite_responses: list[TerminalInfo]):
        super().__init__(f"Failed to query subnetwork, dendrites {dendrite_responses}")
        self.queried_axons = queried_axons
        self.dendrite_responses = dendrite_responses


class NeuronGenerationInfo(BaseModel):
    process_time: float
    uid: int
    hotkey: str


class ImageGenerationResult(BaseModel):
    images: list[bytes]
    validator_info: NeuronGenerationInfo
    miner_info: NeuronGenerationInfo


class ImageGenerationClientInputs(ImageGenerationInputs):
    watermark: bool = True
    miner_uid: int | None
    validator_uid: int | None


class WomboSubnetAPI(SubnetsAPI):
    def __init__(self):
        client_config = self.client_config()

        self.config = client_config
        bt.logging.check_config(client_config)

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
        asyncio.get_event_loop().run_until_complete(sync_neuron_info(self, self.dendrite))

        self.periodic_metagraph_resync: Task
        self.neuron_info = {}

    def prepare_synapse(self, inputs: ImageGenerationClientInputs) -> ImageGenerationClientSynapse:
        return ImageGenerationClientSynapse(
            inputs=inputs,
            watermark=inputs.watermark,
            miner_uid=inputs.miner_uid,
        )

    async def process_responses(
        self,
        responses: AsyncGenerator[ImageGenerationClientSynapse, None],
    ) -> ImageGenerationClientSynapse:
        bad_responses = []

        async for response in responses:
            if response.output:
                return response
            else:
                bad_responses.append(response)

        bad_axons = [response.axon for response in bad_responses]
        bad_dendrites = [response.dendrite for response in bad_responses]

        raise ValidatorQueryException(bad_axons, bad_dendrites)

    async def get_responses(
        self,
        axons: bt.axon | list[bt.axon],
        synapse: ImageGenerationClientSynapse,
        timeout: int,
    ) -> AsyncGenerator[ImageGenerationClientSynapse, None]:
        if isinstance(axons, list):
            responses = asyncio.as_completed([
                self.dendrite.forward(
                    axons=axon,
                    synapse=synapse,
                    deserialize=False,
                    timeout=timeout,
                )
                for axon in axons
            ])

            for future in responses:
                yield await future
        else:
            yield await self.dendrite(
                axons=axons,
                synapse=synapse,
                deserialize=False,
                timeout=timeout,
            )

    # noinspection PyMethodOverriding
    async def query_api(
        self,
        inputs: ImageGenerationClientInputs,
        axons: Axon | list[Axon],
        timeout: int = CLIENT_REQUEST_TIMEOUT,
    ) -> ImageGenerationClientSynapse:
        synapse = self.prepare_synapse(inputs)
        bt.logging.debug(f"Querying validator axons with synapse {synapse.name}...")

        return await self.process_responses(self.get_responses(axons, synapse, timeout))

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

                await asyncio.sleep(1200)

        self.periodic_metagraph_resync = asyncio.get_event_loop().create_task(resync_metagraph())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.periodic_metagraph_resync.cancel()

    @classmethod
    def client_config(cls):
        return config(add_args)

    async def generate(
        self,
        input_parameters: ImageGenerationClientInputs,
    ) -> ImageGenerationResult:
        validator_uids = (
            get_best_uids(self.config.blacklist, self.metagraph, self.neuron_info, validators=True)
            if input_parameters.validator_uid is None
            else tensor([input_parameters.validator_uid])
        )

        if not len(validator_uids):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No suitable validators found",
            )

        axons: list[AxonInfo] = [self.metagraph.axons[uid.item()] for uid in validator_uids]

        axon_uids = {
            axon.hotkey: uid.item()
            for uid, axon in zip(validator_uids, axons)
        }

        response = await self.query_api(
            input_parameters,
            axons,
        )

        output = cast(MinerGenerationOutput, response.output)

        return ImageGenerationResult(
            images=output.images,
            validator_info=NeuronGenerationInfo(
                process_time=response.dendrite.process_time,
                uid=axon_uids[response.axon.hotkey],
                hotkey=response.axon.hotkey,
            ),
            miner_info=NeuronGenerationInfo(
                process_time=output.process_time,
                uid=output.miner_uid,
                hotkey=output.miner_hotkey,
            ),
        )


async def main():
    app = FastAPI()
    logger = logging.getLogger("client")

    @app.exception_handler(ValidatorQueryException)
    async def validator_query_handler(_: Request, exception: ValidatorQueryException) -> JSONResponse:
        logger.error("Subnetwork error", exc_info=exception)

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
        async def generate(input_parameters: Annotated[ImageGenerationClientInputs, Body()]) -> ImageGenerationResult:
            return await client.generate(input_parameters)

        @app.get("/")
        def healthcheck():
            return datetime.utcnow()

        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", str(8000))))


if __name__ == "__main__":
    asyncio.run(main())
