import asyncio
import os
from datetime import datetime
from typing import Annotated

import bittensor
import uvicorn

from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBasicCredentials
from pydantic import Json
from starlette import status
from substrateinterface import Keypair

from gpu_pipeline.pipeline import get_pipeline
from validator_api.validator_pipeline import validate_frames
from image_generation_protocol.io_protocol import ImageGenerationInputs


NETWORK = os.environ["NETWORK"]
NETUID = int(os.environ["NETUID"])


def get_hotkey(credentials: HTTPBasicCredentials) -> str:
    keypair = Keypair(ss58_address=credentials.username)

    if keypair.verify(credentials.username, credentials.password):
        return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Signature mismatch",
    )


def main():
    app = FastAPI()

    gpu_semaphore, pipelines = get_pipeline()

    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)

    async def resync_metagraph():
        while True:
            """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
            bittensor.logging.info("resync_metagraph()")

            # Sync the metagraph.
            metagraph.sync(subtensor=subtensor)

            await asyncio.sleep(90)

    asyncio.get_event_loop().create_task(resync_metagraph())

    @app.post("/api/validate")
    async def validate(
        input_parameters: Annotated[Json[ImageGenerationInputs], Form(media_type="application/json")],
        frames: Annotated[UploadFile, File(media_type="application/octet-stream")],
        hotkey: str = Depends(get_hotkey),
    ) -> float:
        uid = metagraph.hotkeys.index(hotkey)

        if not metagraph.validator_permit[uid]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Validator permit required",
            )

        frames = await frames.read()
        return await validate_frames(
            gpu_semaphore,
            pipelines,
            frames,
            input_parameters,
        )

    @app.get("/")
    def healthcheck():
        return datetime.utcnow()

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
