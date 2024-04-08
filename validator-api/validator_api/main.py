import asyncio
import os
import traceback
from datetime import datetime
from typing import Annotated

import bittensor
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from pydantic import Json
from starlette import status
from substrateinterface import Keypair

from gpu_pipeline.pipeline import get_pipeline
from image_generation_protocol.io_protocol import ImageGenerationInputs
from validator_api.validator_pipeline import validate_frames

NETWORK = os.environ["NETWORK"]
NETUID = int(os.environ["NETUID"])


security = HTTPBasic()


def get_hotkey(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
    keypair = Keypair(ss58_address=credentials.username)

    if keypair.verify(credentials.username, credentials.password):
        return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Signature mismatch",
    )


async def main():
    app = FastAPI()

    gpu_semaphore, pipelines = get_pipeline()

    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)

    async def resync_metagraph():
        while True:
            """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
            bittensor.logging.info("resync_metagraph()")

            try:
                # Sync the metagraph.
                metagraph.sync(subtensor=subtensor)
            except Exception as _:
                bittensor.logging.error("Failed to sync metagraph, ", traceback.format_exc())

            await asyncio.sleep(1200)

    @app.post("/api/validate")
    async def validate(
        input_parameters: Annotated[Json[ImageGenerationInputs], Form(media_type="application/json")],
        frames: Annotated[UploadFile, File(media_type="application/octet-stream")],
        hotkey: Annotated[str, Depends(get_hotkey)],
    ) -> float:
        uid = metagraph.hotkeys.index(hotkey)

        if not metagraph.validator_permit[uid]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Validator permit required",
            )

        frames_bytes = await frames.read()
        return await validate_frames(
            gpu_semaphore,
            pipelines,
            frames_bytes,
            input_parameters,
        )

    @app.get("/")
    def healthcheck():
        return datetime.utcnow()

    server = asyncio.to_thread(
        uvicorn.run,
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", str(8001))),
    )

    metagraph_resync = resync_metagraph()

    await asyncio.gather(server, metagraph_resync)


if __name__ == "__main__":
    asyncio.run(main())
