from datetime import datetime
from typing import Annotated

import uvicorn

from fastapi import FastAPI, Form, File
from pydantic import Json

from gpu_pipeline.pipeline import get_pipeline
from validator_api.validator_pipeline import validate_frames
from image_generation_protocol.io_protocol import ImageGenerationInputs


def main():
    app = FastAPI()

    gpu_semaphore, pipelines = get_pipeline()

    @app.post("/api/validate")
    async def validate(
        input_parameters: Annotated[Json[ImageGenerationInputs], Form(media_type="application/json")],
        frames: Annotated[bytes, File(media_type="application/octet-stream")],
    ) -> float:
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
