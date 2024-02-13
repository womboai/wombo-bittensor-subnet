from datetime import datetime
from typing import Annotated

import uvicorn

from fastapi import FastAPI, Body, Form, File

from gpu_pipeline.pipeline import get_pipeline
from validator_api.validator_pipeline import validate_frames
from image_generation_protocol.io_protocol import ImageGenerationInputs


def main():
    app = FastAPI()

    gpu_semaphore, pipelines = get_pipeline()

    @app.post("/api/validate")
    async def validate(
        input_parameters: Annotated[ImageGenerationInputs, File()],
        frames: Annotated[bytes, File()],
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
