from datetime import datetime
from typing import Annotated

import uvicorn

from fastapi import FastAPI, Body

from gpu_generation.pipeline import get_pipeline
from validator_api import validate_frames
from image_generation_protocol.io_protocol import ValidationInputs, ValidationOutputs

if __name__ == "__main__":
    app = FastAPI()

    gpu_semaphore, pipeline = get_pipeline()

    @app.post("/api/validate")
    async def validate(inputs: Annotated[ValidationInputs, Body()]) -> ValidationOutputs:
        return await validate_frames(
            gpu_semaphore,
            pipeline,
            inputs.frames,
            inputs.input_parameters,
        )

    @app.get("/")
    def healthcheck():
        return datetime.utcnow()

    uvicorn.run(app, host="0.0.0.0", port=8001)
