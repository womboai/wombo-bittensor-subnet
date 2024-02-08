from datetime import datetime
from typing import Annotated

import uvicorn

from fastapi import FastAPI, Body

from gpu_pipeline.pipeline import get_pipeline
from starlette.responses import JSONResponse
from validator_api.validator_pipeline import validate_frames
from image_generation_protocol.io_protocol import ValidationInputs


def main():
    app = FastAPI()

    gpu_semaphore, pipeline = get_pipeline()

    @app.post("/api/validate")
    async def validate(inputs: Annotated[ValidationInputs, Body()]) -> JSONResponse:
        return JSONResponse(await validate_frames(
            gpu_semaphore,
            pipeline,
            inputs.frames,
            inputs.input_parameters,
        ))

    @app.get("/")
    def healthcheck():
        return datetime.utcnow()

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
