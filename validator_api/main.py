from typing import List

from pydantic import BaseModel
import uvicorn

from fastapi import FastAPI, Body

from validator_api.validator_pipeline import SDXLValidatorPipeline
from image_generation_protocol.io import ImageGenerationInputs


class ValidationInputs(BaseModel):
    input_parameters: ImageGenerationInputs
    frames: List


if __name__ == "__main__":
    app = FastAPI()

    pipeline: SDXLValidatorPipeline = (
        SDXLValidatorPipeline
        .from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        .to("cuda")
    )

    @app.post("/api/validate")
    def validate(inputs: ValidationInputs = Body()) -> float:
        return pipeline.validate(
            inputs.frames, inputs.input_parameters
        )

    uvicorn.run(app, host="0.0.0.0", port=8001)
