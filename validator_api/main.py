from typing import Dict, Any, List

from pydantic import BaseModel
import uvicorn

from fastapi import FastAPI, Body

from validator_api.validator_pipeline import SDXLValidatorPipeline
from image_generation_protocol.io import ImageGenerationOutput


class ValidationInputs(BaseModel):
    input_parameters: Dict[str, Any]
    frames: List


class ValidationOutputs(BaseModel):
    validity_score: float


if __name__ == "__main__":
    app = FastAPI()

    pipeline: SDXLValidatorPipeline = (
        SDXLValidatorPipeline
        .from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        .to("cuda")
    )

    @app.post("/api/validate")
    def validate(inputs: ValidationInputs = Body()) -> ImageGenerationOutput:
        return ValidationOutputs(validity_score=pipeline.validate(
            inputs.frames, inputs.input_parameters
        ))

    uvicorn.run(app, host="0.0.0.0", port=8001)
