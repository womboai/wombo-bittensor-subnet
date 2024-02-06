import base64
from io import BytesIO
from typing import Dict, Any

import torch
import uvicorn
from PIL import Image

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline
)
from fastapi import FastAPI, Body

from image_generation_protocol.output import ImageGenerationOutput


class SDXLMinerPipeline(StableDiffusionXLPipeline):
    def generate(self, **inputs):
        frames = []

        inputs["generator"] = torch.Generator().manual_seed(inputs["seed"])
        inputs["output_type"] = "pil"

        def save_frames(_pipe, _step_index, _timestep, callback_kwargs):
            frames.append(callback_kwargs["latents"])
            return callback_kwargs

        output = self(
            **inputs,
            callback_on_step_end=save_frames,
        )

        frames_tensor = torch.stack(frames)

        return frames_tensor, output.images


def save_image_base64(image: Image.Image) -> bytes:
    with BytesIO() as output:
        image.save(output, format="jpeg")

        return base64.b64encode(output.getvalue())


if __name__ == "__main__":
    app = FastAPI()

    pipeline: SDXLMinerPipeline = (
        SDXLMinerPipeline
        .from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        .to("cuda")
    )

    @app.post("/api/generate")
    def generate(input_parameters: Dict[str, Any] = Body()) -> ImageGenerationOutput:
        frames_tensor, images = pipeline.generate(**input_parameters)

        return ImageGenerationOutput(
            frames=frames_tensor.tolist(),
            images=[save_image_base64(image) for image in images]
        )


    uvicorn.run(app, host="0.0.0.0", port=8001)
