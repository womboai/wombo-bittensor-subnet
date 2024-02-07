import base64
from io import BytesIO
from typing import Annotated

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, Body

from gpu_generation.pipeline import get_pipeline
from image_generation_protocol.io_protocol import ImageGenerationInputs, ImageGenerationOutput


def save_image_base64(image: Image.Image) -> bytes:
    with BytesIO() as output:
        image.save(output, format="jpeg")

        return base64.b64encode(output.getvalue())


if __name__ == "__main__":
    app = FastAPI()

    gpu_semaphore, pipeline = get_pipeline()


    async def generate(**inputs):
        frames = []

        inputs["generator"] = torch.Generator().manual_seed(inputs["seed"])
        inputs["output_type"] = "pil"

        def save_frames(_pipe, _step_index, _timestep, callback_kwargs):
            frames.append(callback_kwargs["latents"])
            return callback_kwargs

        async with gpu_semaphore:
            output = pipeline(
                **inputs,
                callback_on_step_end=save_frames,
            )

        frames_tensor = torch.stack(frames)

        return frames_tensor, output.images

    @app.post("/api/generate")
    async def generate_image(input_parameters: Annotated[ImageGenerationInputs, Body()]) -> ImageGenerationOutput:
        frames_tensor, images = await generate(**input_parameters.model_dump())

        return ImageGenerationOutput(
            frames=frames_tensor.tolist(),
            images=[save_image_base64(image) for image in images]
        )


    uvicorn.run(app, host="0.0.0.0", port=8001)
