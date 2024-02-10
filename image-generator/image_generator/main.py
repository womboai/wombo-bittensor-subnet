import base64
from asyncio import Semaphore
from datetime import datetime
from io import BytesIO
from typing import Annotated, Tuple, List

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, Body
from safetensors.torch import save as save_tensor

from gpu_pipeline.pipeline import get_pipeline, SDXLPipelines, parse_input_parameters
from image_generation_protocol.io_protocol import ImageGenerationInputs, ImageGenerationOutput


def save_image_base64(image: Image.Image) -> bytes:
    with BytesIO() as output:
        image.save(output, format="jpeg")

        return base64.b64encode(output.getvalue())


async def generate(
    gpu_semaphore: Semaphore,
    pipelines: SDXLPipelines,
    inputs: ImageGenerationInputs,
) -> Tuple[bytes, List[bytes]]:
    frames = []

    def save_frames(_pipe, _step_index, _timestep, callback_kwargs):
        frames.append(callback_kwargs["latents"])
        return callback_kwargs

    selected_pipeline, input_kwargs = parse_input_parameters(pipelines, inputs)
    async with gpu_semaphore:
        output = selected_pipeline(
            **input_kwargs,
            callback_on_step_end=save_frames,
        )

    frame_st_bytes = save_tensor({"frames": torch.stack(frames)})

    return frame_st_bytes, [save_image_base64(image) for image in output.images]


def main():
    app = FastAPI()

    gpu_semaphore, pipelines = get_pipeline()

    @app.post("/api/generate")
    async def generate_image(input_parameters: Annotated[ImageGenerationInputs, Body()]) -> ImageGenerationOutput:
        frames_bytes, images = await generate(gpu_semaphore, pipelines, input_parameters)

        return ImageGenerationOutput(
            frames=frames_bytes,
            images=images,
        )

    @app.get("/")
    def healthcheck():
        return datetime.utcnow()

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
