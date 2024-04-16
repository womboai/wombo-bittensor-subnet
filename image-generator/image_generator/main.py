import os
from asyncio import Semaphore
from datetime import datetime
from io import BytesIO
from typing import Annotated

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, Body
from requests_toolbelt import MultipartEncoder
from starlette.responses import Response

from gpu_pipeline.pipeline import get_pipeline, SDXLPipelines, parse_input_parameters
from gpu_pipeline.tensor import save_tensor
from image_generation_protocol.io_protocol import ImageGenerationInputs


def image_stream(image: Image.Image) -> BytesIO:
    output = BytesIO()
    image.save(output, format="jpeg")
    output.seek(0)

    return output


async def generate(
    gpu_semaphore: Semaphore,
    pipelines: SDXLPipelines,
    inputs: ImageGenerationInputs,
) -> tuple[bytes, list[BytesIO]]:
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

    if len(frames):
        frame_st_bytes = save_tensor(torch.stack(frames))
    else:
        frame_st_bytes = None

    return frame_st_bytes, [image_stream(image) for image in output.images]


def main():
    app = FastAPI()

    gpu_semaphore, pipelines = get_pipeline()

    @app.post("/api/generate")
    async def generate_image(inputs: Annotated[ImageGenerationInputs, Body()]) -> Response:
        frames_bytes, images = await generate(gpu_semaphore, pipelines, inputs)

        fields = {
            f"image_{index}": (None, image, "image/jpeg")
            for index, image in enumerate(images)
        }

        if frames_bytes:
            fields["frames"] = (None, BytesIO(frames_bytes), "application/octet-stream")

        multipart = MultipartEncoder(fields=fields)

        return Response(
            multipart.to_string(),
            media_type=multipart.content_type,
        )

    @app.get("/")
    def healthcheck():
        return datetime.utcnow()

    uvicorn.run(app, host=os.getenv("BIND_IP", "0.0.0.0"), port=int(os.getenv("PORT", str(8001))))


if __name__ == "__main__":
    main()
