from asyncio import Semaphore
from datetime import datetime
from io import BytesIO
from typing import Annotated, Tuple, List

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, Body
from requests_toolbelt import MultipartEncoder
from safetensors.torch import save as save_tensor

from gpu_pipeline.pipeline import get_pipeline, SDXLPipelines, parse_input_parameters
from image_generation_protocol.io_protocol import ImageGenerationInputs
from starlette.responses import Response


def image_stream(image: Image.Image) -> BytesIO:
    output = BytesIO()
    image.save(output, format="jpeg")
    output.seek(0)

    return output


async def generate(
    gpu_semaphore: Semaphore,
    pipelines: SDXLPipelines,
    inputs: ImageGenerationInputs,
) -> Tuple[bytes, List[BytesIO]]:
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

    return frame_st_bytes, [image_stream(image) for image in output.images]


def main():
    app = FastAPI()

    gpu_semaphore, pipelines = get_pipeline()

    @app.post("/api/generate")
    async def generate_image(input_parameters: Annotated[ImageGenerationInputs, Body()]) -> Response:
        frames_bytes, images = await generate(gpu_semaphore, pipelines, input_parameters)

        multipart = MultipartEncoder(
            fields={
                "frames": frames_bytes,
                **{
                    f"image_{index}": image
                    for index, image in enumerate(images)
                },
            }
        )

        return Response(
            multipart.to_string(),
            media_type=multipart.content_type,
        )

    @app.get("/")
    def healthcheck():
        return datetime.utcnow()

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
