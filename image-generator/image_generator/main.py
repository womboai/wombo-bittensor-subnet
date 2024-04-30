#  The MIT License (MIT)
#  Copyright © 2023 Yuma Rao
#  Copyright © 2024 WOMBO
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#
#

import os
from asyncio import Semaphore
from datetime import datetime
from io import BytesIO
from typing import Annotated

import torch
import uvicorn
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from fastapi import FastAPI, Body
from requests_toolbelt import MultipartEncoder
from starlette.responses import Response
from transformers import CLIPConfig, CLIPImageProcessor

from gpu_pipeline.pipeline import get_pipeline, parse_input_parameters
from gpu_pipeline.tensor import save_tensor
from image_generation_protocol.io_protocol import ImageGenerationInputs


def image_stream(image: Image.Image) -> BytesIO:
    output = BytesIO()
    image.save(output, format="jpeg")
    output.seek(0)

    return output


async def generate(
    image_processor: CLIPImageProcessor,
    safety_checker: StableDiffusionSafetyChecker,
    gpu_semaphore: Semaphore,
    pipeline: StableDiffusionXLControlNetPipeline,
    inputs: ImageGenerationInputs,
) -> tuple[bytes, list[BytesIO]]:
    frames = []

    def save_frames(_pipe, _step_index, _timestep, callback_kwargs):
        frames.append(callback_kwargs["latents"])

        return callback_kwargs

    input_kwargs = parse_input_parameters(inputs)

    async with gpu_semaphore:
        output = pipeline(
            **input_kwargs,
            callback_on_step_end=save_frames,
        )

    image = output.images[0]

    safety_checker_input = image_processor(image, return_tensors="pt").to(device=safety_checker.device)

    [image], _ = safety_checker(
        images=[image],
        clip_input=safety_checker_input.pixel_values.to(torch.float16),
    )

    if len(frames):
        frame_st_bytes = save_tensor(torch.stack(frames))
    else:
        frame_st_bytes = None

    return frame_st_bytes, [image_stream(image)]


def main():
    app = FastAPI()

    device = os.getenv("DEVICE", "cuda")
    concurrency, pipeline = get_pipeline(device)
    gpu_semaphore = Semaphore(concurrency)
    image_processor = pipeline.feature_extractor or CLIPImageProcessor()
    safety_checker = StableDiffusionSafetyChecker(CLIPConfig()).to(device)

    @app.post("/api/generate")
    async def generate_image(inputs: Annotated[ImageGenerationInputs, Body()]) -> Response:
        frames_bytes, images = await generate(
            image_processor,
            safety_checker,
            gpu_semaphore,
            pipeline,
            inputs,
        )

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
