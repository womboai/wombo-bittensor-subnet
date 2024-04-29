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

from asyncio import Semaphore
from io import BytesIO

import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline

from gpu_pipeline.pipeline import parse_input_parameters
from gpu_pipeline.tensor import save_tensor
from image_generation_protocol.io_protocol import ImageGenerationInputs


def image_stream(image: Image.Image) -> BytesIO:
    output = BytesIO()
    image.save(output, format="jpeg")
    output.seek(0)

    return output


async def generate(
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

    if len(frames):
        frame_st_bytes = save_tensor(torch.stack(frames))
    else:
        frame_st_bytes = None

    return frame_st_bytes, [image_stream(image) for image in output.images]
