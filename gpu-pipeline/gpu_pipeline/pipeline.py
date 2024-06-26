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
import re
from asyncio import Semaphore
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel, DPMSolverMultistepScheduler,
)

from tensor.protos.inputs_pb2 import GenerationRequestInputs

TAO_IMAGE_CACHE: dict[(int, int), Image.Image] = {}

TAO_PATTERN = r'\b(?:' + '|'.join(
    re.escape(keyword) for keyword in sorted(
        [
            "bittensor symbol", "bittensor logo",
            "tao symbol", "tao logo",
            "tau symbol", "tau logo",
            "bittensor", "tao", "tau",
        ], key=len, reverse=True
    )
) + r')\b'


def size_key(size: int):
    return size / 8 - 64


def get_tao_img(width: int, height: int):
    cache_key = (size_key(width), size_key(height))

    cached = TAO_IMAGE_CACHE.get(cache_key)

    if cached:
        return cached

    tao_img = Image.open(Path(__file__).parent.parent / "tao.jpg")
    scale_factor = min(width / tao_img.width, height / tao_img.height)
    tao_img = tao_img.resize((int(tao_img.width * scale_factor), int(tao_img.height * scale_factor)))
    new_img = Image.new("RGB", (width, height), (255, 255, 255))
    new_img.paste(tao_img, (int((width - tao_img.width) / 2), int((height - tao_img.height) * 0.2)))
    new_img = Image.fromarray(255 - np.array(new_img))
    image = np.array(new_img)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    new_img = Image.fromarray(image)

    TAO_IMAGE_CACHE[cache_key] = new_img

    return new_img


def replace_keywords_with_tau_symbol(input_string):
    replaced_string = re.sub(TAO_PATTERN, "tau symbol", input_string, flags=re.IGNORECASE)
    return replaced_string


def parse_input_parameters(inputs: GenerationRequestInputs, device) -> dict[str, Any]:
    input_kwargs = {
        "prompt": replace_keywords_with_tau_symbol(inputs.prompt),
        "prompt_2": inputs.prompt_2,
        "width": inputs.width,
        "height": inputs.height,
        "num_inference_steps": inputs.num_inference_steps,
        "guidance_scale": inputs.guidance_scale,
        "negative_prompt": inputs.negative_prompt,
        "negative_prompt_2": inputs.negative_prompt_2,
        "controlnet_conditioning_scale": inputs.controlnet_conditioning_scale,
        "image": get_tao_img(inputs.width, inputs.height),
        "output_type": "latent",
    }

    if inputs.seed:
        input_kwargs["generator"] = torch.Generator(device).manual_seed(inputs.seed)

    return input_kwargs


def ensure_file_at_path(path: str, url: str) -> str:
    full_path = Path(__file__).parent.parent.parent / "checkpoints" / path

    if not os.path.exists(full_path):
        full_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} to {full_path}")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(full_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    return str(full_path)


def get_model_path() -> str:
    return ensure_file_at_path(
        path="newdreamxl_v10.safetensors",
        url="https://civitai.com/api/download/models/173961",
    )


def get_tao_lora_path() -> str:
    return ensure_file_at_path(
        path="bittensor_tao_lora.safetensors",
        url="https://d3j730xi5ph1dq.cloudfront.net/checkpoints/bittensor_tao_lora.safetensors",
    )


def get_pipeline(device: str | None) -> tuple[Semaphore, StableDiffusionXLControlNetPipeline]:
    concurrency = int(os.getenv("CONCURRENCY", str(1)))

    pipeline = (
        StableDiffusionXLPipeline
        .from_single_file(get_model_path(), torch_dtype=torch.float16)
        .to(device)
    )

    pipeline.load_lora_weights(get_tao_lora_path())
    pipeline.fuse_lora()

    pipeline.scheduler = DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++",
    )

    cn_pipeline = StableDiffusionXLControlNetPipeline(
        **pipeline.components,
        controlnet=ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        ),
    ).to(device)

    return Semaphore(concurrency), cn_pipeline
