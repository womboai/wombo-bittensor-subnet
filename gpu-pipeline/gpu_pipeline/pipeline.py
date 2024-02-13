from asyncio import Semaphore
from collections import namedtuple
import os
from PIL import Image
import re
import requests
from typing import Tuple, Dict, Union

import cv2
import numpy as np
from diffusers import (
    StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel,
)
import torch

from image_generation_protocol.io_protocol import ImageGenerationInputs


SDXLPipelines = namedtuple('SDXLPipelines', ['t2i_pipe', 'cn_pipe'])
TAO_PATTERN = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in sorted([
    "bittensor symbol", "bittensor logo",
    "tao symbol", "tao logo",
    "tau symbol", "tau logo",
    "bittensor", "tao", "tau",
], key=len, reverse=True)) + r')\b'


def get_tao_img(width: int, height: int):
    tao_img = Image.open("tao.jpg")
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
    return new_img


def replace_keywords_with_tau_symbol(input_string):
    replaced_string = re.sub(TAO_PATTERN, "tau symbol", input_string, flags=re.IGNORECASE)
    return replaced_string


def parse_input_parameters(
    pipelines: SDXLPipelines,
    inputs: ImageGenerationInputs,
) -> Tuple[Union[StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline], Dict[str, any]]:
    input_kwargs = inputs.dict()
    input_kwargs["prompt"] = replace_keywords_with_tau_symbol(inputs.prompt)
    input_kwargs["generator"] = torch.Generator().manual_seed(input_kwargs.pop("seed"))
    input_kwargs["output_type"] = "pil"

    if inputs.controlnet_conditioning_scale > 0:
        selected_pipeline = pipelines.cn_pipe
        input_kwargs["image"] = get_tao_img(inputs.width, inputs.height)
    else:
        selected_pipeline = pipelines.t2i_pipe
        input_kwargs.pop("controlnet_conditioning_scale")

    return selected_pipeline, input_kwargs


def ensure_file_at_path(path: str, url: str) -> str:
    if not os.path.exists(path):
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        print(f"Downloading {url} to {path}")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    return path


def get_model_path() -> str:
    return ensure_file_at_path(
        path="../checkpoints/newdreamxl_v10.safetensors",
        url="https://civitai.com/api/download/models/173961",
    )


def get_tao_lora_path() -> str:
    return ensure_file_at_path(
        path="../checkpoints/bittensor_tao_lora.safetensors",
        url="https://d3j730xi5ph1dq.cloudfront.net/checkpoints/bittensor_tao_lora.safetensors",
    )


def get_pipeline() -> Tuple[Semaphore, SDXLPipelines]:
    device = "cuda"

    pipeline = (
        StableDiffusionXLPipeline
        .from_single_file(get_model_path())
        .to(device)
    )

    pipeline.load_lora_weights(get_tao_lora_path())
    pipeline.fuse_lora()

    cn_pipeline = StableDiffusionXLControlNetPipeline(
        **pipeline.components,
        controlnet=ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0"),
    ).to(device)

    return Semaphore(), SDXLPipelines(t2i_pipe=pipeline, cn_pipe=cn_pipeline)
