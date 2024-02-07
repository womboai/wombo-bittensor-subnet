from asyncio import Semaphore
import os
import requests
from typing import Tuple

from diffusers import StableDiffusionXLPipeline


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
        path="newdreamxl_v10.safetensors",
        url="https://civitai.com/api/download/models/173961",
    )


def get_tau_lora_path() -> str:
    return ensure_file_at_path(
        path="tau_lora_v3_epoch3.safetensors",
        url=str(None),
    )


def get_pipeline() -> Tuple[Semaphore, StableDiffusionXLPipeline]:
    pipeline = (
        StableDiffusionXLPipeline
        .from_pretrained(get_model_path())
        .to("cuda")
    )

    # TODO: uncomment once we're ready to use LORA
    # pipeline.load_lora_weights(get_tau_lora_path())
    # pipeline.fuse_lora()

    return Semaphore(), pipeline
