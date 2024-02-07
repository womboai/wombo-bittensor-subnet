from asyncio import Semaphore
from typing import Tuple

from diffusers import StableDiffusionXLPipeline


def get_pipeline() -> Tuple[Semaphore, StableDiffusionXLPipeline]:
    pipeline = (
        StableDiffusionXLPipeline
        .from_pretrained("stablediffusionapi/newdream-sdxl-20")
        .to("cuda")
    )

    return Semaphore(), pipeline
