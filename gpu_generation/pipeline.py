from asyncio import Semaphore
from typing import Tuple

from diffusers import StableDiffusionXLPipeline


def get_pipeline() -> Tuple[Semaphore, StableDiffusionXLPipeline]:
    pipeline = (
        StableDiffusionXLPipeline
        .from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        .to("cuda")
    )

    return Semaphore(), pipeline
