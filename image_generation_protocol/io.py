import random
from typing import List, Optional, Annotated

from pydantic import BaseModel, Field


DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 1344
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 7.0

MIN_SIZE = 512
MAX_SIZE = 1536
MAX_STEPS = 100
MIN_IMAGES = 1
MAX_IMAGES = 4


GenerationResolution = Annotated[int, Field(ge=MIN_SIZE, le=MAX_SIZE)]


class ImageGenerationInputs(BaseModel):
    prompt: str
    prompt_2: Optional[str] = None
    height: Optional[GenerationResolution] = DEFAULT_HEIGHT
    width: Optional[GenerationResolution] = DEFAULT_WIDTH
    num_inference_steps: Optional[Annotated[int, Field(gt=0, le=MAX_STEPS)]] = DEFAULT_STEPS
    guidance_scale: Optional[float] = DEFAULT_GUIDANCE
    negative_prompt: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    num_images_per_prompt: Optional[Annotated[int, Field(ge=MIN_IMAGES, le=MAX_IMAGES)]] = MIN_IMAGES
    seed: Optional[int] = Field(default_factory=lambda: random.randint(0, int(1e9)))


class ImageGenerationOutput(BaseModel):
    frames: List
    images: List[bytes]
