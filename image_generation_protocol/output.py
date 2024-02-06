from typing import List

from pydantic import BaseModel


class ImageGenerationOutput(BaseModel):
    frames: List
    images: List[bytes]
