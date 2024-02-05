import base64
from io import BytesIO

from PIL import Image


def save_image_base64(image: Image.Image) -> bytes:
    with BytesIO() as output:
        image.save(output, format="jpeg")

        return base64.b64encode(output.getvalue())


def load_base64_image(data: bytes) -> Image.Image:
    with BytesIO(base64.b64decode(data)) as input_data:
        return Image.open(input_data)
