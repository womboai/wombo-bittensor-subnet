import base64
from io import BytesIO

from PIL.Image import Image

from base.protocol import ImageGenerationSynapse

import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline


class SDXLMinerPipeline(StableDiffusionXLPipeline):
    def generate(self, **inputs):
        frames = []

        inputs["generator"] = torch.Generator().manual_seed(inputs["seed"])
        inputs["output_type"] = "pil"

        def save_frames(_pipe, _step_index, _timestep, callback_kwargs):
            frames.append(callback_kwargs["latents"])
            return callback_kwargs

        output = self(
            **inputs,
            callback_on_step_end=save_frames,
        )

        frames_tensor = torch.stack(frames)

        return frames_tensor, output.images


def _image_base64(image: Image) -> bytes:
    with BytesIO() as output:
        image.save(output, format=image.format)

        return base64.b64encode(output.getvalue())


def forward(self, request: ImageGenerationSynapse):
    frames_tensor, images = self.pipeline.generate(**request.input_parameters)

    images = [_image_base64(image) for image in images]

    request.output_data = frames_tensor.tolist(), images
