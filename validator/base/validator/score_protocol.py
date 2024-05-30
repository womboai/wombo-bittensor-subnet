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

from inspect import Parameter, Signature, signature
from typing import Annotated

from fastapi import File
from pydantic import BaseModel

from image_generation_protocol.io_protocol import ImageGenerationInputs


class OutputScoreRequest(BaseModel):
    inputs: ImageGenerationInputs
    frames: Annotated[bytes, File(media_type="application/x-octet-stream")]


def form_model(model_type: type[BaseModel]):
    parameters = [
        Parameter(
            parameter.alias,
            Parameter.POSITIONAL_ONLY,
            default=Signature.empty if parameter.required else parameter.default,
            annotation=parameter.outer_type_,
        )
        for name, parameter in model_type.__fields__.items()
    ]

    def as_form(**data):
        return model_type(**data)

    as_form.__signature__ = signature(as_form).replace(parameters=parameters)

    return as_form
