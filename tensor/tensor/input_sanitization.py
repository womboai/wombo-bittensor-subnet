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

from os import urandom

from tensor.protos.inputs_pb2 import GenerationRequestInputs

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.0

MIN_SIZE = 512
MAX_SIZE = 1536
MAX_STEPS = 100


# I dislike how manual this is, but it's probably fine
def sanitize_inputs(inputs: GenerationRequestInputs):
    if not inputs.width or inputs.width < MIN_SIZE or inputs.height > MAX_SIZE:
        inputs.width = DEFAULT_WIDTH

    if not inputs.height or inputs.height < MIN_SIZE or inputs.height > MAX_SIZE:
        inputs.height = DEFAULT_HEIGHT

    if not inputs.num_inference_steps or inputs.num_inference_steps > MAX_STEPS:
        inputs.num_inference_steps = DEFAULT_STEPS

    if not inputs.guidance_scale:
        inputs.guidance_scale = DEFAULT_GUIDANCE

    if not inputs.seed:
        inputs.seed = int.from_bytes(urandom(4), "little")

    return inputs
