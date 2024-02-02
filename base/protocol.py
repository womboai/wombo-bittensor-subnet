# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 WOMBO
import pickle
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Dict, Tuple, Union, List, Optional, Any, Callable

from numpy import ndarray
import bittensor as bt
import torch


class ImageGenerationSynapse(bt.Synapse):
    """
    A simple image generation protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling image generation request and response communication between
    the miner and the validator.

    Attributes:
    - input_parameters: A dictionary containing the image generation inputs
    - output_data: An optional tuple value which, when filled, represents the response from the miner
        which the frames tensor in the first argument and the list of images in the second.
    """

    input_parameters: Dict[str, Any]

    # Optional request output, filled by receiving axon.
    output_data: Optional[bytes] = None

    def deserialize(self) -> Tuple[torch.Tensor, List[Any]]:
        result = pickle.loads(self.output_data)

        bt.logging.info(f"Deserialized synapse {self.output_data} to {result}")

        return result
