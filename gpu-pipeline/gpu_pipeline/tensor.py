import torch
from safetensors.torch import save, load


__TENSOR_KEY = "tensor"


def save_tensor(tensor: torch.Tensor) -> bytes:
    return save({__TENSOR_KEY: tensor})


def load_tensor(data: bytes) -> torch.Tensor:
    return load(data)[__TENSOR_KEY]
