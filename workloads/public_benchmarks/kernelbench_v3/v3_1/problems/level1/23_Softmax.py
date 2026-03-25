import torch
import torch.nn as nn


OP_TYPE = "softmax"
SUPPORTED_PRECISIONS = ["fp16", "bf16", "fp32"]
HARDWARE_REQUIRED = ["RTX3090", "H100", "B200"]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)


batch_size = 256
dim = 16384


def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
