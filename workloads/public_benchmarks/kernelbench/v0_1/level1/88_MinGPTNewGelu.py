import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


batch_size = 8192
dim = 8192


def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
