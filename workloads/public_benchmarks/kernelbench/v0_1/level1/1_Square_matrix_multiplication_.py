import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


N = 2048 * 2


def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]


def get_init_inputs():
    return []
