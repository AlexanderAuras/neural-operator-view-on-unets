from math import ceil

import torch
from torch import Tensor


def generate_noise_filter(size: int) -> Tensor:
    K = ceil(size / 2)
    matrix = 1.0 / (torch.maximum(torch.abs(torch.arange(-K, K, 1)[None].repeat(2 * K, 1)), torch.abs(torch.arange(-K, K, 1)[:, None].repeat(1, 2 * K))) * 2 + 1) ** 2
    if size % 2 == 1:
        return torch.fft.ifftshift(matrix[1:, 1:])
    return torch.fft.ifftshift(matrix)
