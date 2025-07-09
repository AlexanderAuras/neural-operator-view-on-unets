from typing import cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import override


## Adapted from: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/differential_conv.py
class DiffConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        zero_mean: bool = False,
        scale: bool = True,
        scale_factor: float = 256.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.zero_mean = zero_mean
        self.padding = padding
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, padding=padding)
        self.weight = conv.weight
        if bias:
            self.bias = conv.bias
        else:
            self.bias = None
        self.scale = scale
        self.scale_factor = scale_factor

    @override
    def forward(self, x: Tensor) -> Tensor:
        conv = F.conv2d(x, self.weight, padding=self.padding)
        kernel_sum = torch.sum(self.weight, dim=(-2, -1), keepdim=True)
        conv_sum = F.conv2d(x, kernel_sum)
        if self.scale:
            grid_width = self.scale_factor / x.shape[-1]
        else:
            grid_width = 1.0
        if self.zero_mean:
            return (conv - conv_sum) / grid_width + cast(Tensor, self.bias).unsqueeze(-1).unsqueeze(-1)  # K(N)x + b
        else:
            return conv / grid_width + (1 - 1 / grid_width) * conv_sum + cast(Tensor, self.bias).unsqueeze(-1).unsqueeze(-1)  # K(N)x + cx + b


class EasyDiffs(nn.Module):
    def __init__(self, scale: bool = True, scale_factor: float = 1.0, zero_mean: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [[[[0.0, 0.0, 0.0], [1.0, -1.0, 0], [0.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0], [0.0, -1.0, 0], [0.0, 0.0, 0.0]]], [[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]]]
            ),
            requires_grad=False,
        )
        self.scale = scale
        self.scale_factor = scale_factor
        self.zero_mean = zero_mean

    @override
    def forward(self, x: Tensor) -> Tensor:
        groups = x.shape[1]
        if self.scale:
            grid_width = self.scale_factor / x.shape[-1]
        else:
            grid_width = self.scale_factor
        if self.zero_mean:
            output = (
                torch.concat(
                    [F.conv2d(x, self.weight[i : i + 1].expand(groups, -1, -1, -1), groups=groups, bias=None, stride=1, padding=1) for i in range(self.weight.shape[0])], dim=1
                )
                / grid_width
            )
        else:
            output = torch.concat(
                [
                    torch.concat(
                        [F.conv2d(x, self.weight[i : i + 1].expand(groups, -1, -1, -1), groups=groups, bias=None, stride=1, padding=1) for i in range(self.weight.shape[0])], dim=1
                    )
                    / grid_width,
                    x,
                ],
                dim=1,
            )
        return output
