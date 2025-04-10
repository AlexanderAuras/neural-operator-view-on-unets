import warnings

import torch
from torch import Tensor, nn
from typing_extensions import override


class FNOUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.__down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            ]
            + [
                nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(base_channels * 2 ** (i - 1), base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(base_channels * 2**i, base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                for i in range(1, depth)
            ]
        )
        self.__central_block = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2 ** (depth - 1), base_channels * 2**depth, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2**depth, base_channels * 2**depth, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2**depth, base_channels * 2 ** (depth - 1), kernel_size=2, stride=2),
        )
        self.__up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(base_channels * 2 ** (i + 1), base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(base_channels * 2**i, base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(base_channels * 2**i, base_channels * 2 ** (i - 1), kernel_size=2, stride=2),
                )
                for i in range(depth - 1, 0, -1)
            ]
            + [
                nn.Sequential(
                    nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(base_channels, out_channels, kernel_size=1),
                )
            ]
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {x.ndim}D input")
        if x.shape[-1] < 2 ** len(self.__down_blocks):
            raise ValueError(f"Input width must be greater than or equal to {2 ** len(self.__down_blocks)}, got {x.shape[-2]} input")
        if x.shape[-2] < 2 ** len(self.__down_blocks):
            raise ValueError(f"Input height must be greater than or equal to {2 ** len(self.__down_blocks)}, got {x.shape[-2]} input")
        if x.shape[-1] % 2 ** len(self.__down_blocks) != 0:
            warnings.warn("Input size is not divisible by the number of downsampling operations. The output size may not match the input size.")
        tmp = []
        for down_block in self.__down_blocks:
            x = down_block(x)
            tmp.append(x)
        x = self.__central_block(x)
        for i, up_block in enumerate(self.__up_blocks):
            x = torch.cat([x, tmp[-(i + 1)]], dim=1)
            x = up_block(x)
        return x
