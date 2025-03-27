from typing import Literal

import torch
from torch import Tensor, nn


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        *,
        dims: Literal[1, 2, 3] = 2,
    ) -> None:
        super().__init__()
        ConvType = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims - 1]
        TransposeConvType = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][dims - 1]
        MaxPoolType = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][dims - 1]
        self.__down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvType(in_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    ConvType(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            ]
            + [
                nn.Sequential(
                    MaxPoolType(2),
                    ConvType(base_channels * 2 ** (i - 1), base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    ConvType(base_channels * 2**i, base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                for i in range(1, depth)
            ]
        )
        self.__central_block = nn.Sequential(
            MaxPoolType(2),
            ConvType(base_channels * 2 ** (depth - 1), base_channels * 2**depth, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvType(base_channels * 2**depth, base_channels * 2**depth, kernel_size=3, padding=1),
            nn.ReLU(),
            TransposeConvType(base_channels * 2**depth, base_channels * 2 ** (depth - 1), kernel_size=2, stride=2),
        )
        self.__up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvType(base_channels * 2 ** (i + 1), base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    ConvType(base_channels * 2**i, base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    TransposeConvType(base_channels * 2**i, base_channels * 2 ** (i - 1), kernel_size=2, stride=2),
                )
                for i in range(depth - 1, 0, -1)
            ]
            + [
                nn.Sequential(
                    ConvType(base_channels * 2, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    ConvType(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    ConvType(base_channels, out_channels, kernel_size=1),
                )
            ]
        )

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore [reportImplicitOverride]
        assert x.shape[-1] % 2 ** len(self.__down_blocks) == 0, "Invalid input size"
        tmp = []
        for down_block in self.__down_blocks:
            x = down_block(x)
            tmp.append(x)
        x = self.__central_block(x)
        for i, up_block in enumerate(self.__up_blocks):
            x = torch.cat([x, tmp[-(i + 1)]], dim=1)
            x = up_block(x)
        return x
