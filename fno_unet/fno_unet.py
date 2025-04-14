from typing import cast
import warnings

import torch
from torch import Tensor, nn
from typing_extensions import override
from fno_utils import SpectralConv2d


class FNOUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        kbase1: int = 128
        kbase2: int = 128
    ) -> None:
        super().__init__()
        self.__down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SpectralConv2d(in_channels, base_channels, ksize1 = kbase1, ksize2 = kbase2),
                    nn.ReLU(),
                    SpectralConv2d(base_channels, base_channels, ksize1 = kbase1, ksize2 = kbase2),
                    nn.ReLU(),
                )
            ]
            + [
                nn.Sequential(
                    SpectralConv2d(base_channels* 2 ** (i - 1), base_channels* 2 ** (i - 1), ksize1 = kbase1//(2**i), ksize2 = kbase2//(2**i)),
                    SpectralConv2d(base_channels * 2 ** (i - 1), base_channels * 2**i, ksize1 = kbase1//(2**i), ksize2 = kbase2//(2**i)),
                    nn.ReLU(),
                    SpectralConv2d(base_channels* 2 ** i, base_channels* 2 ** i, ksize1 = kbase1//(2**i), ksize2 = kbase2//(2**i)),,
                    nn.ReLU(),
                )
                for i in range(1, depth)
            ]
        )
        self.__central_block = nn.Sequential(
            SpectralConv2d(base_channels* 2 ** (depth-1), base_channels* 2 ** (depth - 1), ksize1 = kbase1//(2**depth), ksize2 = kbase2//(2**depth)),
            SpectralConv2d(base_channels * 2 ** (depth - 1), base_channels * 2**depth, ksize1 = kbase1//(2**depth), ksize2 = kbase2//(2**depth)),
            nn.ReLU(),
            SpectralConv2d(base_channels * 2**depth, base_channels * 2**depth, ksize1 = kbase1//(2**depth), ksize2 = kbase2//(2**depth)),
            nn.ReLU(),
            SpectralConv2d(base_channels * 2**depth, base_channels * 2 ** (depth - 1), ksize1 = kbase1//(2**(depth-1)), ksize2 = kbase2//(2**(depth-1))),
        )
        self.__up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SpectralConv2d(base_channels * 2 ** (i + 1), base_channels * 2**i, ksize1 = kbase1//(2**i), ksize2 = kbase2//(2**i)),
                    nn.ReLU(),
                    SpectralConv2d(base_channels * 2**i, base_channels * 2**i, ksize1 = kbase1//(2**i), ksize2 = kbase2//(2**i)),
                    nn.ReLU(),
                    SpectralConv2d(base_channels * 2**i, base_channels * 2 ** (i - 1), ksize1 = kbase1//(2**(i-1)), ksize2 = kbase2//(2**(i-1))),
                )
                for i in range(depth - 1, 0, -1)
            ]
            + [
                nn.Sequential(
                    SpectralConv2d(base_channels * 2, base_channels, ksize1 = kbase1, ksize2 = kbase2),
                    nn.ReLU(),
                    SpectralConv2d(base_channels, base_channels, ksize1 = kbase1, ksize2 = kbase2),
                    nn.ReLU(),
                    nn.Conv2d(base_channels, out_channels, kernel_size=1, bias = False),
                )
            ]
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {x.ndim}D input")
        if x.shape[3] < 2 ** len(self.__down_blocks):
            raise ValueError(f"Input width must be greater than or equal to {2 ** len(self.__down_blocks)}, got {x.shape[3]}")
        if x.shape[2] < 2 ** len(self.__down_blocks):
            raise ValueError(f"Input height must be greater than or equal to {2 ** len(self.__down_blocks)}, got {x.shape[2]}")
        allowed_input_channels = cast(tuple[int, ...], cast(nn.Sequential, self.__down_blocks[0])[0].weight.shape)[1]
        if x.shape[1] != allowed_input_channels:
            raise ValueError(f"Input has an invalid number of channels, expected {allowed_input_channels}, got {x.shape[1]}")
        if x.shape[3] % 2 ** len(self.__down_blocks) != 0:
            warnings.warn("Input width is not divisible by two to the power of the number of downsampling operations. The output size may not match the input size.")
        if x.shape[2] % 2 ** len(self.__down_blocks) != 0:
            warnings.warn("Input height is not divisible by two to the power of the number of downsampling operations. The output size may not match the input size.")
        tmp = []
        for down_block in self.__down_blocks:
            x = down_block(x)
            tmp.append(x)
        x = self.__central_block(x)
        for i, up_block in enumerate(self.__up_blocks):
            x = torch.cat([x, tmp[-(i + 1)]], dim=1)
            x = up_block(x)
        return x
