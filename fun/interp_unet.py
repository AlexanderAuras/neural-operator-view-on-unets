import math
from typing import Literal, cast
import warnings

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import override


class InterpolatingConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_kernel_size: int,
        base_input_size: int,
        max_scale_factor: int,
        *,
        padding: Literal["same", "valid"] = "valid",
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.__base_input_size = base_input_size
        self.__max_scale_factor = max_scale_factor
        self.__pad = padding == "same"
        self.__padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, base_kernel_size * max_scale_factor, base_kernel_size * max_scale_factor), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty((out_channels,), device=device, dtype=dtype)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # pyright: ignore [reportPrivateUsage]
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    @override
    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 4, "Input must be 4D (N, C, H, W)"
        assert x.shape[2] == x.shape[3], "Input must be square (H == W)"
        assert x.shape[3] >= self.__base_input_size, f"Input size {x.shape[3]} is smaller than the valid base input size {self.__base_input_size}"
        max_input_size = self.__max_scale_factor * self.__base_input_size
        assert x.shape[3] <= max_input_size, f"Input size {x.shape[3]} is larger than the maximal supported input size {max_input_size}"
        assert x.shape[3] % self.__base_input_size == 0, f"Input size {x.shape[3]} is not an integer multiple of {self.__base_input_size}"
        scale_factor = x.shape[3] / (self.__base_input_size * self.__max_scale_factor)
        weight = F.interpolate(self.weight, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        if self.__pad:
            total_padding = weight.shape[-1] - 1
            padding_start = total_padding // 2
            padding_end = total_padding - padding_start
            x = F.pad(x, (padding_start, padding_end, padding_start, padding_end), mode=self.__padding_mode)
        return F.conv2d(x, weight, self.bias)


class InterpolatingUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        *,
        base_input_size: int,
        max_scale_factor: int,
    ) -> None:
        super().__init__()
        self.__down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    InterpolatingConv2d(in_channels, base_channels, 3, base_input_size, max_scale_factor, padding="same"),
                    nn.ReLU(),
                    InterpolatingConv2d(base_channels, base_channels, 3, base_input_size, max_scale_factor, padding="same"),
                    nn.ReLU(),
                )
            ]
            + [
                nn.Sequential(
                    nn.MaxPool2d(2),
                    InterpolatingConv2d(base_channels * 2 ** (i - 1), base_channels * 2**i, 3, base_input_size // 2**i, max_scale_factor, padding="same"),
                    nn.ReLU(),
                    InterpolatingConv2d(base_channels * 2**i, base_channels * 2**i, 3, base_input_size // 2**i, max_scale_factor, padding="same"),
                    nn.ReLU(),
                )
                for i in range(1, depth)
            ]
        )
        self.__central_block = nn.Sequential(
            nn.MaxPool2d(2),
            InterpolatingConv2d(base_channels * 2 ** (depth - 1), base_channels * 2**depth, 3, base_input_size // 2**depth, max_scale_factor, padding="same"),
            nn.ReLU(),
            InterpolatingConv2d(base_channels * 2**depth, base_channels * 2**depth, 3, base_input_size // 2**depth, max_scale_factor, padding="same"),
            nn.ReLU(),
            # InterpolatingConvTranspose2d(base_channels * 2**depth, base_channels * 2 ** (depth - 1), kernel_size=2, stride=2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_channels * 2**depth, base_channels * 2 ** (depth - 1), kernel_size=3, padding="same"),
        )
        self.__up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    InterpolatingConv2d(base_channels * 2 ** (i + 1), base_channels * 2**i, 3, base_input_size // 2**i, max_scale_factor, padding="same"),
                    nn.ReLU(),
                    InterpolatingConv2d(base_channels * 2**i, base_channels * 2**i, 3, base_input_size // 2**i, max_scale_factor, padding="same"),
                    nn.ReLU(),
                    # InterpolatingConvTranspose2d(base_channels * 2**i, base_channels * 2 ** (i - 1), kernel_size=2, stride=2),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(base_channels * 2**i, base_channels * 2 ** (i - 1), kernel_size=3, padding="same"),
                )
                for i in range(depth - 1, 0, -1)
            ]
            + [
                nn.Sequential(
                    InterpolatingConv2d(base_channels * 2, base_channels, 3, base_input_size, max_scale_factor, padding="same"),
                    nn.ReLU(),
                    InterpolatingConv2d(base_channels, base_channels, 3, base_input_size, max_scale_factor, padding="same"),
                    nn.ReLU(),
                    InterpolatingConv2d(base_channels, out_channels, 1, base_input_size, max_scale_factor, padding="same"),
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
