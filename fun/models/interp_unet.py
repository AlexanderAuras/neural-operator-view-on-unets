import math
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import override

from fun.models.unet_base import UNetBase


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


class InterpolatingUNet(UNetBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        use_checkpointing: bool = False,
        *,
        base_input_size: int,
        max_scale_factor: int,
    ) -> None:
        super().__init__(in_channels, out_channels, depth, base_channels, use_checkpointing)
        self._down_blocks = nn.ModuleList(
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
        self._central_block = nn.Sequential(
            nn.MaxPool2d(2),
            InterpolatingConv2d(base_channels * 2 ** (depth - 1), base_channels * 2**depth, 3, base_input_size // 2**depth, max_scale_factor, padding="same"),
            nn.ReLU(),
            InterpolatingConv2d(base_channels * 2**depth, base_channels * 2**depth, 3, base_input_size // 2**depth, max_scale_factor, padding="same"),
            nn.ReLU(),
            # InterpolatingConvTranspose2d(base_channels * 2**depth, base_channels * 2 ** (depth - 1), kernel_size=2, stride=2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_channels * 2**depth, base_channels * 2 ** (depth - 1), kernel_size=3, padding="same"),
        )
        self._up_blocks = nn.ModuleList(
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
