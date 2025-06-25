import math
from typing import Literal, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import override


def interp_conv2d(
    x: Tensor,
    kernel: Tensor,
    base_input_size: int,
    bias: Tensor | None = None,
    pad: bool = True,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    groups: int = 1,
) -> Tensor:
    assert x.ndim == 4, "Input must be 4D (N, C, H, W)"
    assert x.shape[2] == x.shape[3], "Input must be square (H == W)"
    assert x.shape[3] >= base_input_size, f"Input size {x.shape[3]} is smaller than the valid base input size {base_input_size}"

    interp_size = int(x.shape[3] / base_input_size * kernel.shape[-1])
    interp_kernel = F.interpolate(kernel, size=interp_size, mode="bilinear", align_corners=False)
    if pad:
        total_padding = interp_kernel.shape[-1] - 1
        padding_start = total_padding // 2
        padding_end = total_padding - padding_start
        x = F.pad(x, (padding_start, padding_end, padding_start, padding_end), padding_mode)
    return F.conv2d(x, interp_kernel, bias=bias, groups=groups)


# from interp_unet.py
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
        self.__padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = cast(
            Literal["zeros", "reflect", "replicate", "circular"], "constant" if padding_mode == "zeros" else padding_mode
        )
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
        return interp_conv2d(x, self.weight, self.__base_input_size, bias=self.bias, pad=self.__pad, padding_mode=self.__padding_mode)
