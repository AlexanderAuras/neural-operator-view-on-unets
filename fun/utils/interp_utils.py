from torch import Tensor
from typing import Literal
import torch.nn.functional as F

def interp_conv2d(x: Tensor, kernel: Tensor, base_input_size: int, bias: Tensor = None, pad: bool = True, padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros", groups: int = 1) -> Tensor:
        assert x.ndim == 4, "Input must be 4D (N, C, H, W)"
        assert x.shape[2] == x.shape[3], "Input must be square (H == W)"
        assert x.shape[3] >= base_input_size, f"Input size {x.shape[3]} is smaller than the valid base input size {base_input_size}"
    
        interp_size = int(x.shape[3] / base_input_size * kernel.shape[-1])
        interp_kernel = F.interpolate(kernel, size = interp_size, mode="bilinear", align_corners=False)
        if pad:
            total_padding = interp_kernel.shape[-1] - 1
            padding_start = total_padding // 2
            padding_end = total_padding - padding_start
            x = F.pad(x, (padding_start, padding_end, padding_start, padding_end), padding_mode)
        return F.conv2d(x, interp_kernel, bias=bias, groups = groups)