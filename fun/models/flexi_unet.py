import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn, tensor
from typing_extensions import override

from fun.models.unet_base import UNetBase
from fun.utils.fno_utils import SpectralConv2d_memory as SpectralConv2d
from fun.utils.fno_utils import spectral_conv2d
from fun.utils.interp_utils import interp_conv2d

## SPECIAL LAYERS #############################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

class CombiIntegral(nn.Module):
    def __init__(self, base_input_size) -> None:
        super().__init__()
        weight = torch.zeros((9,1,3,3))
        for i in range(9):
            weight[i,0,i//3,i%3] = 1
        #self.integral_weight = nn.Parameter(weight, requires_grad = False)
        self.spectral_weight = nn.Parameter(torch.fft.rfft2(weight, norm = 'forward'), requires_grad = False)
        self.diff_weight = nn.Parameter(tensor([[[[0.0, 0.0, 0.0], [1.0, -1.0, 0], [0.0, 0.0, 0.0]]],\
                                                [[[0.0, 1.0, 0.0], [0.0, -1.0, 0], [0.0, 0.0, 0.0]]],\
                                                [[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]]]), requires_grad=False)
        self.base_input_size = base_input_size
        
    @override
    def forward(self, x:Tensor) -> Tensor:
        groups = x.shape[1]
        return torch.concat(
            [
                torch.concat([spectral_conv2d(x, self.spectral_weight[i:i+1].expand(-1, groups, -1, -1), kernel_shape = (3,3), norm = 'forward', groups = groups)\
                              for i in range(self.spectral_weight.shape[0])], dim = 1),
                #torch.concat([interp_conv2d(x, self.integral_weight[i:i+1].expand(groups, -1, -1, -1), self.base_input_size, groups = groups, bias = None, pad = True, padding_mode = "constant")\
                #              for i in range(self.integral_weight.shape[0])], dim = 1),
                torch.concat([F.conv2d(x, self.diff_weight[i:i+1].expand(groups, -1, -1, -1), groups=groups, bias=None, stride=1, padding=1)\
                              for i in range(self.diff_weight.shape[0])], dim = 1 )
                
            ], dim = 1)
        

class EasyDiffs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(tensor([[[[0.0, 0.0, 0.0], [1.0, -1.0, 0], [0.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0], [0.0, -1.0, 0], [0.0, 0.0, 0.0]]]]), requires_grad=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        groups = x.shape[1]
        return torch.concat(
            [
                F.conv2d(x, self.weight[0:1].expand(groups, -1, -1, -1), groups=groups, bias=None, stride=1, padding=1),
                F.conv2d(x, self.weight[1:].expand(groups, -1, -1, -1), groups=groups, bias=None, stride=1, padding=1),
            ],
            dim=1,
        )


## Adapted from: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/differential_conv.py#L86
class DiffConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        zero_mean: bool = False,
        scale: bool = False,
    ) -> None:
        super().__init__()
        self.zero_mean = zero_mean
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding=padding)
        self.weight = self.conv.weight
        self.scale = scale

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.zero_mean:
            if self.scale:
                grid_width = 256 / x.shape[-1]
            else:
                grid_width = 1
            conv = self.conv(x)
            conv_sum = torch.sum(self.weight, dim=(-2, -1), keepdim=True)
            conv_sum = F.conv2d(x, conv_sum)
            return (conv - conv_sum) / grid_width
        else:
            return self.conv(x)


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
        return interp_conv2d(x, self.weight, self.__base_input_size, bias = self.bias, pad = self.__pad, padding_mode = self.__padding_mode)


###############################################################################################################################################
###############################################################################################################################################


## Functions to create special layers #########################################################################################################
###############################################################################################################################################
###############################################################################################################################################
def create_classic_layer(in_channels: int, out_channels: int, level: int = 0) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


def create_fno_layer(in_channels: int, out_channels: int, level: int = 0) -> SpectralConv2d:
    kbase1 = 64
    kbase2 = 64
    return SpectralConv2d(in_channels, out_channels, ksize1=kbase1 // (2**level), ksize2=kbase2 // (2**level))


def create_finite_diff_layer(in_channels: int, out_channels: int, level: int = 0) -> nn.Sequential:
    return nn.Sequential(EasyDiffs(), nn.Conv2d(in_channels * 2, out_channels, kernel_size=1))


def create_diff_layer(in_channels: int, out_channels: int, level: int = 0) -> DiffConv2d:
    kernel_size = 3
    padding = 1
    zero_mean = True
    scale = True
    return DiffConv2d(in_channels, out_channels, kernel_size, padding, zero_mean, scale)


def create_interp_layer(in_channels: int, out_channels: int, level: int = 0) -> InterpolatingConv2d:
    base_kernel_size = 3
    base_input_size = 64
    max_scale_factor = 4
    return InterpolatingConv2d(in_channels, out_channels, base_kernel_size, base_input_size // 2 ** (level), max_scale_factor, padding="same")

def create_combi_layer(in_channels: int, out_channels: int, level: int = 0) -> nn.Sequential:
    base_input_size = 64
    return nn.Sequential(CombiIntegral(base_input_size // 2 ** (level)), nn.Conv2d(in_channels * 12, out_channels, kernel_size=1))

###############################################################################################################################################
###############################################################################################################################################


def create_block(
    base_channels: int,
    updown: Literal["first", "last", "down", "up", "central"],
    mode: Literal["classic", "fno", "findiff", "diff", "interp"],
    level: int = 0,
    in_channels: int | None = None,
    out_channels: int | None = None,
) -> nn.Sequential:
    modedict = {"classic": create_classic_layer, "fno": create_fno_layer, "findiff": create_finite_diff_layer, "diff": create_diff_layer, "interp": create_interp_layer, "combi": create_combi_layer}
    if updown == "first":
        if in_channels is None:
            in_channels = base_channels
            logging.getLogger(__name__).warning("No in_channels specified, using in_channels = base_channels")
        return nn.Sequential(
            modedict[mode](in_channels, base_channels, level),
            nn.ReLU(),
            modedict[mode](base_channels, base_channels, level),
            nn.ReLU(),
        )
    elif updown == "last":
        if out_channels is None:
            out_channels = base_channels
            logging.getLogger(__name__).warning("No out_channels specified, using out_channels = base_channels")
        return nn.Sequential(
            modedict[mode](base_channels * 2, base_channels, level),
            nn.ReLU(),
            modedict[mode](base_channels, base_channels, level),
            nn.ReLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )
    elif updown == "down":
        return nn.Sequential(
            nn.MaxPool2d(2),
            modedict[mode](base_channels * 2 ** (level - 1), base_channels * 2**level, level),
            nn.ReLU(),
            modedict[mode](base_channels * 2**level, base_channels * 2**level, level),
            nn.ReLU(),
        )
    elif updown == "up":
        return nn.Sequential(
            modedict[mode](base_channels * 2 ** (level + 1), base_channels * 2**level, level),
            nn.ReLU(),
            modedict[mode](base_channels * 2**level, base_channels * 2**level, level),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            modedict[mode](base_channels * 2**level, base_channels * 2**(level-1), level-1)

        )
    elif updown == "central":
        return nn.Sequential(
            nn.MaxPool2d(2),
            modedict[mode](base_channels * 2 ** (level - 1), base_channels * 2**level, level),
            nn.ReLU(),
            modedict[mode](base_channels * 2**level, base_channels * 2**level, level),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            modedict[mode](base_channels * 2**level, base_channels * 2**(level-1), level-1)

        )


class FlexiUNet(UNetBase):
    """
    An implementation of the classic U-Net architecture.
    This implementation includes paddings and changed transpose convolution
    parameters to ensure that the output size is the same as the input size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        modes: dict[Literal["down", "central", "up"], Literal["classic", "fno", "findiff", "diff"]] | None = None,
    ) -> None:
        """
        Args:
            in_channels: The amount of channels of the input tensor.
            out_channels: The amount of channels of the output tensor.
            depth: The number of downsampling (and upsampling) operations.
            base_channels: The number of channels to convolve the input to in the first block.
        """
        super().__init__(in_channels, out_channels, depth, base_channels)
        if modes is None:
            modes = {"down": "classic", "central": "classic", "up": "classic"}
        self._down_blocks = nn.ModuleList(
            [create_block(base_channels, updown="first", mode=modes["down"], in_channels=in_channels)]
            + [create_block(base_channels, updown="down", mode=modes["down"], level=i) for i in range(1, depth)]
        )
        self._central_block = create_block(base_channels, updown="central", mode=modes["central"], level=depth)
        self._up_blocks = nn.ModuleList(
            [create_block(base_channels, updown="up", mode=modes["up"], level=i) for i in range(depth - 1, 0, -1)]
            + [create_block(base_channels, updown="last", mode=modes["up"], out_channels=out_channels)]
        )
