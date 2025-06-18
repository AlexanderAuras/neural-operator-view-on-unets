import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn, tensor
from typing_extensions import override

from fun.models.unet_base import UNetBase
from fun.utils.fno_utils import SpectralConv2d_memory as SpectralConv2d
from fun.utils.interp_utils import InterpolatingConv2d
from fun.utils.diff_utils import DiffConv2d, EasyDiffs

## SPECIAL LAYERS #############################################################################################################################
###############################################################################################################################################
##############################################################################################################################################

class Residual_Layer(nn.Module):
    def __init__(self, linear_layer: nn.Module, activation_function: nn.Module) -> None:
        super().__init__()
        self.linear_layer = linear_layer
        self.activation_function = activation_function

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x + self.activation_function(self.linear_layer(x))


class Combi_Layer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, level:int, base_kernel_size:int, base_input_size:int, max_scale_factor:int, kbase1:int, kbase2:int) -> None:
        super().__init__()
        self.diff_layer = nn.Sequential(EasyDiffs(), nn.Conv2d(in_channels * 3, out_channels, kernel_size=1))
        self.fno_layer = SpectralConv2d(in_channels, out_channels, ksize1=kbase1 // (2**level), ksize2=kbase2 // (2**level))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.diff_layer(x) + self.fno_layer(x)

###############################################################################################################################################
###############################################################################################################################################


## Functions to create layers #########################################################################################################
###############################################################################################################################################
###############################################################################################################################################
def create_classic_layer(in_channels: int, out_channels: int, level: int = 0) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


def create_fno_layer(in_channels: int, out_channels: int, level: int = 0) -> SpectralConv2d:
    kbase1 = 64
    kbase2 = 64
    return SpectralConv2d(in_channels, out_channels, ksize1=kbase1 // (2**level), ksize2=kbase2 // (2**level))


def create_jump_layer(in_channels: int, out_channels: int, level: int = 0) -> DiffConv2d:
    kernel_size = 3
    padding = 1
    zero_mean = False
    scale = False
    return DiffConv2d(in_channels, out_channels, kernel_size, padding=padding, zero_mean=zero_mean, scale=scale)


def create_diff_layer(in_channels: int, out_channels: int, level: int = 0) -> DiffConv2d:
    kernel_size = 3
    padding = 1
    zero_mean = False
    scale = True
    scale_factor = 256.
    return DiffConv2d(in_channels, out_channels, kernel_size, padding=padding, zero_mean=zero_mean, scale=scale, scale_factor=scale_factor)


def create_interp_layer(in_channels: int, out_channels: int, level: int = 0) -> InterpolatingConv2d:
    base_kernel_size = 3
    base_input_size = 64
    max_scale_factor = 4
    return InterpolatingConv2d(in_channels, out_channels, base_kernel_size, base_input_size // 2 ** (level), max_scale_factor, padding="same")

def create_easydiff_layer(in_channels: int, out_channels: int, level: int = 0) -> nn.Sequential:
    scale = True
    scale_factor = 256.
    zero_mean = False
    return nn.Sequential(EasyDiffs(scale = scale, scale_factor = scale_factor, zero_mean=zero_mean), nn.Conv2d(in_channels * 4, out_channels, kernel_size=1))

def create_easyjump_layer(in_channels: int, out_channels: int, level: int = 0) -> nn.Sequential:
    scale = False
    scale_factor = 1.
    zero_mean = False
    return nn.Sequential(EasyDiffs(scale = scale, scale_factor = scale_factor, zero_mean=zero_mean), nn.Conv2d(in_channels * 4, out_channels, kernel_size=1))


def create_combi_layer(in_channels: int, out_channels: int, level: int = 0) -> nn.Sequential:
    base_kernel_size = 3
    base_input_size = 64
    max_scale_factor = 4
    kbase1 = 64
    kbase2 = 64
    return Combi_Layer(in_channels, out_channels, level, base_kernel_size, base_input_size, max_scale_factor, kbase1, kbase2)

###############################################################################################################################################
###############################################################################################################################################


def create_resblock(
    base_channels: int,
    updown: Literal["first", "last", "down", "up", "central"],
    mode: Literal["classic", "fno", "jump", "diff", "interp", "combi"],
    level: int = 0,
    in_channels: int | None = None,
    out_channels: int | None = None,
) -> nn.Sequential:
    modedict = {"classic": create_classic_layer, "fno": create_fno_layer, "jump": create_jump_layer, "diff": create_diff_layer, "interp": create_interp_layer, "combi": create_combi_layer}
    if updown == "first":
        if in_channels is None:
            in_channels = base_channels
            logging.getLogger(__name__).warning("No in_channels specified, using in_channels = base_channels")
        return nn.Sequential(
            modedict[mode](in_channels, base_channels, level),
            nn.ReLU(),
            Residual_Layer(modedict[mode](base_channels, base_channels, level), nn.ReLU()),
        )
    elif updown == "last":
        if out_channels is None:
            out_channels = base_channels
            logging.getLogger(__name__).warning("No out_channels specified, using out_channels = base_channels")
        return nn.Sequential(
            Residual_Layer(modedict[mode](base_channels, base_channels, level), nn.ReLU()),
            Residual_Layer(modedict[mode](base_channels, base_channels, level), nn.ReLU()),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )
    elif updown == "down":
        return nn.Sequential(
            nn.MaxPool2d(2),
            Residual_Layer(modedict[mode](base_channels, base_channels, level), nn.ReLU()),
            Residual_Layer(modedict[mode](base_channels, base_channels, level), nn.ReLU()),
        )
    elif updown == "up":
        return nn.Sequential(
            Residual_Layer(modedict[mode](base_channels, base_channels, level), nn.ReLU()),
            Residual_Layer(modedict[mode](base_channels, base_channels, level), nn.ReLU()),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            Residual_Layer(modedict[mode](base_channels, base_channels, level-1), nn.Identity()),
        )
    elif updown == "central":
        return nn.Sequential(
            nn.MaxPool2d(2),
            Residual_Layer(modedict[mode](base_channels, base_channels, level), nn.ReLU()),
            Residual_Layer(modedict[mode](base_channels, base_channels, level), nn.ReLU()),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            Residual_Layer(modedict[mode](base_channels, base_channels, level-1), nn.Identity())

        )

def create_block(
    base_channels: int,
    updown: Literal["first", "last", "down", "up", "central"],
    mode: Literal["classic", "fno", "jump", "diff", "interp", "combi"],
    level: int = 0,
    in_channels: int | None = None,
    out_channels: int | None = None,
) -> nn.Sequential:
    modedict = {"classic": create_classic_layer, "fno": create_fno_layer, "jump": create_jump_layer, "diff": create_diff_layer, "interp": create_interp_layer, "combi": create_combi_layer}
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
    An implementation of a flexible U-Net architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        modes: dict[Literal["down", "central", "up"], Literal["classic", "fno", "jump", "diff", "interp", "combi"]] | None = None,
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

class FlexiUNet_Res(UNetBase):
    """
    A residual implementation of the flexible U-Net architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        modes: dict[Literal["down", "central", "up"], Literal["classic", "fno", "jump", "diff", "interp", "combi"]] | None = None,
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
            [create_resblock(base_channels, updown="first", mode=modes["down"], in_channels=in_channels)]
            + [create_resblock(base_channels, updown="down", mode=modes["down"], level=i) for i in range(1, depth)]
        )
        self._central_block = create_resblock(base_channels, updown="central", mode=modes["central"], level=depth)
        self._up_blocks = nn.ModuleList(
            [create_resblock(base_channels, updown="up", mode=modes["up"], level=i) for i in range(depth - 1, 0, -1)]
            + [create_resblock(base_channels, updown="last", mode=modes["up"], out_channels=out_channels)]
        )

    def __partial_forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        tmp = []
        x = x.to(next(self._down_blocks.parameters()).device)
        for down_block in self._down_blocks:
            x = down_block(x)
            tmp.append(x)
        x = x.to(next(self._central_block.parameters()).device)
        x = self._central_block(x)
        return x, tmp

    @override
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {x.ndim}D input")
        if x.shape[3] < 2 ** len(self._down_blocks):
            raise ValueError(f"Input width must be greater than or equal to {2 ** len(self._down_blocks)}, got {x.shape[3]}")
        if x.shape[2] < 2 ** len(self._down_blocks):
            raise ValueError(f"Input height must be greater than or equal to {2 ** len(self._down_blocks)}, got {x.shape[2]}")
        #allowed_input_channels = cast(tuple[int, ...], cast(nn.Sequential, self._down_blocks[0])[0].weight.shape)[1]
        #if x.shape[1] != allowed_input_channels:
        #    raise ValueError(f"Input has an invalid number of channels, expected {allowed_input_channels}, got {x.shape[1]}")
        if x.shape[3] % 2 ** len(self._down_blocks) != 0:
            raise ValueError(
                f"Input width is not divisible by {2 ** len(self._down_blocks)}, got {x.shape[3]}" + f" ({x.shape[3]} / {2 ** len(self._down_blocks)} = {x.shape[3] / 2 ** len(self._down_blocks)})."
            )
        orig_device = x.device
        if self._use_checkpointing:
            x, tmp = cast(Tensor, torch.utils.checkpoint.checkpoint(self.__partial_forward, x, use_reentrant=False))
        else:
            x, tmp = self.__partial_forward(x)
        dev = next(self._up_blocks.parameters()).device
        x = x.to(dev)
        tmp = [y.to(dev) for y in tmp]
        for i, up_block in enumerate(self._up_blocks):
            x = x + tmp[-(i + 1)]
            x = up_block(x)
        return x.to(orig_device)