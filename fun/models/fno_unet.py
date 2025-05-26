from typing import cast
import warnings

import torch
from torch import Tensor, nn
import torch.utils.checkpoint
from typing_extensions import override

from fun.models.unet_base import UNetBase
from fun.utils.fno_utils import SpectralConv2d_memory as SpectralConv2d


class Residual_Layer(nn.Module):
    def __init__(self, linear_layer: nn.Module, activation_function: nn.Module) -> None:
        super().__init__()
        self.linear_layer = linear_layer
        self.activation_function = activation_function

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x + self.activation_function(self.linear_layer(x))


class FNOUNet(UNetBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        use_checkpointing: bool = False,
        base_channels: int = 64,
        kbase1: int = 128,
        kbase2: int = 128,
    ) -> None:
        super().__init__(in_channels, out_channels, depth, base_channels, use_checkpointing)
        self._down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SpectralConv2d(in_channels, base_channels, ksize1=kbase1, ksize2=kbase2),
                    nn.ReLU(),
                    SpectralConv2d(base_channels, base_channels, ksize1=kbase1, ksize2=kbase2),
                    nn.ReLU(),
                )
            ]
            + [
                nn.Sequential(
                    SpectralConv2d(base_channels * 2 ** (i - 1), base_channels * 2 ** (i - 1), ksize1=kbase1 // (2**i), ksize2=kbase2 // (2**i)),
                    SpectralConv2d(base_channels * 2 ** (i - 1), base_channels * 2**i, ksize1=kbase1 // (2**i), ksize2=kbase2 // (2**i)),
                    nn.ReLU(),
                    SpectralConv2d(base_channels * 2**i, base_channels * 2**i, ksize1=kbase1 // (2**i), ksize2=kbase2 // (2**i)),
                    nn.ReLU(),
                )
                for i in range(1, depth)
            ]
        )
        self._central_block = nn.Sequential(
            SpectralConv2d(base_channels * 2 ** (depth - 1), base_channels * 2 ** (depth - 1), ksize1=kbase1 // (2**depth), ksize2=kbase2 // (2**depth)),
            SpectralConv2d(base_channels * 2 ** (depth - 1), base_channels * 2**depth, ksize1=kbase1 // (2**depth), ksize2=kbase2 // (2**depth)),
            nn.ReLU(),
            SpectralConv2d(base_channels * 2**depth, base_channels * 2**depth, ksize1=kbase1 // (2**depth), ksize2=kbase2 // (2**depth)),
            nn.ReLU(),
            SpectralConv2d(base_channels * 2**depth, base_channels * 2 ** (depth - 1), ksize1=kbase1 // (2 ** (depth - 1)), ksize2=kbase2 // (2 ** (depth - 1))),
        )
        self._up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SpectralConv2d(base_channels * 2 ** (i + 1), base_channels * 2**i, ksize1=kbase1 // (2**i), ksize2=kbase2 // (2**i)),
                    nn.ReLU(),
                    SpectralConv2d(base_channels * 2**i, base_channels * 2**i, ksize1=kbase1 // (2**i), ksize2=kbase2 // (2**i)),
                    nn.ReLU(),
                    SpectralConv2d(base_channels * 2**i, base_channels * 2 ** (i - 1), ksize1=kbase1 // (2 ** (i - 1)), ksize2=kbase2 // (2 ** (i - 1))),
                )
                for i in range(depth - 1, 0, -1)
            ]
            + [
                nn.Sequential(
                    SpectralConv2d(base_channels * 2, base_channels, ksize1=kbase1, ksize2=kbase2),
                    nn.ReLU(),
                    SpectralConv2d(base_channels, base_channels, ksize1=kbase1, ksize2=kbase2),
                    nn.ReLU(),
                    nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=False),
                )
            ]
        )

    def __partial_forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        tmp = []
        dev = next(filter(lambda m: hasattr(m, "weight"), cast(list[list[nn.Module]], self._down_blocks)[0])).weight.device
        x = x.to(dev)
        for down_block in self._down_blocks:
            x = down_block(x)
            tmp.append(x)
        dev = next(filter(lambda m: hasattr(m, "weight"), cast(list[nn.Module], self._central_block))).weight.device
        x = x.to(dev)
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
        allowed_input_channels = cast(tuple[int, ...], cast(nn.Sequential, self._down_blocks[0])[0].weight.shape)[
            0
        ]  # differs from CNN: CNN weight shape is out x in, FNO weight shape is in x out
        if x.shape[1] != allowed_input_channels:
            raise ValueError(f"Input has an invalid number of channels, expected {allowed_input_channels}, got {x.shape[1]}")
        if x.shape[3] % 2 ** len(self._down_blocks) != 0:
            warnings.warn("Input width is not divisible by two to the power of the number of downsampling operations. The output size may not match the input size.")
        if x.shape[2] % 2 ** len(self._down_blocks) != 0:
            warnings.warn("Input height is not divisible by two to the power of the number of downsampling operations. The output size may not match the input size.")
        orig_device = x.device
        if self._use_checkpointing:
            x, tmp = cast(Tensor, torch.utils.checkpoint.checkpoint(self.__partial_forward, x, use_reentrant=False))
        else:
            x, tmp = self.__partial_forward(x)
        dev = next(filter(lambda m: hasattr(m, "weight"), cast(list[list[nn.Module]], self._up_blocks)[0])).weight.device
        x = x.to(dev)
        for i, up_block in enumerate(self._up_blocks):
            x = torch.cat([x, tmp[-(i + 1)]], dim=1)
            x = up_block(x)
        return x.to(orig_device)


class HeatUNet(UNetBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        use_checkpointing: bool = False,
        kbase1: int = 128,
        kbase2: int = 128,
    ) -> None:
        super().__init__(in_channels, out_channels, depth, base_channels, use_checkpointing)
        self._down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SpectralConv2d(in_channels, base_channels, ksize1=kbase1, ksize2=kbase2),
                    nn.ReLU(),
                    Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1, ksize2=kbase2), nn.ReLU()),
                )
            ]
            + [
                nn.Sequential(
                    Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (2**i), ksize2=kbase2 // (2**i)), nn.ReLU()),
                    Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (2**i), ksize2=kbase2 // (2**i)), nn.ReLU()),
                    Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (2**i), ksize2=kbase2 // (2**i)), nn.ReLU()),
                )
                for i in range(1, depth)
            ]
        )
        self._central_block = nn.Sequential(
            Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (2**depth), ksize2=kbase2 // (2**depth)), nn.ReLU()),
            Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (2**depth), ksize2=kbase2 // (2**depth)), nn.ReLU()),
            Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (2**depth), ksize2=kbase2 // (2**depth)), nn.ReLU()),
            Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (2 ** (depth - 1)), ksize2=kbase2 // (2 ** (depth - 1))), nn.ReLU()),
        )
        self._up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (2**i), ksize2=kbase2 // (2**i)), nn.ReLU()),
                    Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (2**i), ksize2=kbase2 // (2**i)), nn.ReLU()),
                    Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (2 ** (i - 1)), ksize2=kbase2 // (2 ** (i - 1))), nn.ReLU()),
                )
                for i in range(depth - 1, 0, -1)
            ]
            + [
                nn.Sequential(
                    Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1, ksize2=kbase2), nn.ReLU()),
                    Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1, ksize2=kbase2), nn.ReLU()),
                    nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=False),
                )
            ]
        )

    def __partial_forward(self, x: Tensor) -> Tensor:
        dev = next(filter(lambda m: hasattr(m, "weight"), cast(list[list[nn.Module]], self._down_blocks)[0])).weight.device
        x = x.to(dev)
        for down_block in self._down_blocks:
            x = down_block(x)
        dev = next(filter(lambda m: hasattr(m, "weight"), cast(list[nn.Module], self._central_block))).weight.device
        x = x.to(dev)
        x = self._central_block(x)
        return x

    @override
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {x.ndim}D input")
        if x.shape[3] < 2 ** len(self._down_blocks):
            raise ValueError(f"Input width must be greater than or equal to {2 ** len(self._down_blocks)}, got {x.shape[3]}")
        if x.shape[2] < 2 ** len(self._down_blocks):
            raise ValueError(f"Input height must be greater than or equal to {2 ** len(self._down_blocks)}, got {x.shape[2]}")
        allowed_input_channels = cast(tuple[int, ...], cast(nn.Sequential, self._down_blocks[0])[0].weight.shape)[
            0
        ]  # differs from CNN: CNN weight shape is out x in, FNO weight shape is in x out
        if x.shape[1] != allowed_input_channels:
            raise ValueError(f"Input has an invalid number of channels, expected {allowed_input_channels}, got {x.shape[1]}")
        if x.shape[3] % 2 ** len(self._down_blocks) != 0:
            warnings.warn("Input width is not divisible by two to the power of the number of downsampling operations. The output size may not match the input size.")
        if x.shape[2] % 2 ** len(self._down_blocks) != 0:
            warnings.warn("Input height is not divisible by two to the power of the number of downsampling operations. The output size may not match the input size.")
        orig_device = x.device
        if self._use_checkpointing:
            x = cast(Tensor, torch.utils.checkpoint.checkpoint(self.__partial_forward, x, use_reentrant=False))
        else:
            x = self.__partial_forward(x)
        dev = next(filter(lambda m: hasattr(m, "weight"), cast(list[list[nn.Module]], self._up_blocks)[0])).weight.device
        x = x.to(dev)
        for i, up_block in enumerate(self._up_blocks):
            x = up_block(x)
        return x.to(orig_device)
