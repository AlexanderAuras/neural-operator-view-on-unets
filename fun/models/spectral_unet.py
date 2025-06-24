import warnings
from typing import cast

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from typing_extensions import override

from fun.models.unet_base import UNetBase
from fun.utils.fno_utils import SpectralConv2d_memory as SpectralConv2d
from fun.utils.fno_utils import TrigonometricResize_2d


class Residual_Layer(nn.Module):
    def __init__(self, linear_layer: nn.Module, activation_function: nn.Module) -> None:
        super().__init__()
        self.linear_layer = linear_layer
        self.activation_function = activation_function

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x + self.activation_function(self.linear_layer(x))

class SpectralResUNet(UNetBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        use_checkpointing: bool = False,
        kbase1: int = 256,
        kbase2: int = 256,
        powerbase: int = 2,
    ) -> None:
        super().__init__(in_channels, out_channels, depth, base_channels, use_checkpointing)
        self._depth = depth
        self._first_block = nn.Sequential(SpectralConv2d(in_channels, base_channels, ksize1=kbase1, ksize2=kbase2),
                    nn.ReLU())
        
        self._downsample_layers = nn.ModuleList(
            [ 
                TrigonometricResize_2d(shape = (kbase1 // (powerbase**i),kbase2 // (powerbase**i)), upsample = False)
                for i in range(1, depth+1)
            ]
        )
        
        self._upsample_layers = nn.ModuleList(
            [ 
                TrigonometricResize_2d(shape = (kbase1 // (powerbase**i),kbase2 // (powerbase**i)), downsample = False)
                for i in range(depth-1, 0, -1)
            ]
        )

        self._down_conv_layers = nn.ModuleList(
            [
                Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (powerbase**i), ksize2=kbase2 // (powerbase**i)), nn.ReLU()) 
                for i in range(1, depth)
            ]
        )

        self._central_conv_layer = Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (powerbase**depth), ksize2=kbase2 // (powerbase**depth)), nn.ReLU())
        
        self._up_conv_layers = nn.ModuleList(
            [
                Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1 // (powerbase**i), ksize2=kbase2 // (powerbase**i)), nn.ReLU()) 
                for i in range(depth-1, 0, -1)
            ]
        )
        self._last_block = nn.Sequential(Residual_Layer(SpectralConv2d(base_channels, base_channels, ksize1=kbase1, ksize2=kbase2), nn.ReLU()),
                                            nn.Conv2d(base_channels, out_channels, kernel_size=1)
                                        )

    def __partial_forward(self, x: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        tmp = []
        keep_shape = []

        x = x.to(next(self._down_conv_layers.parameters()).device)
        x = self._first_block(x)
        tmp.append(x)
        for i in range(self._depth-1):
            x, ks = self._downsample_layers[i](x)
            keep_shape.append(ks)
            x = self._down_conv_layers[i](x)
            tmp.append(x)
        
        x = x.to(next(self._central_conv_layer.parameters()).device)
        x, ks = self._downsample_layers[self._depth-1](x)
        keep_shape.append(ks)
        x = self._central_conv_layer(x)

        return x, tmp, keep_shape

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
        orig_shape = x.shape[-2:]

        if self._use_checkpointing:
            x, tmp, keep_shape = cast(Tensor, torch.utils.checkpoint.checkpoint(self.__partial_forward, x, use_reentrant=False))
        else:
            x, tmp, keep_shape = self.__partial_forward(x)
        dev = next(self._up_conv_layers.parameters()).device
        x = x.to(dev)
        tmp = [y.to(dev) for y in tmp]

        for i in range(self._depth-1):
            x = self._upsample_layers[i](x, keep_shape = keep_shape[-(i + 1)])
            x = x + tmp[-(i + 1)]
            x = self._up_conv_layers[i](x)
        
        x = TrigonometricResize_2d(shape = orig_shape)(x)
        x = x + tmp[0]
        x = self._last_block(x)

        return x.to(orig_device)