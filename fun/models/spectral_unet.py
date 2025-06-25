import warnings
from typing import cast, Literal
import warnings

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from typing_extensions import override

from fun.models.unet_base import UNetBase
from fun.utils.fno_utils import SpectralConv2d
from fun.utils.fno_utils import TrigonometricResize_2d, gen_from_Conv2d
from fun.models.fno_unet import Residual_Layer

class MultiDimUNet(UNetBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        u_shape: bool = True,
        use_checkpointing: bool = False
    ) -> None:
        super().__init__(in_channels, out_channels, depth, use_checkpointing)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._depth = depth
        self._u_shape = u_shape
        
        self._first_block = nn.Sequential()
        self._down_conv_layers = nn.ModuleList()
        self._central_conv_layer = nn.Module()
        self._up_conv_layers = nn.ModuleList()
        self._last_block = nn.Sequential()

        if u_shape:
            self._downsample_layers = nn.ModuleList()
            self._upsample_layers = nn.ModuleList()
        else: 
            self._downsample_layers =  None
            self._upsample_layers = None

    def __partial_forward(self, x: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        tmp = []
        keep_shape = []

        x = x.to(next(self._down_conv_layers.parameters()).device)
        x = self._first_block(x)
        tmp.append(x)
        for i in range(self._depth-1):
            if self._u_shape:
                x, ks = self._downsample_layers[i](x)
                keep_shape.append(ks)
            x = self._down_conv_layers[i](x)
            tmp.append(x)
        
        x = x.to(next(self._central_conv_layer.parameters()).device)
        if self._u_shape:
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
            if self._u_shape:
                x = self._upsample_layers[i](x, keep_shape = keep_shape[-(i + 1)])
            x = x + tmp[-(i + 1)]
            x = self._up_conv_layers[i](x)
        
        if self._u_shape:
            x = TrigonometricResize_2d(shape = orig_shape)(x)
        x = x + tmp[0]
        x = self._last_block(x)

        return x.to(orig_device)

def create_conv_layer(parametrization: Literal["spectral", "spatial"], in_channels, out_channels, ksize1, ksize2, kernel_size):
    if parametrization == 'spectral':
        return SpectralConv2d(in_channels, out_channels, ksize1=ksize1, ksize2=ksize2)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = kernel_size//2, padding_mode = 'circular')

class SpectralResUNet(MultiDimUNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        parametrization: Literal["spectral", "spatial"] = "spectral",
        use_checkpointing: bool = False,
        kbase1: int = 256,
        kbase2: int = 256,
        powerbase: int = 2,
        kernel_size: int = 3,
        u_shape: bool = True,
    ) -> None:
        super().__init__(in_channels, out_channels, depth = depth, u_shape = u_shape, use_checkpointing = use_checkpointing)
        self.base_channels = base_channels
        self.parametrization = parametrization
        self.kbase1 = kbase1
        self.kbase2 = kbase2
        self.powerbase = powerbase
        self.kernel_size = kernel_size
        
        ## Convolutional Layers
        self._first_block = nn.Sequential(
            create_conv_layer(self.parametrization, in_channels, base_channels, ksize1 = kbase1, ksize2 = kbase2, kernel_size = kernel_size), nn.ReLU())
        
        self._down_conv_layers = nn.ModuleList(
            [
                Residual_Layer(create_conv_layer(self.parametrization,base_channels, base_channels, ksize1=kbase1 // (powerbase**i), ksize2=kbase2 // (powerbase**i), kernel_size = kernel_size), nn.ReLU()) 
                for i in range(1, depth)
            ]
        )

        self._central_conv_layer = Residual_Layer(create_conv_layer(self.parametrization, base_channels, base_channels, ksize1=kbase1 // (powerbase**depth), ksize2=kbase2 // (powerbase**depth), kernel_size = kernel_size), nn.ReLU())
        
        self._up_conv_layers = nn.ModuleList(
            [
                Residual_Layer(create_conv_layer(self.parametrization, base_channels, base_channels, ksize1=kbase1 // (powerbase**i), ksize2=kbase2 // (powerbase**i), kernel_size = kernel_size), nn.ReLU()) 
                for i in range(depth-1, 0, -1)
            ]
        )
        self._last_block = nn.Sequential(Residual_Layer(create_conv_layer(self.parametrization, base_channels, base_channels, ksize1=kbase1, ksize2=kbase2, kernel_size = kernel_size), nn.ReLU()),
                                            nn.Conv2d(base_channels, out_channels, kernel_size=1)
                                        )

        ## Down-/upsample Layers
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
        
    def convert_to_spectral(self):
        if self.parametrization == "spectral":
            warnings.warn("Current parametrization is already spectral. Returning self.")
            return self
        if not self._u_shape:
            warnings.warn("Spatial architecture without dimensionality reduction! Spectral kernels can become very large!")
        new_model = SpectralResUNet(self._in_channels, self._out_channels, depth = self._depth, base_channels = self.base_channels, parametrization = 'spectral', use_checkpointing = self._use_checkpointing, kbase1 = self.kbase1, kbase2 = self.kbase2, powerbase = self.powerbase, u_shape = self._u_shape)
        
        new_model._first_block[0] = gen_from_Conv2d(self._first_block[0], ksize1 = self.kbase1, ksize2 = self.kbase2)
        
        for i in range(1, self._depth):
            new_model._down_conv_layers[i-1].linear_layer = \
            gen_from_Conv2d(self._down_conv_layers[i-1].linear_layer,\
                            ksize1 = self.kbase1 // (self.powerbase**(i*self._u_shape)),\
                            ksize2 = self.kbase2 // (self.powerbase**(i*self._u_shape)))
            
        new_model._central_conv_layer.linear_layer = \
        gen_from_Conv2d(self._central_conv_layer.linear_layer,\
                        ksize1 = self.kbase1 // (self.powerbase**(self._depth*self._u_shape)),\
                        ksize2 = self.kbase2 // (self.powerbase**(self._depth*self._u_shape)))

        for i in range(self._depth-1, 0, -1):
            new_model._up_conv_layers[-i].linear_layer = \
            gen_from_Conv2d(self._up_conv_layers[-i].linear_layer,\
                            ksize1 = self.kbase1 // (self.powerbase**(i*self._u_shape)),\
                            ksize2 = self.kbase2 // (self.powerbase**(i*self._u_shape)))

        new_model._last_block[0].linear_layer = gen_from_Conv2d(self._last_block[0].linear_layer,\
                                                               ksize1 = self.kbase1, ksize2 = self.kbase2)
        new_model._last_block[1].weight = nn.Parameter(self._last_block[1].weight.clone())
        new_model._last_block[1].bias = nn.Parameter(self._last_block[1].bias.clone())

        return new_model
        
            
        
class SpectralUNet(MultiDimUNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        parametrization: Literal["spectral", "spatial"] = "spectral",
        use_checkpointing: bool = False,
        kbase1: int = 256,
        kbase2: int = 256,
        powerbase: int = 2,
        kernel_size: int = 3,
        u_shape: bool = True,
    ) -> None:
        super().__init__(in_channels, out_channels, depth = depth, u_shape = u_shape, use_checkpointing = use_checkpointing)
        
        self.base_channels = base_channels
        self.parametrization = parametrization
        self.kbase1 = kbase1
        self.kbase2 = kbase2
        self.powerbase = powerbase
        self.kernel_size = kernel_size
        
        ## Convolutional Layers
        self._first_block = nn.Sequential(create_conv_layer(parametrization, in_channels, base_channels, ksize1=kbase1, ksize2=kbase2, kernel_size = kernel_size),
                    nn.ReLU())
        self._down_conv_layers = nn.ModuleList(
            [
                nn.Sequential(create_conv_layer(parametrization, base_channels*(2**(i-1)), base_channels*(2**i), ksize1=kbase1 // (powerbase**i), ksize2=kbase2 // (powerbase**i), kernel_size = kernel_size), nn.ReLU()) 
                for i in range(1, depth)
            ]
        )

        self._central_conv_layer = nn.Sequential(create_conv_layer(parametrization, base_channels*(2**(depth-1)), base_channels*(2**(depth-1)), ksize1=kbase1 // (powerbase**depth), ksize2=kbase2 // (powerbase**depth), kernel_size = kernel_size), nn.ReLU())
        
        self._up_conv_layers = nn.ModuleList(
            [
                nn.Sequential(create_conv_layer(parametrization, base_channels*(2**i), base_channels*(2**(i-1)), ksize1=kbase1 // (powerbase**i), ksize2=kbase2 // (powerbase**i), kernel_size = kernel_size), nn.ReLU()) 
                for i in range(depth-1, 0, -1)
            ]
        )
        self._last_block = nn.Sequential(create_conv_layer(parametrization, base_channels, base_channels, ksize1=kbase1, ksize2=kbase2, kernel_size = kernel_size),
                                         nn.ReLU(),
                                         nn.Conv2d(base_channels, out_channels, kernel_size=1)
                                        )

        ## Down-/upsample Layers
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

    def convert_to_spectral(self):
        if self.parametrization == "spectral":
            warnings.warn("Current parametrization is already spectral. Returning self.")
            return self
        if not self._u_shape:
            warnings.warn("Spatial architecture without dimensionality reduction! Spectral kernels can become very large!")
        new_model = SpectralUNet(self._in_channels, self._out_channels, depth = self._depth, base_channels = self.base_channels, parametrization = 'spectral', use_checkpointing = self._use_checkpointing, kbase1 = self.kbase1, kbase2 = self.kbase2, powerbase = self.powerbase, u_shape = self._u_shape)
        
        new_model._first_block[0] = gen_from_Conv2d(self._first_block[0], ksize1 = self.kbase1, ksize2 = self.kbase2)
        
        for i in range(1, self._depth):
            new_model._down_conv_layers[i-1][0] = \
            gen_from_Conv2d(self._down_conv_layers[i-1][0],\
                            ksize1 = self.kbase1 // (self.powerbase**(i*self._u_shape)),\
                            ksize2 = self.kbase2 // (self.powerbase**(i*self._u_shape)))
            
        new_model._central_conv_layer[0] = \
        gen_from_Conv2d(self._central_conv_layer[0],\
                        ksize1 = self.kbase1 // (self.powerbase**(self._depth*self._u_shape)),\
                        ksize2 = self.kbase2 // (self.powerbase**(self._depth*self._u_shape)))

        for i in range(self._depth-1, 0, -1):
            new_model._up_conv_layers[-i][0] = \
            gen_from_Conv2d(self._up_conv_layers[-i][0],\
                            ksize1 = self.kbase1 // (self.powerbase**(i*self._u_shape)),\
                            ksize2 = self.kbase2 // (self.powerbase**(i*self._u_shape)))

        new_model._last_block[0] = gen_from_Conv2d(self._last_block[0],\
                                                               ksize1 = self.kbase1, ksize2 = self.kbase2)
        new_model._last_block[2].weight = nn.Parameter(self._last_block[2].weight.clone())
        new_model._last_block[2].bias = nn.Parameter(self._last_block[2].bias.clone())

        return new_model

            
        
