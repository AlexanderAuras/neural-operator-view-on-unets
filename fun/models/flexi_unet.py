import torch
from torch import nn, tensor, Tensor
import torch.nn.functional as F
from typing_extensions import override

from fun.models.unet_base import UNetBase
from fun.utils.fno_utils import SpectralConv2d_memory as SpectralConv2d

## SPECIAL LAYERS #############################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

class easyDiffs(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight= nn.Parameter(tensor([[[[0.,0.,0.],[1.,-1.,0],[0.,0.,0.]]], [[[0.,1.,0.],[0.,-1.,0],[0.,0.,0.]]]]), requires_grad = False)

    def forward(self, x):
        groups = x.shape[1]
        return torch.concat(
            [conv2d(x, self.weight[0:1].expand(groups,-1,-1,-1), groups = groups, bias=None, stride=1, padding=1),
             conv2d(x, self.weight[1:].expand(groups,-1,-1,-1), groups = groups, bias = None, stride=1, padding=1)],
            dim = 1)

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


###############################################################################################################################################
###############################################################################################################################################



## Functions to create special layers #########################################################################################################
###############################################################################################################################################
###############################################################################################################################################
def classicLayer(in_channels, out_channels, level = 0):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

def fnoLayer(in_channels, out_channels, level = 0):
    kbase1 = 32
    kbase2 = 32
    return SpectralConv2d(in_channels, out_channels, ksize1=kbase1//(2**level), ksize2=kbase2//(2**level))

def finiteDiffLayer(in_channels, out_channels, level=0):
    return nn.Sequential(easyDiffs(), nn.Conv2d(in_channels*2, out_channels, kernel_size=1))

def diffLayer(in_channels, out_channels, level = 0):
    kernel_size = 3
    padding = 1
    zero_mean = True
    scale = True
    return DiffConv2d(in_channels, out_channels, kernel_size, padding, zero_mean, scale)

###############################################################################################################################################
###############################################################################################################################################


modedict = {'classic': classicLayer, 'fno': fnoLayer, 'findiff': finiteDiffLayer, 'diff': diffLayer}

def createBlock(base_channels: int,
        updown,
        mode,
        level: int = 0,
        in_channels = None,
        out_channels = None):

    if updown == 'first':
        if in_channels is None:
            in_channels = base_channels
            print('No in_channels specified, using in_channels = base_channels')    
        return nn.Sequential(
                    modedict[mode](in_channels, base_channels, level),
                    nn.ReLU(),
                    modedict[mode](base_channels, base_channels, level),
                    nn.ReLU(),
                )
        
    elif updown == 'last':
        if out_channels is None:
            out_channels = base_channels
            print('No out_channels specified, using out_channels = base_channels')
        return nn.Sequential(
                    modedict[mode](base_channels*2, base_channels, level),
                    nn.ReLU(),
                    modedict[mode](base_channels, base_channels, level),
                    nn.ReLU(),
                    nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    elif updown == 'down':
        return nn.Sequential(
                    nn.MaxPool2d(2),
                    modedict[mode](base_channels * 2 ** (level - 1), base_channels * 2**level, level),
                    nn.ReLU(),
                    modedict[mode](base_channels * 2**level, base_channels * 2**level, level),
                    nn.ReLU(),
        )

    elif updown == 'up':
        return nn.Sequential(
                    modedict[mode](base_channels * 2 ** (level + 1), base_channels * 2**level, level),
                    nn.ReLU(),
                    modedict[mode](base_channels * 2**level, base_channels * 2**level, level),
                    nn.ReLU(),
                    nn.ConvTranspose2d(base_channels * 2**level, base_channels * 2 ** (level - 1), kernel_size=2, stride=2),

        )

    elif updown == 'central': 
        return nn.Sequential(
            nn.MaxPool2d(2),
            modedict[mode](base_channels * 2 ** (level - 1), base_channels * 2**level, level),
            nn.ReLU(),
            modedict[mode](base_channels * 2**level, base_channels * 2**level, level),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2**level, base_channels * 2 ** (level - 1), kernel_size=2, stride=2),
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
        modes = {'down': 'classic', 'central':'classic', 'up':'classic'}
    ) -> None:
        """
        Args:
            in_channels: The amount of channels of the input tensor.
            out_channels: The amount of channels of the output tensor.
            depth: The number of downsampling (and upsampling) operations.
            base_channels: The number of channels to convolve the input to in the first block.
        """
        super().__init__(in_channels, out_channels, depth, base_channels)

        self._down_blocks = nn.ModuleList(
            [
                createBlock(base_channels, updown = 'first', mode = modes['down'], in_channels = in_channels)
            ]
            + [
                createBlock(base_channels, updown = 'down', mode = modes['down'], level = i) 
                for i in range(1, depth)
            ]
        )
        self._central_block = createBlock(base_channels, updown = 'central', mode = modes['central'], level = depth
        )
        self._up_blocks = nn.ModuleList(
            [
                createBlock(base_channels, updown = 'up', mode = modes['up'], level = i)
                for i in range(depth - 1, 0, -1)
            ]
            + [
                createBlock(base_channels, updown = 'last', mode = modes['up'], out_channels = out_channels)
            ]
        )
