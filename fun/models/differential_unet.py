import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing_extensions import override

from fun.models.unet_base import UNetBase


## Adapted from: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/differential_conv.py#L86
class DiffConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        zero_mean: bool = False,
        scale: bool = False,
    ) -> None:
        super().__init__()
        self.zero_mean = zero_mean
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding=1)
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
            conv_sum = torch.mean(self.weight, dim=(-2, -1), keepdim=True)
            conv_sum = F.conv2d(x, conv_sum)
            return (conv - conv_sum) / grid_width
        else:
            return self.conv(x)


class DiffUNet(UNetBase):
    """
    An implementation of a differential U-Net architecture with the option to get a classical U-Net by setting zero_mean = False, scale = False.
    This implementation includes paddings and changed transpose convolution
    parameters to ensure that the output size is the same as the input size.

     Reference for differential layers
        ----------
        .. [1] : Liu-Schiaffini, M., et al. (2024). "Neural Operators with
            Localized Integral and Differential Kernels".
            ICML 2024, https://arxiv.org/abs/2402.16845.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        zero_mean: bool = True,
        scale: bool = True,
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
                nn.Sequential(
                    DiffConv2d(in_channels, base_channels, kernel_size=3, zero_mean=zero_mean, scale=scale),
                    nn.ReLU(),
                    DiffConv2d(base_channels, base_channels, kernel_size=3, zero_mean=zero_mean, scale=scale),
                    nn.ReLU(),
                )
            ]
            + [
                nn.Sequential(
                    nn.MaxPool2d(2),
                    DiffConv2d(base_channels * 2 ** (i - 1), base_channels * 2**i, kernel_size=3, zero_mean=zero_mean, scale=scale),
                    nn.ReLU(),
                    DiffConv2d(base_channels * 2**i, base_channels * 2**i, kernel_size=3, zero_mean=zero_mean, scale=scale),
                    nn.ReLU(),
                )
                for i in range(1, depth)
            ]
        )
        self._central_block = nn.Sequential(
            nn.MaxPool2d(2),
            DiffConv2d(base_channels * 2 ** (depth - 1), base_channels * 2**depth, kernel_size=3, zero_mean=zero_mean, scale=scale),
            nn.ReLU(),
            DiffConv2d(base_channels * 2**depth, base_channels * 2**depth, kernel_size=3, zero_mean=zero_mean, scale=scale),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2**depth, base_channels * 2 ** (depth - 1), kernel_size=2, stride=2),
        )
        self._up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    DiffConv2d(base_channels * 2 ** (i + 1), base_channels * 2**i, kernel_size=3, zero_mean=zero_mean, scale=scale),
                    nn.ReLU(),
                    DiffConv2d(base_channels * 2**i, base_channels * 2**i, kernel_size=3, zero_mean=zero_mean, scale=scale),
                    nn.ReLU(),
                    nn.ConvTranspose2d(base_channels * 2**i, base_channels * 2 ** (i - 1), kernel_size=2, stride=2),
                )
                for i in range(depth - 1, 0, -1)
            ]
            + [
                nn.Sequential(
                    DiffConv2d(base_channels * 2, base_channels, kernel_size=3, zero_mean=zero_mean, scale=scale),
                    nn.ReLU(),
                    DiffConv2d(base_channels, base_channels, kernel_size=3, zero_mean=zero_mean, scale=scale),
                    nn.ReLU(),
                    nn.Conv2d(base_channels, out_channels, kernel_size=1),
                )
            ]
        )
