from torch import nn

from fun.models.unet_base import UNetBase


class UNet(UNetBase):
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
                    nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            ]
            + [
                nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(base_channels * 2 ** (i - 1), base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(base_channels * 2**i, base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                for i in range(1, depth)
            ]
        )
        self._central_block = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2 ** (depth - 1), base_channels * 2**depth, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2**depth, base_channels * 2**depth, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2**depth, base_channels * 2 ** (depth - 1), kernel_size=2, stride=2),
        )
        self._up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(base_channels * 2 ** (i + 1), base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(base_channels * 2**i, base_channels * 2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(base_channels * 2**i, base_channels * 2 ** (i - 1), kernel_size=2, stride=2),
                )
                for i in range(depth - 1, 0, -1)
            ]
            + [
                nn.Sequential(
                    nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(base_channels, out_channels, kernel_size=1),
                )
            ]
        )
