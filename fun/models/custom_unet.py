from typing import Any, cast

from torch import Tensor, nn
from typing_extensions import override

from fun.models.unet_base import UNetBase


class OptionalMaxPool2d(nn.Module):
    def __init__(self, max_out_size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.__max_out_size = max_out_size
        self.__pooling_layer = nn.MaxPool2d(*args, **kwargs)

    @override
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] <= self.__max_out_size:
            return x
        result = self.__pooling_layer(x)
        return result


class ToggledLayer(nn.Module):
    def __init__(self, layer: nn.Module, active: bool = True) -> None:
        super().__init__()
        self.__layer = layer
        self.__active = active

    @property
    def active(self) -> bool:
        return self.__active

    @active.setter
    def active(self, value: bool) -> None:
        self.__active = value

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.__active:
            result = self.__layer(x)
            return result
        return x


class SwitchedLayer(nn.Module):
    def __init__(self, layer1: nn.Module, layer2: nn.Module) -> None:
        super().__init__()
        self.__layer1 = layer1
        self.__layer2 = layer2
        self.__first_active = True

    @property
    def first_active(self) -> bool:
        return self.__first_active

    @first_active.setter
    def first_active(self, value: bool) -> None:
        self.__first_active = value

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.__first_active:
            result = self.__layer1(x)
            return result
        result = self.__layer2(x)
        return result


class CustomUNet(UNetBase):
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
        nonresize_convs_per_block: int = 0,
        use_checkpointing: bool = False,
        optional_pool_base_size: int | None = None,
        conv_type: type[nn.Module] = nn.Conv2d,
        conv_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            in_channels: The amount of channels of the input tensor.
            out_channels: The amount of channels of the output tensor.
            depth: The number of downsampling (and upsampling) operations.
            base_channels: The number of channels to convolve the input to in the first block.
        """
        super().__init__(in_channels, out_channels, depth, base_channels, use_checkpointing)
        conv_kwargs = conv_kwargs if conv_kwargs is not None else {}
        self.__optional_pool_base_size = optional_pool_base_size
        self._down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_type(in_channels, base_channels, kernel_size=3, padding=1, **conv_kwargs),
                    nn.ReLU(),
                    conv_type(base_channels, base_channels, kernel_size=3, padding=1, **conv_kwargs),
                    nn.ReLU(),
                    *[layer for _ in range(nonresize_convs_per_block) for layer in [conv_type(base_channels, base_channels, kernel_size=3, padding=1, **conv_kwargs), nn.ReLU()]],
                )
            ]
            + [
                nn.Sequential(
                    OptionalMaxPool2d(optional_pool_base_size // 2**i, 2) if optional_pool_base_size is not None else nn.MaxPool2d(2),
                    conv_type(base_channels * 2 ** (i - 1), base_channels * 2**i, kernel_size=3, padding=1, **conv_kwargs),
                    nn.ReLU(),
                    *[
                        layer
                        for _ in range(nonresize_convs_per_block)
                        for layer in [conv_type(base_channels * 2**i, base_channels * 2**i, kernel_size=3, padding=1, **conv_kwargs), nn.ReLU()]
                    ],
                )
                for i in range(1, depth)
            ]
        )
        self._central_block = nn.Sequential(
            OptionalMaxPool2d(optional_pool_base_size // 2**depth, 2) if optional_pool_base_size is not None else nn.MaxPool2d(2),
            conv_type(base_channels * 2 ** (depth - 1), base_channels * 2**depth, kernel_size=3, padding=1, **conv_kwargs),
            nn.ReLU(),
            *[
                layer
                for _ in range(nonresize_convs_per_block)
                for layer in [conv_type(base_channels * 2**depth, base_channels * 2**depth, kernel_size=3, padding=1, **conv_kwargs), nn.ReLU()]
            ],
            SwitchedLayer(
                nn.ConvTranspose2d(base_channels * 2**depth, base_channels * 2 ** (depth - 1), kernel_size=2, stride=2),
                nn.ConvTranspose2d(base_channels * 2**depth, base_channels * 2 ** (depth - 1), kernel_size=1, stride=1),
            ),
        )
        self._up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_type(base_channels * 2 ** (i + 1), base_channels * 2**i, kernel_size=3, padding=1, **conv_kwargs),
                    nn.ReLU(),
                    *[
                        layer
                        for _ in range(nonresize_convs_per_block)
                        for layer in [conv_type(base_channels * 2**i, base_channels * 2**i, kernel_size=3, padding=1, **conv_kwargs), nn.ReLU()]
                    ],
                    SwitchedLayer(
                        nn.ConvTranspose2d(base_channels * 2**i, base_channels * 2 ** (i - 1), kernel_size=2, stride=2),
                        nn.ConvTranspose2d(base_channels * 2**i, base_channels * 2 ** (i - 1), kernel_size=1, stride=1),
                    ),
                )
                for i in range(depth - 1, 0, -1)
            ]
            + [
                nn.Sequential(
                    conv_type(base_channels * 2, base_channels, kernel_size=3, padding=1, **conv_kwargs),
                    nn.ReLU(),
                    *[layer for _ in range(nonresize_convs_per_block) for layer in [conv_type(base_channels, base_channels, kernel_size=3, padding=1, **conv_kwargs), nn.ReLU()]],
                    conv_type(base_channels, out_channels, kernel_size=1, padding=0, **conv_kwargs),
                )
            ]
        )
        self.__upscaling_ops = [self._central_block[-1], *[cast(list[nn.Module], block)[-1] for block in cast(list[nn.Module], self._up_blocks)[:-1]]]

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.__optional_pool_base_size is not None:
            tmp = x.shape[-1]
            for i in range(len(self._down_blocks) - 1):
                self.__upscaling_ops[-i - 1].first_active = tmp > self.__optional_pool_base_size // 2 ** (i + 1)
                tmp = tmp // 2 if tmp > self.__optional_pool_base_size // 2**i else tmp
            self.__upscaling_ops[0].first_active = tmp > self.__optional_pool_base_size // 2 ** len(self._down_blocks)
        return super().forward(x)
