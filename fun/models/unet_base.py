from abc import ABC
from typing import Self, cast

import torch
from torch import Tensor, nn
import torch.utils.checkpoint
from typing_extensions import override


class UNetBase(nn.Module, ABC):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        base_channels: int = 64,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self._use_checkpointing = use_checkpointing
        self._down_blocks = nn.ModuleList()
        self._central_block = nn.Sequential()
        self._up_blocks = nn.ModuleList()

    def to_multi_dev(self, *args: torch.device) -> Self:
        if len(args) == 0:
            raise ValueError("At least one device must be specified")
        elif len(args) == 1:
            self.to(args[0])
        elif len(args) == 2:
            self.to(args[0])
            self._central_block.to(args[1])
            for up_block in self._up_blocks:
                up_block.to(args[1])
        elif len(args) == 3:
            self.to(args[0])
        else:
            raise ValueError("Too many devices specified")
        return self

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
        allowed_input_channels = cast(tuple[int, ...], cast(nn.Sequential, self._down_blocks[0])[0].weight.shape)[1]
        if x.shape[1] != allowed_input_channels:
            raise ValueError(f"Input has an invalid number of channels, expected {allowed_input_channels}, got {x.shape[1]}")
        if x.shape[3] % 2 ** len(self._down_blocks) != 0:
            raise ValueError(
                f"Input width is not divisible by {2 ** len(self._down_blocks)}, got {x.shape[3]}"
                + f" ({x.shape[3]} / {2 ** len(self._down_blocks)} = {x.shape[3] / 2 ** len(self._down_blocks)})."
            )
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
