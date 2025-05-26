from math import sqrt
from typing import cast

import torch
from torch import Tensor, nn
import torch.utils
import torch.utils.checkpoint
from typing_extensions import override


class DnCNN(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        depth: int = 17,
        channel_count: int = 64,
        kernel_size: int = 3,
        *,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.__use_checkpointing = use_checkpointing
        self.__blocks = nn.ModuleList()
        _ = self.__blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels=image_channels, out_channels=channel_count, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True),
            )
        )
        for _ in range(depth - 2):
            _ = self.__blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=channel_count, out_channels=channel_count, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                    nn.BatchNorm2d(channel_count, momentum=0.95),
                    nn.ReLU(inplace=True),
                )
            )
        self.__blocks[-1].append(nn.Conv2d(in_channels=channel_count, out_channels=image_channels, kernel_size=kernel_size, padding=kernel_size // 2))
        for block in self.__blocks:
            for layer in cast(list[nn.Module], block):
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data.normal_(0.0, sqrt(2.0 / (kernel_size**2 * channel_count)))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.weight.data.normal_(0.0, sqrt(2.0 / (kernel_size**2 * channel_count))).clip_(min=0.025)
                    nn.init.constant_(layer.bias, 0)
                    if hasattr(layer, "running_var") and layer.running_var is not None:
                        nn.init.constant_(layer.running_var, 0.01)

    def __partial__forward(self, x: Tensor) -> Tensor:
        for block in cast(list[nn.Sequential], self.__blocks)[: len(self.__blocks) // 2]:
            x = x.to(block[0].weight.device)
            x = block(x)
        return x

    @override
    def forward(self, x: Tensor) -> Tensor:
        orig_x = x
        if self.__use_checkpointing:
            x = cast(Tensor, torch.utils.checkpoint.checkpoint(self.__partial__forward, x, use_reentrant=False))
        else:
            x = self.__partial__forward(x)
        for block in cast(list[nn.Sequential], self.__blocks)[len(self.__blocks) // 2 :]:
            x = x.to(block[0].weight.device)
            x = block(x)
        return orig_x - x
