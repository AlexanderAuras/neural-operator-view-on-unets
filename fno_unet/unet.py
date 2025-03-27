import torch
from torch import Tensor, nn


class _Block(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.__layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore [reportImplicitOverride]
        return self.__layers(x)


class UNet(nn.Module):
    def __init__(self, depth: int = 4, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.__depth = depth
        self.__blocks = nn.ModuleList(
            [_Block(in_channels if i == 0 else 2 ** (i + 5), 2 ** (i + 6)) for i in range(depth)]
            + [_Block(2 ** (depth + 5), 2 ** (depth + 6))]
            + [_Block(2 ** (i + 6), 2 ** (i + 5)) for i in range(depth, -1, -1)]
        )
        self.__last_conv = nn.Conv2d(64, out_channels, 1)
        self.__max_pools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in range(depth)])
        self.__up_convs = nn.ModuleList([nn.ConvTranspose2d(2 ** (i + 6), 2 ** (i + 5), 2, stride=2) for i in range(depth, 0, -1)])

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore [reportImplicitOverride]
        tmp = list[Tensor]()
        for i in range(self.__depth):
            x = self.__blocks[i](x)
            tmp.append(x)
            x = self.__max_pools[i](x)
        x = self.__blocks[self.__depth](x)
        for i in range(self.__depth):
            x = self.__up_convs[i](x)
            x = torch.concat((x, tmp[self.__depth - 1 - i]), dim=-3)
            x = self.__blocks[self.__depth + 1 + i](x)
        return self.__last_conv(x)
