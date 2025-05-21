from math import sqrt

from torch import Tensor, nn
from typing_extensions import override


class DnCNN(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        depth: int = 17,
        channel_count: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super(DnCNN, self).__init__()
        layers = [
            nn.Conv2d(in_channels=image_channels, out_channels=channel_count, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(in_channels=channel_count, out_channels=channel_count, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                    nn.BatchNorm2d(channel_count, momentum=0.95),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(in_channels=channel_count, out_channels=image_channels, kernel_size=kernel_size, padding=kernel_size // 2))
        self.__layers = nn.Sequential(*layers)
        for layer in self.__layers:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data.normal_(0.0, sqrt(2.0 / (kernel_size**2 * channel_count)))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.normal_(0.0, sqrt(2.0 / (kernel_size**2 * channel_count))).clip_(min=0.025)
                nn.init.constant_(layer.bias, 0)
                if hasattr(layer, "running_var") and layer.running_var is not None:
                    nn.init.constant_(layer.running_var, 0.01)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x - self.__layers(x)
