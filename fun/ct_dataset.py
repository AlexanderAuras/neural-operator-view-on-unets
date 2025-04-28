from collections.abc import Sized
from typing import Literal, cast

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing_extensions import override

from fun.radon_operator import FilteredBackprojection, Radon


class CTPostProcessDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        image_dataset: Dataset[dict[str, Tensor]],
        target_shape: tuple[int, int],
        angles: Tensor | None = None,
        pos_count: int = 1000,
        noise_type: Literal["gaussian", "poisson"] = "gaussian",
        noise_level: float = 0.0,
    ) -> None:
        super().__init__()
        self.__image_dataset = image_dataset
        self.__target_shape = target_shape
        self.__pos_count = pos_count
        self.__angles = angles if angles is not None else torch.linspace(0.0, torch.pi, 1800)
        self.__noise_level = noise_level
        self.__noise_type = noise_type

    def __len__(self) -> int:
        return len(cast(Sized, self.__image_dataset))

    @override
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            raise IndexError()
        if index >= len(cast(Sized, self.__image_dataset)):
            raise StopIteration()
        groundtruth = self.__image_dataset[index]["input"]
        measurement = cast(Tensor, Radon.apply(groundtruth[None], self.__pos_count, self.__angles))[0]
        target = F.interpolate(groundtruth[None], size=self.__target_shape, mode="bilinear", align_corners=False)[0]
        if self.__noise_type == "gaussian":
            measurement = measurement + self.__noise_level * torch.randn_like(measurement)
        elif self.__noise_type == "poisson":
            measurement = torch.poisson(measurement)
        else:
            raise ValueError(f"Unknown noise type: {self.__noise_type}")
        input_ = cast(Tensor, FilteredBackprojection.apply(measurement[None], self.__target_shape, self.__pos_count, self.__angles))[0]
        return {"input": input_ * self.__target_shape[-1], "target": target * groundtruth.shape[-1]}
