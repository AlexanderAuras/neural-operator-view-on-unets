from collections.abc import Sized
from typing import cast

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing_extensions import override

from fno_unet.radon_operator import FilteredBackprojection, Radon


class CTPostProcessDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        image_dataset: Dataset[dict[str, Tensor]],
        noise_level: float,
        pos_count: int = 1000,
        angles: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.__image_dataset = image_dataset
        self.__noise_level = noise_level
        self.__pos_count = pos_count
        self.__angles = angles if angles is not None else torch.linspace(0.0, torch.pi, 1800)

    def __len__(self) -> int:
        return len(cast(Sized, self.__image_dataset))

    @override
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            raise IndexError()
        if index >= len(cast(Sized, self.__image_dataset)):
            raise StopIteration()
        groundtruth = self.__image_dataset[index]["image"]
        ideal_measurement = cast(Tensor, Radon.apply(groundtruth[None], self.__pos_count, self.__angles))
        target = F.interpolate(ideal_measurement[None], size=groundtruth.shape, mode="bilinear", align_corners=False)[0]
        target = cast(Tensor, FilteredBackprojection.apply(ideal_measurement[None], groundtruth.shape, self.__pos_count, self.__angles))
        real_measurement = ideal_measurement[: ideal_measurement // 2]
        real_measurement += self.__noise_level * torch.randn_like(real_measurement)
        input_ = cast(Tensor, FilteredBackprojection.apply(real_measurement[None], groundtruth.shape, self.__pos_count, self.__angles[: ideal_measurement // 2]))
        return {"input": input_, "target": target}
