from collections.abc import Sized
from typing import cast

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
        noise_level: float,
        hr_angles: Tensor | None = None,
        hr_pos_count: int = 1000,
        mr_sino_shape: tuple[int, int] = (1440, 800),
        lr_sino_shape: tuple[int, int] = (900, 800),
    ) -> None:
        super().__init__()
        self.__image_dataset = image_dataset
        self.__noise_level = noise_level
        self.__gt_pos_count = hr_pos_count
        self.__gt_angles = hr_angles if hr_angles is not None else torch.linspace(0.0, torch.pi, 1800)
        self.__mr_sino_shape = mr_sino_shape
        self.__lr_sino_shape = lr_sino_shape

    def __len__(self) -> int:
        return len(cast(Sized, self.__image_dataset))

    @override
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            raise IndexError()
        if index >= len(cast(Sized, self.__image_dataset)):
            raise StopIteration()
        hr_groundtruth = self.__image_dataset[index]["input"]
        hr_measurement = cast(Tensor, Radon.apply(hr_groundtruth[None], self.__gt_pos_count, self.__gt_angles))
        mr_measurement = F.interpolate(hr_measurement[None], size=self.__mr_sino_shape, mode="bilinear", align_corners=False)[0]
        mr_angles = F.interpolate(self.__gt_angles[None, None], size=self.__mr_sino_shape[0], mode="linear", align_corners=False)[0, 0]
        mr_recon = cast(Tensor, FilteredBackprojection.apply(mr_measurement[None], hr_groundtruth.shape, self.__mr_sino_shape[1], mr_angles))[0]
        lr_measurement = F.interpolate(hr_measurement[None], size=self.__lr_sino_shape, mode="bilinear", align_corners=False)[0]
        lr_measurement += self.__noise_level * torch.randn_like(lr_measurement)
        lr_angles = F.interpolate(self.__gt_angles[None, None], size=self.__lr_sino_shape[0], mode="linear", align_corners=False)[0, 0]
        lr_recon = cast(Tensor, FilteredBackprojection.apply(lr_measurement[None], hr_groundtruth.shape, self.__lr_sino_shape[1], lr_angles))[0]
        return {"input": lr_recon, "target": mr_recon}
