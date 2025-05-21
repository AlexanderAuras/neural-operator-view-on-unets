from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import Any, Literal, Self, cast

import h5py
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import trange
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
        radon_device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.__image_dataset = image_dataset
        self.__target_shape = target_shape
        self.__pos_count = pos_count
        self.__angles = angles if angles is not None else torch.linspace(0.0, torch.pi, 1800)
        self.__noise_level = noise_level
        self.__noise_type = noise_type
        self.__radon_device = radon_device if radon_device is not None else torch.device("cpu")
        self.__file = None

    @classmethod
    def from_file(cls: type[Self], path: str | Path) -> Self:
        instance = __class__.__new__(cls)
        instance.__file = Path(path)
        return instance

    def __len__(self) -> int:
        if self.__file is not None:
            with h5py.File(self.__file) as hdf_file:
                return len(hdf_file) // 2
        return len(cast(Sized, self.__image_dataset))

    @override
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            raise IndexError()
        if index >= len(self):
            raise StopIteration()
        if self.__file is not None:
            with h5py.File(self.__file) as hdf_file:
                input_ = cast(Any, hdf_file[f"input-{index}"])[:]
                target = cast(Any, hdf_file[f"target-{index}"])[:]
            return {"input": torch.from_numpy(input_), "target": torch.from_numpy(target)}
        groundtruth = self.__image_dataset[index]["input"]
        measurement = cast(Tensor, Radon.apply(groundtruth[None].to(self.__radon_device), self.__pos_count, self.__angles))[0].cpu()
        target = F.interpolate(groundtruth[None], size=self.__target_shape, mode="bilinear", align_corners=False)[0]
        if self.__noise_type == "gaussian":
            measurement = measurement + self.__noise_level * torch.randn_like(measurement)
        elif self.__noise_type == "poisson":
            measurement = torch.poisson(measurement)
        else:
            raise ValueError(f"Unknown noise type: {self.__noise_type}")
        input_ = cast(Tensor, FilteredBackprojection.apply(measurement[None].to(self.__radon_device), self.__target_shape, self.__pos_count, self.__angles))[0].cpu()
        return {"input": input_ * self.__target_shape[-1], "target": target * groundtruth.shape[-1]}

    def save_to_file(self, path: str | Path, progress: bool = False) -> None:
        if self.__file is not None:
            raise RuntimeError("Dataset already loaded from file")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as hdf_file:
            if progress:
                iter_ = trange(len(cast(Sized, self.__image_dataset)))
            else:
                iter_ = range(len(cast(Sized, self.__image_dataset)))
            for index in iter_:
                item = self[index]
                hdf_file.create_dataset(f"input-{index}", data=item["input"].numpy())
                hdf_file.create_dataset(f"target-{index}", data=item["target"].numpy())
