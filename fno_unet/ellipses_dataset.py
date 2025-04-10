import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import override


class EllipsesDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        image_count: int,
        image_size: tuple[int, int] | int,
        ellipses_per_image: int,
        *,
        binary_output: bool = True,
        min_excentricity: float = 0.975,
        max_excentricity: float = 1.0,
        ellipse_scales: tuple[float, float] = (0.01, 0.06),
        ellipse_intensities: tuple[float, float] = (0.1, 1.0),
        normalize_intensities: bool = True,
    ) -> None:
        super().__init__()
        self.__image_count = image_count
        self.__image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.__ellipses_per_image = ellipses_per_image
        self.__binary_output = binary_output
        self.__min_excentricity = min_excentricity
        self.__max_excentricity = max_excentricity
        self.__ellipse_scales = ellipse_scales
        self.__ellipse_intensities = ellipse_intensities
        self.__normalize_intensities = normalize_intensities

    def __len__(self) -> int:
        return self.__image_count

    @override
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            raise IndexError()
        if index >= self.__image_count:
            raise StopIteration()
        generator = torch.Generator()
        generator.manual_seed(index)
        sizes = torch.rand((1, 1, self.__ellipses_per_image, 2), generator=generator).sort(dim=-1, descending=True).values
        sizes = self.__ellipse_scales[0] + (self.__ellipse_scales[1] - self.__ellipse_scales[0]) * sizes
        tmp = sizes[..., 1] > (1.0 - self.__min_excentricity**2) ** 0.5 * sizes[..., 0]
        if tmp.any():
            sizes[torch.stack([torch.full_like(tmp, False), tmp], dim=-1)] = (1.0 - self.__min_excentricity**2) ** 0.5 * sizes[tmp][:, 0]
        tmp = sizes[..., 1] < (1.0 - self.__max_excentricity**2) ** 0.5 * sizes[..., 0]
        if tmp.any():
            sizes[torch.stack([torch.full_like(tmp, False), tmp], dim=-1)] = (1.0 - self.__max_excentricity**2) ** 0.5 * sizes[tmp][:, 0]
        positions = torch.rand((1, 1, self.__ellipses_per_image, 2), generator=generator)
        angles = torch.pi * torch.rand((1, 1, self.__ellipses_per_image), generator=generator)
        intensities = self.__ellipse_intensities[0] + (self.__ellipse_intensities[1] - self.__ellipse_intensities[0]) * torch.rand((1, 1, self.__ellipses_per_image), generator=generator)
        coords = torch.stack(torch.meshgrid(torch.linspace(0.0, 1.0, self.__image_size[1]), torch.linspace(0.0, 1.0, self.__image_size[0]), indexing="xy"), dim=-1)[:, :, None]
        distances = 1 / sizes[..., 0] * (torch.cos(angles) * (coords[..., 0] - positions[..., 0]) + torch.sin(angles) * (coords[..., 1] - positions[..., 1])) ** 2
        distances += 1 / sizes[..., 1] * (torch.cos(angles) * (coords[..., 1] - positions[..., 1]) - torch.sin(angles) * (coords[..., 0] - positions[..., 0])) ** 2
        tmp = distances <= 1.0
        distances[tmp] = (tmp * intensities)[tmp]
        distances[~tmp] = 0.0
        if self.__binary_output:
            groundtruth = (distances.sum(-1) >= 1.0).to(torch.get_default_dtype())
        else:
            groundtruth = distances.sum(-1)
        if self.__normalize_intensities:
            groundtruth = groundtruth / groundtruth.max()
        return {"input": groundtruth}
