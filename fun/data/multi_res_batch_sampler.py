from collections.abc import Generator, Sequence
import random


def batched(list_: list[int], n: int = 1) -> Generator[list[int], None, None]:
    for i in range(0, len(list_), n):
        yield list_[i : i + n]


class MultiResolutionBatchSampler:
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool,
        drop_incomplete: bool,
    ) -> None:
        super().__init__()
        self.__batches: list[list[int]] = []
        curr_len = 0
        for length in lengths:
            batches = list(batched(list(range(length)), batch_size))
            if len(batches[-1]) != batch_size and drop_incomplete:
                batches = batches[:-1]
            self.__batches.extend([[y + curr_len for y in x] for x in batches])
            curr_len += length
        if shuffle:
            random.shuffle(self.__batches)

    def __iter__(self) -> Generator[list[int], None, None]:
        yield from self.__batches

    def __len__(self) -> int:
        return len(self.__batches)
