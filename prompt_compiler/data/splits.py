from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class DatasetSplit:
    train: list[T]
    validation: list[T]
    test: list[T]


def split_dataset(items: Sequence[T]) -> DatasetSplit[T]:
    values = list(items)
    n = len(values)
    if n <= 3:
        return DatasetSplit(train=values, validation=values, test=[])
    train_end = max(1, int(n * 0.6))
    validation_end = max(train_end + 1, int(n * 0.8))
    return DatasetSplit(
        train=values[:train_end],
        validation=values[train_end:validation_end],
        test=values[validation_end:],
    )

