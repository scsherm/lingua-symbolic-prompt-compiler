from __future__ import annotations

from typing import Sequence, TypeVar


T = TypeVar("T")


def curriculum_subset(items: Sequence[T], epoch: int) -> list[T]:
    values = list(items)
    if not values:
        return []
    if epoch <= 2:
        return values[: min(len(values), 5)]
    if epoch <= 5:
        return values[: max(1, int(len(values) * 0.3))]
    return values

