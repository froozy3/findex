"""Set operations on sorted doc-id lists: two-pointer merges and their `set`-based twins."""

from collections.abc import Callable, Sequence
from typing import Literal

Engine = Literal["merge", "set"]
DocIds = Sequence[int]


def merge_and(left: DocIds, right: DocIds) -> list[int]:
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return result


def merge_or(left: DocIds, right: DocIds) -> list[int]:
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_difference(left: DocIds, right: DocIds) -> list[int]:
    result: list[int] = []
    j = 0
    for doc_id in left:
        while j < len(right) and right[j] < doc_id:
            j += 1
        if j == len(right) or right[j] != doc_id:
            result.append(doc_id)
    return result


def set_and(left: DocIds, right: DocIds) -> list[int]:
    return sorted(set(left) & set(right))


def set_or(left: DocIds, right: DocIds) -> list[int]:
    return sorted(set(left) | set(right))


def set_difference(left: DocIds, right: DocIds) -> list[int]:
    return sorted(set(left) - set(right))


type Operation = Callable[[DocIds, DocIds], list[int]]

OPERATIONS: dict[Engine, tuple[Operation, Operation, Operation]] = {
    "merge": (merge_and, merge_or, merge_difference),
    "set": (set_and, set_or, set_difference),
}
