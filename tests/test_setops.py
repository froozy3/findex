from hypothesis import given
from hypothesis import strategies as st

from findex.setops import (
    merge_and,
    merge_difference,
    merge_or,
    set_and,
    set_difference,
    set_or,
)

sorted_unique_ids = st.lists(st.integers(0, 200), unique=True).map(sorted)


def test_merge_examples() -> None:
    assert merge_and([1, 3, 5, 7], [3, 4, 5, 8]) == [3, 5]
    assert merge_or([1, 3], [2, 3, 4]) == [1, 2, 3, 4]
    assert merge_difference([1, 2, 3, 4], [2, 4, 9]) == [1, 3]
    assert merge_and([], [1]) == []


@given(sorted_unique_ids, sorted_unique_ids)
def test_merge_and_matches_set_intersection(left: list[int], right: list[int]) -> None:
    assert merge_and(left, right) == set_and(left, right)


@given(sorted_unique_ids, sorted_unique_ids)
def test_merge_or_is_sorted_union(left: list[int], right: list[int]) -> None:
    result = merge_or(left, right)
    assert result == set_or(left, right)
    assert result == sorted(set(result))


@given(sorted_unique_ids, sorted_unique_ids)
def test_difference_and_intersection_partition_the_left_side(left: list[int], right: list[int]) -> None:
    assert sorted(merge_difference(left, right) + merge_and(left, right)) == left
    assert merge_difference(left, right) == set_difference(left, right)
