import pytest

from findex.index import DocMeta, Index, Posting


def test_postings_are_sorted_and_counted(fixture_index: Index) -> None:
    assert [p.doc_id for p in fixture_index["quick"]] == [0, 1, 4]
    assert fixture_index["python"][0].tf == 4


def test_positions_are_recorded(fixture_index: Index) -> None:
    the_in_first_doc = fixture_index["the"][0]
    assert the_in_first_doc.positions == (0, 6)
    assert the_in_first_doc.tf == len(the_in_first_doc.positions)


def test_doc_lengths_and_meta(fixture_index: Index) -> None:
    assert fixture_index.doc_length(0) == 9
    assert fixture_index.doc_meta[1].title == "b"
    assert fixture_index.num_docs == 5


def test_mapping_protocol(fixture_index: Index) -> None:
    assert "fox" in fixture_index
    assert "elephant" not in fixture_index
    assert len(fixture_index) == len(list(fixture_index))
    with pytest.raises(KeyError):
        fixture_index["elephant"]
    assert "terms=" in repr(fixture_index)


def test_df_and_average_length(fixture_index: Index) -> None:
    assert fixture_index.df("fox") == 2
    assert fixture_index.df("elephant") == 0
    assert fixture_index.avg_doc_length == sum(fixture_index.doc_lengths.values()) / 5


def test_posting_and_docmeta_are_frozen_hashable_slotted() -> None:
    posting = Posting(1, 2, (3, 4))
    assert {posting, Posting(1, 2, (3, 4))} == {posting}
    assert not hasattr(posting, "__dict__")
    with pytest.raises(AttributeError):
        posting.tf = 5  # type: ignore[misc]  # proving that the dataclass is frozen
    assert len({DocMeta(0, "p", "t", 0), DocMeta(0, "p", "t", 0)}) == 1


def test_index_is_hashable_by_identity(fresh_index: Index) -> None:
    assert hash(fresh_index) == id(fresh_index)
    assert fresh_index == fresh_index
