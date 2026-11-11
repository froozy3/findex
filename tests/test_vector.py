from pathlib import Path

import numpy as np
import pytest

from findex.index import Index
from findex.query import QuerySyntaxError
from findex.scoring import SCORERS
from findex.search import search
from findex.vector import ScorerChoice, VectorIndex, top_k, vector_search

QUERIES = [
    "quick",
    "quick fox",
    "quick OR lazy",
    "quick NOT brown",
    "(lazy OR red) fox",
    "python",
    "the OR a",
    "nothing",
]


@pytest.fixture(scope="module")
def vector_index(fixture_index: Index) -> VectorIndex:
    return VectorIndex.from_index(fixture_index)


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
@pytest.mark.parametrize("query", QUERIES)
def test_vectorized_ranking_is_identical_to_the_python_engine(
    fixture_index: Index, vector_index: VectorIndex, query: str, scorer: ScorerChoice
) -> None:
    expected = search(fixture_index, query, SCORERS[scorer], k=10)
    actual = vector_search(vector_index, query, scorer, k=10)
    assert [r.doc_id for r in actual] == [r.doc_id for r in expected]
    assert [r.score for r in actual] == pytest.approx([r.score for r in expected])


def test_top_k_breaks_ties_by_doc_id() -> None:
    ids = np.array([5, 2, 9, 7], dtype=np.int32)
    scores = np.zeros(10)
    scores[[5, 2, 9, 7]] = [1.0, 1.0, 3.0, 1.0]
    assert top_k(ids, scores, 3).tolist() == [9, 2, 5]
    assert top_k(ids, scores, 10).tolist() == [9, 2, 5, 7]


def test_phrase_queries_are_rejected(vector_index: VectorIndex) -> None:
    with pytest.raises(QuerySyntaxError):
        vector_search(vector_index, '"quick brown"')


def test_postings_are_int32_arrays(vector_index: VectorIndex) -> None:
    ids, tfs = vector_index.postings_of("quick")
    assert ids.dtype == np.int32
    assert tfs.dtype == np.int32
    assert ids.tolist() == [0, 1, 4]
    assert vector_index.postings_of("missing")[0].size == 0


@pytest.mark.parametrize("mmap", [True, False])
def test_save_load_round_trip_with_and_without_memmap(vector_index: VectorIndex, tmp_path: Path, mmap: bool) -> None:
    vector_index.save(tmp_path / "vectors")
    loaded = VectorIndex.load(tmp_path / "vectors", mmap=mmap)
    assert [r.doc_id for r in vector_search(loaded, "quick OR lazy")] == [
        r.doc_id for r in vector_search(vector_index, "quick OR lazy")
    ]
    assert isinstance(loaded.doc_ids, np.memmap) is mmap


def test_cli_vectorize_then_search_matches_python_engine(corpus_dir: Path, tmp_path: Path) -> None:
    import json

    from typer.testing import CliRunner

    from findex.cli import app

    runner = CliRunner()
    index_path, vectors = tmp_path / "index.pkl", tmp_path / "vectors"
    assert runner.invoke(app, ["index", str(corpus_dir), "--out", str(index_path), "--workers", "1"]).exit_code == 0
    assert runner.invoke(app, ["vectorize", str(index_path), "--out", str(vectors)]).exit_code == 0

    def ids(path: Path) -> list[int]:
        output = runner.invoke(app, ["search", str(path), "quick OR lazy", "--json"]).stdout
        return [json.loads(line)["doc_id"] for line in output.splitlines()]

    assert ids(vectors) == ids(index_path)
