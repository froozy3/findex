from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from findex.index import Index
from findex.semantic import (
    EmbeddingStore,
    FloatMatrix,
    SearchMode,
    as_or_query,
    build_embeddings,
    has_operators,
    mode_search,
    reciprocal_rank_fusion,
    semantic_ranking,
)
from findex.tokenize import tokenize

DIMENSIONS = 64


class HashingEncoder:
    """Deterministic stand-in for a neural model: normalized bag of hashed words."""

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        vectors = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                vectors[row, sum(map(ord, token)) % DIMENSIONS] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


@pytest.fixture(scope="module")
def store(fixture_index: Index) -> EmbeddingStore:
    return build_embeddings(fixture_index, HashingEncoder())


def test_embeddings_are_unit_rows_with_a_chunk_to_doc_map(store: EmbeddingStore, fixture_index: Index) -> None:
    assert store.vectors.shape == (fixture_index.num_docs, DIMENSIONS)
    assert np.allclose(np.linalg.norm(store.vectors, axis=1), 1.0)
    assert store.chunk_doc.tolist() == list(range(fixture_index.num_docs))


def test_semantic_ranking_puts_the_matching_document_first(store: EmbeddingStore) -> None:
    ranking = semantic_ranking(store, HashingEncoder(), "python programming language")
    assert ranking[0][0] == 3
    assert ranking[0][1] >= ranking[-1][1]


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 9]])
    assert {doc_id for doc_id, _ in fused[:2]} == {2, 3}
    assert dict(fused)[2] == pytest.approx(1 / 62 + 1 / 62)


def test_query_helpers() -> None:
    assert has_operators("a OR b")
    assert has_operators('"a b"')
    assert not has_operators("plain words only")
    assert as_or_query("Quick, brown fox") == "quick OR brown OR fox"


@pytest.mark.parametrize("mode", list(SearchMode))
def test_every_mode_returns_ranked_titles(fixture_index: Index, store: EmbeddingStore, mode: SearchMode) -> None:
    results = mode_search(fixture_index, "quick brown fox", mode, k=3, store=store, encoder=HashingEncoder())
    assert 1 <= len(results) <= 3
    assert results[0].title in {"a", "b", "e"}


def test_hybrid_finds_documents_even_when_implicit_and_matches_nothing(
    fixture_index: Index, store: EmbeddingStore
) -> None:
    query = "python quick fox"
    assert mode_search(fixture_index, query, SearchMode.KEYWORD) == []
    assert mode_search(fixture_index, query, SearchMode.HYBRID, store=store, encoder=HashingEncoder())


def test_semantic_mode_without_embeddings_is_an_error(fixture_index: Index) -> None:
    with pytest.raises(RuntimeError, match="embeddings"):
        mode_search(fixture_index, "fox", SearchMode.SEMANTIC)


def test_store_round_trip(store: EmbeddingStore, tmp_path: Path) -> None:
    store.save(tmp_path / "emb")
    loaded = EmbeddingStore.load(tmp_path / "emb")
    assert np.array_equal(loaded.vectors, store.vectors)
    assert np.array_equal(loaded.chunk_doc, store.chunk_doc)
