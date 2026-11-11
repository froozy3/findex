"""Semantic and hybrid search: passage embeddings (sentence-transformers), cosine via one matrix product, RRF fusion."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from findex.corpus import load_document_text, split_into_chunks
from findex.index import Index
from findex.scoring import BM25, Scorer
from findex.search import SearchResult, search
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

type FloatMatrix = NDArray[np.float32]

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WINDOW_CHARS = 900  # ~200 tokens: MiniLM truncates at 256
CANDIDATE_CHUNKS = 200
RANKING_DEPTH = 100
RRF_CONSTANT = 60
QUERY_OPERATORS = frozenset({"AND", "OR", "NOT"})


class SearchMode(StrEnum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        """Return L2-normalized row vectors."""
        ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        # sentence-transformers' heavily overloaded `encode` is not fully typed
        vectors = self.model.encode(  # pyright: ignore[reportUnknownMemberType]
            list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass(slots=True)
class EmbeddingStore:
    vectors: FloatMatrix  # [n_chunks, d], rows are unit vectors
    chunk_doc: NDArray[np.int32]  # chunk -> doc_id

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        np.save(directory / "chunk_doc.npy", self.chunk_doc)

    @classmethod
    def load(cls, directory: Path) -> "EmbeddingStore":
        return cls(np.load(directory / "vectors.npy", mmap_mode="r"), np.load(directory / "chunk_doc.npy"))


def build_embeddings(
    index: Index, encoder: Encoder, on_batch: Callable[[int], None] | None = None, batch_size: int = 512
) -> EmbeddingStore:
    """Chunk every document, embed the chunks, keep a chunk -> doc map."""
    texts: list[str] = []
    owners: list[int] = []
    for doc_id in sorted(index.doc_meta):
        meta = index.doc_meta[doc_id]
        for window in split_into_chunks(load_document_text(meta.path, meta.locator), WINDOW_CHARS):
            texts.append(window)
            owners.append(doc_id)

    batches: list[FloatMatrix] = []
    for start in range(0, len(texts), batch_size):
        batches.append(encoder.encode(texts[start : start + batch_size]))
        if on_batch:
            on_batch(min(batch_size, len(texts) - start))
    return EmbeddingStore(np.vstack(batches), np.array(owners, dtype=np.int32))


def semantic_ranking(
    store: EmbeddingStore, encoder: Encoder, query: str, depth: int = RANKING_DEPTH
) -> list[tuple[int, float]]:
    """Best chunk per document, as (doc_id, cosine) sorted best first."""
    similarities = store.vectors @ encoder.encode([query])[0]
    count = min(CANDIDATE_CHUNKS, len(similarities))
    top_chunks = np.argpartition(similarities, -count)[-count:]
    best_per_doc: dict[int, float] = {}
    for chunk in top_chunks[np.argsort(-similarities[top_chunks])]:
        best_per_doc.setdefault(int(store.chunk_doc[chunk]), float(similarities[chunk]))
    return list(best_per_doc.items())[:depth]


def has_operators(query: str) -> bool:
    return any(token in QUERY_OPERATORS for token in query.split()) or any(char in query for char in '()"')


def as_or_query(text: str) -> str:
    return " OR ".join(tokenize(text))


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], constant: int = RRF_CONSTANT) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def mode_search(
    index: Index,
    query: str,
    mode: SearchMode,
    k: int = 10,
    scorer: Scorer | None = None,
    store: EmbeddingStore | None = None,
    encoder: Encoder | None = None,
) -> list[SearchResult]:
    scorer = scorer or BM25()
    if mode is SearchMode.KEYWORD:
        return search(index, query, scorer, k)
    if store is None or encoder is None:
        raise RuntimeError("semantic and hybrid search need embeddings (run `findex embed`) and an encoder")

    semantic = semantic_ranking(store, encoder, query)
    if mode is SearchMode.SEMANTIC:
        pairs = semantic[:k]
    else:
        keyword_query = query if has_operators(query) else as_or_query(query)
        keyword = search(index, keyword_query, scorer, RANKING_DEPTH)
        fused = reciprocal_rank_fusion([[r.doc_id for r in keyword], [doc_id for doc_id, _ in semantic]])
        pairs = fused[:k]
    return [SearchResult(doc_id, score, index.doc_meta[doc_id].title) for doc_id, score in pairs]


def load_semantic_backend(embeddings_dir: Path, model_name: str = DEFAULT_MODEL) -> tuple[EmbeddingStore, Encoder]:
    return EmbeddingStore.load(embeddings_dir), SentenceTransformerEncoder(model_name)
