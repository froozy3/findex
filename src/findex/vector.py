"""NumPy-backed index: postings in CSR layout (three flat int32 arrays), vectorized BM25 / TF-IDF scoring."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from findex.index import DocMeta, Index
from findex.query import And, Node, Not, Or, Phrase, QuerySyntaxError, Term, parse
from findex.search import SearchResult

type IntArray = NDArray[np.int32]
type FloatArray = NDArray[np.float64]
ScorerChoice = Literal["bm25", "tfidf"]

BM25_K1 = 1.5
BM25_B = 0.75


@dataclass(slots=True)
class VectorIndex:
    terms: list[str]
    term_row: dict[str, int]
    offsets: NDArray[np.int64]  # postings of row r live in [offsets[r], offsets[r + 1])
    doc_ids: IntArray
    tfs: IntArray
    doc_len: IntArray
    doc_meta: list[DocMeta]

    @property
    def num_docs(self) -> int:
        return len(self.doc_len)

    @property
    def avg_doc_length(self) -> float:
        return float(self.doc_len.mean())

    def postings_of(self, term: str) -> tuple[IntArray, IntArray]:
        row = self.term_row.get(term)
        if row is None:
            empty: IntArray = np.empty(0, dtype=np.int32)
            return empty, empty
        start, end = self.offsets[row], self.offsets[row + 1]
        return self.doc_ids[start:end], self.tfs[start:end]

    @classmethod
    def from_index(cls, index: Index) -> "VectorIndex":
        terms = sorted(index.postings)
        sizes = [len(index.postings[term]) for term in terms]
        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])

        doc_ids: IntArray = np.empty(int(offsets[-1]), dtype=np.int32)
        tfs: IntArray = np.empty(int(offsets[-1]), dtype=np.int32)
        for row, term in enumerate(terms):
            postings = index.postings[term]
            start = int(offsets[row])
            doc_ids[start : start + len(postings)] = [posting.doc_id for posting in postings]
            tfs[start : start + len(postings)] = [posting.tf for posting in postings]

        doc_len: IntArray = np.array(
            [index.doc_lengths[doc_id] for doc_id in sorted(index.doc_lengths)], dtype=np.int32
        )
        doc_meta = [index.doc_meta[doc_id] for doc_id in sorted(index.doc_meta)]
        return cls(terms, {term: row for row, term in enumerate(terms)}, offsets, doc_ids, tfs, doc_len, doc_meta)

    def save(self, directory: Path) -> None:
        """One `.npy` per array so that `load(mmap=True)` can map them without reading."""
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("offsets", "doc_ids", "tfs", "doc_len"):
            np.save(directory / f"{name}.npy", getattr(self, name))
        (directory / "terms.json").write_text(json.dumps(self.terms), encoding="utf-8")
        rows = [[m.doc_id, m.path, m.title, m.locator] for m in self.doc_meta]
        (directory / "docs.json").write_text(json.dumps(rows), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, mmap: bool = True) -> "VectorIndex":
        mode = "r" if mmap else None
        terms: list[str] = json.loads((directory / "terms.json").read_text(encoding="utf-8"))
        rows: list[list[int | str]] = json.loads((directory / "docs.json").read_text(encoding="utf-8"))
        return cls(
            terms,
            {term: row for row, term in enumerate(terms)},
            np.load(directory / "offsets.npy", mmap_mode=mode),
            np.load(directory / "doc_ids.npy", mmap_mode=mode),
            np.load(directory / "tfs.npy", mmap_mode=mode),
            np.load(directory / "doc_len.npy", mmap_mode=mode),
            [DocMeta(int(r[0]), str(r[1]), str(r[2]), int(r[3])) for r in rows],
        )


def matching_ids(node: Node, vector_index: VectorIndex) -> IntArray:
    match node:
        case Term(term):
            return vector_index.postings_of(term)[0]
        case And(left, right):
            return np.intersect1d(
                matching_ids(left, vector_index), matching_ids(right, vector_index), assume_unique=True
            )
        case Or(left, right):
            return np.union1d(matching_ids(left, vector_index), matching_ids(right, vector_index))
        case Not(child):
            everything: IntArray = np.arange(vector_index.num_docs, dtype=np.int32)
            return np.setdiff1d(everything, matching_ids(child, vector_index), assume_unique=True)
        case Phrase():
            raise QuerySyntaxError("phrase queries are not supported by the vectorized engine (needs positions)")
        case _:
            raise TypeError(f"unknown query node {node!r}")


def term_contribution(term: str, vector_index: VectorIndex, scorer: ScorerChoice) -> tuple[IntArray, FloatArray]:
    ids, tfs = vector_index.postings_of(term)
    num_docs, document_frequency = vector_index.num_docs, len(ids)
    if document_frequency == 0:
        return ids, np.empty(0, dtype=np.float64)
    term_frequency = tfs.astype(np.float64)
    if scorer == "tfidf":
        return ids, (1 + np.log(term_frequency)) * np.log(num_docs / document_frequency)

    inverse_document_frequency = np.log(1 + (num_docs - document_frequency + 0.5) / (document_frequency + 0.5))
    length_ratio = vector_index.doc_len[ids] / vector_index.avg_doc_length
    saturated = term_frequency * (BM25_K1 + 1) / (term_frequency + BM25_K1 * (1 - BM25_B + BM25_B * length_ratio))
    return ids, inverse_document_frequency * saturated


def top_k(ids: IntArray, scores: FloatArray, k: int) -> IntArray:
    """Best `k` doc ids: higher score first, lower doc_id first on ties (same order as the Python engine)."""
    candidate_scores = scores[ids]
    if len(ids) > k:
        threshold = np.partition(candidate_scores, -k)[-k]
        keep = candidate_scores >= threshold
        ids, candidate_scores = ids[keep], candidate_scores[keep]
    order = np.lexsort((ids, -candidate_scores))
    return ids[order][:k]


def vector_search(
    vector_index: VectorIndex, query: str, scorer: ScorerChoice = "bm25", k: int = 10
) -> list[SearchResult]:
    node = parse(query)
    matched = matching_ids(node, vector_index)
    scores: FloatArray = np.zeros(vector_index.num_docs, dtype=np.float64)
    is_matched = np.zeros(vector_index.num_docs, dtype=np.bool_)
    is_matched[matched] = True

    for term in sorted(node.terms()):
        ids, contribution = term_contribution(term, vector_index, scorer)
        keep = is_matched[ids]
        scores[ids[keep]] += contribution[keep]

    return [
        SearchResult(int(doc_id), float(scores[doc_id]), vector_index.doc_meta[int(doc_id)].title)
        for doc_id in top_k(matched, scores, k)
    ]
