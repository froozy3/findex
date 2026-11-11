"""Inverted index: build (lab 2) and the `Index` object model (lab 3)."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from findex.corpus import Document, iter_documents_from_paths
from findex.timing import timed
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    locator: int


RawPosting = tuple[int, int, tuple[int, ...]]


@dataclass(slots=True)
class PartialIndex:
    """Compact, picklable index of one chunk of the corpus; `doc_id`s are local (from 0)."""

    postings: dict[str, list[RawPosting]]
    doc_lengths: list[int]
    doc_meta: list[tuple[str, str, int]]
    has_positions: bool


class Index(Mapping[str, list[Posting]]):
    """term -> postings sorted by `doc_id`, plus the per-document data ranking needs."""

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        has_positions: bool = False,
    ) -> None:
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta
        self.has_positions = has_positions

    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self)}, docs={self.num_docs}, positions={self.has_positions})"

    # identity semantics: Mapping makes instances unhashable, but `lru_cache` needs to hash the index
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @cached_property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def all_doc_ids(self) -> list[int]:
        return sorted(self.doc_lengths)


def build_partial_from_documents(documents: Iterable[Document], positions: bool = False) -> PartialIndex:
    """One streaming pass: count a document's terms, flush them into the postings, move on."""
    postings: defaultdict[str, list[RawPosting]] = defaultdict(list)
    doc_lengths: list[int] = []
    doc_meta: list[tuple[str, str, int]] = []

    for local_id, document in enumerate(documents):
        tokens = list(tokenize(document.text))
        doc_lengths.append(len(tokens))
        doc_meta.append((document.path, document.title, document.locator))

        if positions:
            positions_by_term: defaultdict[str, list[int]] = defaultdict(list)
            for position, token in enumerate(tokens):
                positions_by_term[token].append(position)
            for term, term_positions in positions_by_term.items():
                postings[term].append((local_id, len(term_positions), tuple(term_positions)))
        else:
            for term, tf in Counter(tokens).items():
                postings[term].append((local_id, tf, ()))

    return PartialIndex(dict(postings), doc_lengths, doc_meta, positions)


def build_partial(paths: Sequence[Path], positions: bool = False) -> PartialIndex:
    """Picklable top-level worker function: index the documents of the given files."""
    return build_partial_from_documents(iter_documents_from_paths(paths), positions)


def merge(partials: Iterable[PartialIndex]) -> Index:
    """Combine partial indexes in order; each partial's local ids are shifted by the docs before it."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    has_positions = False
    offset = 0

    for partial in partials:
        has_positions = has_positions or partial.has_positions
        for local_id, length in enumerate(partial.doc_lengths):
            path, title, locator = partial.doc_meta[local_id]
            doc_lengths[offset + local_id] = length
            doc_meta[offset + local_id] = DocMeta(offset + local_id, path, title, locator)
        for term, raw_postings in partial.postings.items():
            postings[term].extend(
                Posting(offset + local_id, tf, term_positions) for local_id, tf, term_positions in raw_postings
            )
        offset += len(partial.doc_lengths)

    # sorted terms: every executor then yields a byte-identical index
    return Index(dict(sorted(postings.items())), doc_lengths, doc_meta, has_positions)


@timed
def build_index(documents: Iterable[Document], positions: bool = False) -> Index:
    return merge([build_partial_from_documents(documents, positions)])
