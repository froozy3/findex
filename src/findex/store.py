"""Persist an index: pickle (fast, Python-only) or JSON (portable, bigger)."""

import json
import pickle
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from findex.index import DocMeta, Index, Posting
from findex.search import matching_doc_ids
from findex.timing import timed


@timed
def save(index: Index, path: Path) -> None:
    if path.suffix == ".json":
        path.write_text(json.dumps(index_to_jsonable(index)), encoding="utf-8")
    else:
        with path.open("wb") as file:
            pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)


@timed
def load(path: Path) -> Index:
    if path.suffix == ".json":
        return index_from_jsonable(json.loads(path.read_text(encoding="utf-8")))
    with path.open("rb") as file:
        # pickle.load runs arbitrary code from the file: only ever load indexes you built yourself
        loaded: Index = pickle.load(file)
    return loaded


@contextmanager
def open_index(path: Path) -> Generator[Index]:
    index = load(path)
    try:
        yield index
    finally:
        matching_doc_ids.cache_clear()


def index_to_jsonable(index: Index) -> dict[str, object]:
    return {
        "has_positions": index.has_positions,
        "doc_lengths": {str(doc_id): length for doc_id, length in index.doc_lengths.items()},
        "doc_meta": [[m.doc_id, m.path, m.title, m.locator] for m in index.doc_meta.values()],
        "postings": {
            term: [[p.doc_id, p.tf, list(p.positions)] for p in postings] for term, postings in index.postings.items()
        },
    }


def index_from_jsonable(data: dict[str, Any]) -> Index:
    postings = {
        term: [Posting(doc_id, tf, tuple(positions)) for doc_id, tf, positions in entries]
        for term, entries in data["postings"].items()
    }
    doc_meta = {row[0]: DocMeta(row[0], row[1], row[2], row[3]) for row in data["doc_meta"]}
    doc_lengths = {int(doc_id): length for doc_id, length in data["doc_lengths"].items()}
    return Index(postings, doc_lengths, doc_meta, data["has_positions"])
