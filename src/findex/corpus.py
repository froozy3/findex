"""Lazy corpus loading: a directory of `.txt` files or a JSON-lines file."""

import json
import logging
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from findex.tokenize import normalize_text

log = logging.getLogger(__name__)

CHUNK_TARGET_CHARS = 6000
GUTENBERG_START_MARKER = "*** START OF"
GUTENBERG_END_MARKER = "*** END OF"


class Document(NamedTuple):
    doc_id: int
    path: str
    title: str
    text: str
    locator: int  # chunk number inside a text file, or line number inside a JSONL file


def read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("bad encoding, falling back to latin-1: %s", path)
        return path.read_text(encoding="latin-1")
    except OSError as error:
        log.warning("cannot read %s: %s", path, error)
        return None


def strip_gutenberg_boilerplate(text: str) -> str:
    start = text.find(GUTENBERG_START_MARKER)
    if start != -1:
        text = text[text.index("\n", start) + 1 :]
    end = text.find(GUTENBERG_END_MARKER)
    if end != -1:
        text = text[:end]
    return text


def extract_title(raw_text: str, fallback: str) -> str:
    for line in raw_text[:3000].splitlines():
        if line.startswith("Title:"):
            return line.removeprefix("Title:").strip()
    return fallback


def split_into_chunks(text: str, target_chars: int = CHUNK_TARGET_CHARS) -> Iterator[str]:
    """Group blank-line separated paragraphs into chunks of roughly `target_chars`."""
    current: list[str] = []
    current_length = 0
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        current.append(paragraph)
        current_length += len(paragraph)
        if current_length >= target_chars:
            yield "\n\n".join(current)
            current, current_length = [], 0
    if current:
        yield "\n\n".join(current)


def iter_text_file_chunks(path: Path, target_chars: int) -> Iterator[tuple[str, int, str]]:
    """Yield `(title, chunk_number, text)` for one `.txt` file."""
    raw_text = read_text_file(path)
    if raw_text is None:
        return
    title = extract_title(raw_text, path.stem)
    body = normalize_text(strip_gutenberg_boilerplate(raw_text))
    for chunk_number, chunk in enumerate(split_into_chunks(body, target_chars)):
        yield title, chunk_number, chunk


def iter_jsonl_file(path: Path) -> Iterator[tuple[str, int, str]]:
    record_number = 0
    with path.open(encoding="utf-8", errors="replace") as lines:
        for line_number, line in enumerate(lines):
            try:
                record = json.loads(line)
                title = str(record.get("title") or record.get("url") or path.stem)
                text = str(record["text"])
            except (json.JSONDecodeError, KeyError, AttributeError):
                log.warning("skipping bad line %d in %s", line_number, path)
                continue
            yield title, record_number, text
            record_number += 1


def list_corpus_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted([*root.rglob("*.txt"), *root.rglob("*.jsonl")])


def iter_documents_from_paths(paths: Iterable[Path], target_chars: int = CHUNK_TARGET_CHARS) -> Iterator[Document]:
    doc_id = 0
    for path in paths:
        chunks = iter_jsonl_file(path) if path.suffix == ".jsonl" else iter_text_file_chunks(path, target_chars)
        for title, locator, text in chunks:
            yield Document(doc_id, str(path), title, text, locator)
            doc_id += 1


def iter_documents(root: Path, target_chars: int = CHUNK_TARGET_CHARS) -> Iterator[Document]:
    yield from iter_documents_from_paths(list_corpus_files(root), target_chars)


@lru_cache(maxsize=4)
def load_file_texts(path: str) -> tuple[str, ...]:
    file_path = Path(path)
    chunks = (
        iter_jsonl_file(file_path)
        if file_path.suffix == ".jsonl"
        else iter_text_file_chunks(file_path, CHUNK_TARGET_CHARS)
    )
    return tuple(text for _, _, text in chunks)


def load_document_text(path: str, locator: int) -> str:
    return load_file_texts(path)[locator]
