from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import Index, build_index

FIXTURE_DOCUMENTS = {
    "a.txt": "The quick brown fox jumps over the lazy dog",
    "b.txt": "The quick red fox",
    "c.txt": "Lazy dogs sleep all day",
    "d.txt": "Python is a programming language. Python, python, PYTHON!",
    "e.txt": "A quick brown dog",
}


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    for name, text in FIXTURE_DOCUMENTS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def fixture_index(corpus_dir: Path) -> Index:
    """Read-only: tests must not mutate it (session scope)."""
    return build_index(iter_documents(corpus_dir), positions=True)


@pytest.fixture
def fresh_index(corpus_dir: Path) -> Index:
    return build_index(iter_documents(corpus_dir), positions=True)
