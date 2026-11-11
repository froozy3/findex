import json
from pathlib import Path

import pytest

from findex import parallel
from findex.corpus import iter_documents, list_corpus_files
from findex.index import Index, build_index
from findex.parallel import ExecutorKind, build_parallel, chunk_paths
from findex.store import index_to_jsonable


def canonical_bytes(index: Index) -> bytes:
    """pickle output depends on object identity (memoization), so compare a canonical encoding instead."""
    return json.dumps(index_to_jsonable(index), sort_keys=True).encode()


def serial_index(corpus_dir: Path) -> Index:
    return build_index(iter_documents(corpus_dir), positions=True)


@pytest.mark.parametrize("kind", list(ExecutorKind))
@pytest.mark.parametrize("workers", [1, 2, 3])
def test_every_executor_builds_a_byte_identical_index(corpus_dir: Path, kind: ExecutorKind, workers: int) -> None:
    index, merge_seconds = build_parallel(corpus_dir, positions=True, kind=kind, workers=workers)
    assert canonical_bytes(index) == canonical_bytes(serial_index(corpus_dir))
    assert merge_seconds >= 0


def test_chunks_are_contiguous_and_cover_every_file(corpus_dir: Path) -> None:
    paths = list_corpus_files(corpus_dir)
    chunks = chunk_paths(paths, 3)
    assert [path for chunk in chunks for path in chunk] == paths
    assert 1 < len(chunks) <= 3
    assert chunk_paths([], 4) == []


def test_worker_exception_surfaces_instead_of_hanging(corpus_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_build_partial(*_args: object) -> None:
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(parallel, "build_partial", broken_build_partial)
    with pytest.raises(RuntimeError, match="worker exploded"):
        build_parallel(corpus_dir, kind=ExecutorKind.THREADS, workers=2)


@pytest.mark.slow
def test_process_pool_indexing_matches_serial_on_a_bigger_corpus(tmp_path: Path) -> None:
    for number in range(40):
        (tmp_path / f"{number:02d}.txt").write_text(
            f"word{number} shared text number {number}\n\n" * 200, encoding="utf-8"
        )
    parallel_index, _ = build_parallel(tmp_path, kind=ExecutorKind.PROCESSES, workers=4)
    assert canonical_bytes(parallel_index) == canonical_bytes(build_index(iter_documents(tmp_path)))
