"""Parallel indexing: split the files into chunks, build partial indexes, merge them in order."""

import logging
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path

from findex.corpus import list_corpus_files
from findex.index import Index, PartialIndex, build_partial, merge

log = logging.getLogger(__name__)


class ExecutorKind(StrEnum):
    SERIAL = "serial"
    THREADS = "threads"
    PROCESSES = "processes"


def chunk_paths(paths: Sequence[Path], chunk_count: int) -> list[list[Path]]:
    """Contiguous groups of roughly equal total size, so merging in order keeps doc_ids stable."""
    if not paths:
        return []
    sizes = [path.stat().st_size for path in paths]
    target = sum(sizes) / max(1, min(chunk_count, len(paths)))
    chunks: list[list[Path]] = [[]]
    filled = 0
    for path, size in zip(paths, sizes, strict=True):
        if chunks[-1] and filled >= target and len(chunks) < chunk_count:
            chunks.append([])
            filled = 0
        chunks[-1].append(path)
        filled += size
    return chunks


def make_executor(kind: ExecutorKind, workers: int) -> Executor:
    if kind is ExecutorKind.THREADS:
        return ThreadPoolExecutor(max_workers=workers)
    return ProcessPoolExecutor(max_workers=workers)


def build_parallel(
    corpus: Path,
    positions: bool = False,
    kind: ExecutorKind = ExecutorKind.PROCESSES,
    workers: int | None = None,
    on_partial_done: Callable[[], None] | None = None,
) -> tuple[Index, float]:
    """Return the index and the time spent merging (seconds)."""
    paths = list_corpus_files(corpus)
    worker_count = workers or os.cpu_count() or 1
    partials: list[PartialIndex]

    if kind is ExecutorKind.SERIAL or worker_count == 1:
        partials = [build_partial(paths, positions)]
    else:
        chunks = chunk_paths(paths, worker_count * 4)
        log.info("indexing %d files in %d chunks with %d %s", len(paths), len(chunks), worker_count, kind)
        with make_executor(kind, worker_count) as executor:
            futures = [executor.submit(build_partial, chunk, positions) for chunk in chunks]
            partials = []
            for future in futures:
                partials.append(future.result())  # re-raises a worker exception with its original traceback
                if on_partial_done:
                    on_partial_done()

    merge_started = time.perf_counter()
    index = merge(partials)
    return index, time.perf_counter() - merge_started
