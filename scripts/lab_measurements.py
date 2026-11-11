"""Measurements for labs 1-2 that go into the README.

uv run python scripts/lab_measurements.py lab1 data/gutenberg
uv run python scripts/lab_measurements.py merge index.pkl
uv run python scripts/lab_measurements.py formats index.pkl
uv run python scripts/lab_measurements.py memory data/gutenberg
"""

import pickle
import subprocess
import sys
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from findex.corpus import iter_documents
from findex.setops import OPERATIONS
from findex.store import load, save
from findex.tokenize import tokenize

BENCHMARK_REPEATS = 200


def lab1(corpus: str) -> None:
    print("| Version | Documents | Peak memory | Elapsed |\n|---|---|---|---|")
    for mode in ("lazy", "eager"):
        flags = ["--eager"] if mode == "eager" else []
        output = subprocess.run(
            [sys.executable, "-m", "findex.stats", corpus, *flags], capture_output=True, text=True, check=True
        ).stdout
        documents = next(line for line in output.splitlines() if line.startswith("documents")).split()[1]
        summary = output.splitlines()[-1]
        elapsed = summary.split("elapsed: ")[1].split()[0]
        peak = summary.split("peak memory: ")[1]
        print(f"| {mode} | {documents} | {peak} | {elapsed} |")


def time_microseconds(function: Any) -> float:
    started = time.perf_counter()
    for _ in range(BENCHMARK_REPEATS):
        function()
    return (time.perf_counter() - started) / BENCHMARK_REPEATS * 1e6


def merge(index_path: Path) -> None:
    index = load(index_path)
    by_frequency = sorted(index.postings, key=lambda term: len(index.postings[term]))
    rare = [term for term in by_frequency if len(index.postings[term]) >= 5][:2]
    common = by_frequency[-2:]
    print("| Pair | Postings sizes | merge AND (µs) | set AND (µs) | merge OR (µs) | set OR (µs) | AND winner |")
    print("|---|---|---|---|---|---|---|")
    for label, (first, second) in (
        ("two most common", common),
        ("two rarest (df>=5)", rare),
        ("common + rare", (common[0], rare[0])),
    ):
        left = [p.doc_id for p in index.postings[first]]
        right = [p.doc_id for p in index.postings[second]]
        (merge_and, merge_or, _), (set_and, set_or, _) = OPERATIONS["merge"], OPERATIONS["set"]
        timings = [time_microseconds(partial(f, left, right)) for f in (merge_and, set_and, merge_or, set_or)]
        winner = "merge" if timings[0] < timings[1] else "set"
        print(
            f"| {label}: `{first}`, `{second}` | {len(left)} / {len(right)} | "
            + " | ".join(f"{t:.0f}" for t in timings)
            + f" | {winner} |"
        )


def formats(index_path: Path) -> None:
    index = load(index_path)
    print("| Format | File size (MB) | Save (s) | Load (s) |\n|---|---|---|---|")
    for suffix in (".pkl", ".json"):
        target = Path(f"/tmp/findex_format_test{suffix}")
        started = time.perf_counter()
        save(index, target)
        save_seconds = time.perf_counter() - started
        started = time.perf_counter()
        load(target)
        load_seconds = time.perf_counter() - started
        print(f"| {suffix[1:]} | {target.stat().st_size / 1e6:.0f} | {save_seconds:.1f} | {load_seconds:.1f} |")
        target.unlink()


@dataclass
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(slots=True)
class SlottedPosting:
    doc_id: int
    tf: int


def build_objects(corpus: Path, posting_class: type[PlainPosting] | type[SlottedPosting]) -> dict[str, list[Any]]:
    postings: defaultdict[str, list[Any]] = defaultdict(list)
    for doc_id, document in enumerate(iter_documents(corpus)):
        for term, tf in Counter(tokenize(document.text)).items():
            postings[term].append(posting_class(doc_id, tf))
    return dict(postings)


def build_arrays(corpus: Path) -> dict[str, tuple[array[int], array[int]]]:
    postings: defaultdict[str, tuple[array[int], array[int]]] = defaultdict(lambda: (array("I"), array("I")))
    for doc_id, document in enumerate(iter_documents(corpus)):
        for term, tf in Counter(tokenize(document.text)).items():
            doc_ids, tfs = postings[term]
            doc_ids.append(doc_id)
            tfs.append(tf)
    return dict(postings)


def memory(corpus: Path) -> None:
    print(
        "| Postings representation | Peak memory, build (MB) | Pickle size (MB) | Pickle load (s) |\n|---|---|---|---|"
    )
    builders = {
        "`list[Posting]`, plain `@dataclass`": lambda: build_objects(corpus, PlainPosting),
        "`list[Posting]`, `slots=True`": lambda: build_objects(corpus, SlottedPosting),
        "`array('I')` pairs, no objects": lambda: build_arrays(corpus),
    }
    for label, builder in builders.items():
        tracemalloc.start()
        built = builder()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        path = Path("/tmp/findex_memory_test.pkl")
        path.write_bytes(pickle.dumps(built, protocol=pickle.HIGHEST_PROTOCOL))
        del built
        started = time.perf_counter()
        pickle.loads(path.read_bytes())
        load_seconds = time.perf_counter() - started
        print(f"| {label} | {peak / 1e6:.0f} | {path.stat().st_size / 1e6:.0f} | {load_seconds:.2f} |")
        path.unlink()


if __name__ == "__main__":
    command, target = sys.argv[1], sys.argv[2]
    {
        "lab1": lambda: lab1(target),
        "merge": lambda: merge(Path(target)),
        "formats": lambda: formats(Path(target)),
        "memory": lambda: memory(Path(target)),
    }[command]()
