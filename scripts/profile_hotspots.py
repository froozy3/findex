# pstats keeps its table in an untyped attribute
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportAttributeAccessIssue=false
"""Profiles and speedup table for lab 8:  uv run python scripts/profile_hotspots.py index.pkl data/gutenberg"""

import cProfile
import pstats
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

from findex.corpus import iter_documents
from findex.evaluation import load_labeled_queries
from findex.index import Index, build_index
from findex.scoring import BM25
from findex.search import search
from findex.store import load
from findex.vector import VectorIndex, vector_search

SEARCH_REPEATS = 300
TIMING_RUNS = 200


def top_functions(profiler: cProfile.Profile, count: int = 8) -> str:
    stream = pstats.Stats(profiler)
    stream.sort_stats("tottime")
    lines: list[str] = []
    for (filename, line, name), (_, calls, own, cumulative, _) in sorted(
        stream.stats.items(),  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        key=lambda item: item[1][2],
        reverse=True,
    )[:count]:
        lines.append(f"| {Path(filename).name}:{line} `{name}` | {calls} | {own:.2f} | {cumulative:.2f} |")
    return "\n".join(lines)


def profile(label: str, function: Callable[[], object]) -> None:
    profiler = cProfile.Profile()
    profiler.runcall(function)
    print(f"\n### {label}\n| function | calls | own time (s) | cumulative (s) |\n|---|---|---|---|")
    print(top_functions(profiler))


def median_ms(function: Callable[[], object]) -> float:
    samples: list[float] = []
    for _ in range(TIMING_RUNS):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def main(index_path: Path, corpus: Path) -> None:
    index: Index = load(index_path)
    profile("findex index (full corpus, no positions)", lambda: build_index(iter_documents(corpus)))
    profile(
        f"search 'white whale captain' x{SEARCH_REPEATS} (no snippets)",
        lambda: [search(index, f"white whale captain {i % 2 * 'the'}") for i in range(SEARCH_REPEATS)],
    )
    profile(
        f"search 'white whale' x{SEARCH_REPEATS} with snippets",
        lambda: [search(index, "white whale", with_snippets=True) for _ in range(SEARCH_REPEATS)],
    )

    vector_index = VectorIndex.from_index(index)
    print(
        "\n### Speedup (median of",
        TIMING_RUNS,
        "runs)\n| Query | Docs matched | Python scorer (ms) | NumPy scorer (ms) | Speedup |\n|---|---|---|---|---|",
    )
    for label, query in (("common term", "the"), ("rare term", "ahab"), ("3-term query", "white whale captain")):
        from findex.search import matching_doc_ids

        def python_engine(q: str = query) -> object:
            matching_doc_ids.cache_clear()
            return search(index, q)

        python_ms = median_ms(python_engine)
        numpy_ms = median_ms(lambda q=query: vector_search(vector_index, q))
        matched = len(matching_doc_ids(index, query, "merge"))
        print(f"| {label} (`{query}`) | {matched} | {python_ms:.2f} | {numpy_ms:.2f} | {python_ms / numpy_ms:.1f}x |")

    mismatches = 0
    for labeled in load_labeled_queries(Path("eval/queries.json")):
        expected = [r.doc_id for r in search(index, labeled.query, BM25(), k=10)]
        actual = [r.doc_id for r in vector_search(vector_index, labeled.query, "bm25", k=10)]
        mismatches += expected != actual
    print(f"\nrankings differing from the Python engine on the 10 evaluation queries: {mismatches}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
