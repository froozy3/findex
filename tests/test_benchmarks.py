"""Micro-benchmarks (pytest-benchmark). The generous ceilings make CI fail on gross regressions."""

from pathlib import Path

from pytest_benchmark.fixture import BenchmarkFixture

from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.search import matching_doc_ids, search
from findex.vector import VectorIndex, vector_search

CEILING_SECONDS = 0.05


def test_benchmark_build_index(benchmark: BenchmarkFixture, corpus_dir: Path) -> None:
    benchmark(lambda: build_index(iter_documents(corpus_dir), positions=True))
    assert build_index(iter_documents(corpus_dir)).num_docs == 5
    assert benchmark.stats is not None
    assert benchmark.stats["mean"] < CEILING_SECONDS


def test_benchmark_python_search(benchmark: BenchmarkFixture, fixture_index: Index) -> None:
    def uncached_search() -> object:
        matching_doc_ids.cache_clear()
        return search(fixture_index, "quick OR lazy fox")

    benchmark(uncached_search)
    assert benchmark.stats is not None
    assert benchmark.stats["mean"] < CEILING_SECONDS


def test_benchmark_vector_search(benchmark: BenchmarkFixture, fixture_index: Index) -> None:
    vector_index = VectorIndex.from_index(fixture_index)
    benchmark(lambda: vector_search(vector_index, "quick OR lazy fox"))
    assert benchmark.stats is not None
    assert benchmark.stats["mean"] < CEILING_SECONDS
