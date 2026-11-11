import json
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.evaluation import LabeledQuery, load_labeled_queries, mean_precision, precision_at_k
from findex.index import Index
from findex.scoring import BM25
from findex.stats import compute_stats, limited


def test_lazy_and_eager_stats_agree(corpus_dir: Path) -> None:
    lazy = compute_stats(iter_documents(corpus_dir))
    eager = compute_stats(iter_documents(corpus_dir), eager=True)
    assert lazy == eager
    assert lazy.documents == 5
    assert lazy.top_terms[0] == ("python", 4)


def test_limit_uses_islice(corpus_dir: Path) -> None:
    assert compute_stats(limited(iter_documents(corpus_dir), 2)).documents == 2


def test_precision_at_k(fixture_index: Index) -> None:
    perfect = LabeledQuery("python", frozenset({"d"}))
    wrong = LabeledQuery("python", frozenset({"a"}))
    assert precision_at_k(fixture_index, perfect, BM25(), k=1) == 1.0
    assert precision_at_k(fixture_index, wrong, BM25(), k=1) == 0.0
    assert precision_at_k(fixture_index, LabeledQuery("zzz", frozenset({"a"})), BM25()) == 0.0
    assert mean_precision(fixture_index, [perfect, wrong], BM25()) == pytest.approx(0.1)


def test_load_labeled_queries(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"query": "a", "relevant_books": ["x", "y"]}]), encoding="utf-8")
    assert load_labeled_queries(path) == [LabeledQuery("a", frozenset({"x", "y"}))]
