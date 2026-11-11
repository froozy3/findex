"""Corpus statistics in one streaming pass:  python -m findex.stats data/ [--limit N] [--eager]"""

import argparse
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize

TOP_TERMS = 50


@dataclass(frozen=True, slots=True)
class CorpusStats:
    documents: int
    total_tokens: int
    vocabulary_size: int
    top_terms: list[tuple[str, int]]


def count_terms_lazy(documents: Iterable[Document]) -> tuple[int, Counter[str]]:
    """Generators all the way: only one document and the Counter are in memory."""
    counts: Counter[str] = Counter()
    document_count = 0
    for document in documents:
        document_count += 1
        counts.update(tokenize(document.text))
    return document_count, counts


def count_terms_eager(documents: Iterable[Document]) -> tuple[int, Counter[str]]:
    """The deliberately naive version: lists of lists of everything."""
    all_documents = list(documents)
    all_tokens = [list(tokenize(document.text)) for document in all_documents]
    counts: Counter[str] = Counter()
    for tokens in all_tokens:
        counts.update(tokens)
    return len(all_documents), counts


def compute_stats(documents: Iterable[Document], eager: bool = False) -> CorpusStats:
    document_count, counts = (count_terms_eager if eager else count_terms_lazy)(documents)
    return CorpusStats(document_count, counts.total(), len(counts), counts.most_common(TOP_TERMS))


def limited(documents: Iterator[Document], limit: int | None) -> Iterator[Document]:
    return islice(documents, limit) if limit is not None else documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    parser.add_argument("--eager", action="store_true", help="use the memory-hungry list version")
    args = parser.parse_args()

    tracemalloc.start()
    started = time.perf_counter()
    stats = compute_stats(limited(iter_documents(args.corpus), args.limit), eager=args.eager)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()

    print(f"documents: {stats.documents}\ntokens: {stats.total_tokens}\nvocabulary: {stats.vocabulary_size}")
    print("top terms:", ", ".join(f"{term}={count}" for term, count in stats.top_terms))
    print(f"mode: {'eager' if args.eager else 'lazy'}  elapsed: {elapsed:.2f}s  peak memory: {peak / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
