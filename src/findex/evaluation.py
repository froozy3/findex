"""precision@k with book-level labels: a result is relevant if its chunk comes from a listed book.

python -m findex.evaluation index.pkl eval/queries.json
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from findex.index import Index
from findex.scoring import SCORERS, Scorer
from findex.search import search

PRECISION_K = 5


@dataclass(frozen=True, slots=True)
class LabeledQuery:
    query: str
    relevant_books: frozenset[str]


def load_labeled_queries(path: Path) -> list[LabeledQuery]:
    return [
        LabeledQuery(row["query"], frozenset(row["relevant_books"]))
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]


def precision_at_k(index: Index, labeled: LabeledQuery, scorer: Scorer, k: int = PRECISION_K) -> float:
    results = search(index, labeled.query, scorer, k)
    if not results:
        return 0.0
    relevant = sum(Path(index.doc_meta[result.doc_id].path).stem in labeled.relevant_books for result in results)
    return relevant / k


def mean_precision(index: Index, queries: Sequence[LabeledQuery], scorer: Scorer) -> float:
    return sum(precision_at_k(index, labeled, scorer) for labeled in queries) / len(queries)


def main() -> None:
    import argparse

    from findex.store import load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("queries", type=Path)
    args = parser.parse_args()

    index = load(args.index)
    queries = load_labeled_queries(args.queries)
    print(f"| query | {' | '.join(f'P@{PRECISION_K} {name}' for name in SCORERS)} |")
    print(f"|---|{'---|' * len(SCORERS)}")
    for labeled in queries:
        cells = " | ".join(f"{precision_at_k(index, labeled, scorer):.1f}" for scorer in SCORERS.values())
        print(f"| {labeled.query} | {cells} |")
    means = " | ".join(f"**{mean_precision(index, queries, scorer):.2f}**" for scorer in SCORERS.values())
    print(f"| mean | {means} |")


if __name__ == "__main__":
    from findex.evaluation import main as run

    run()
