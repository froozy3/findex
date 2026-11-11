"""precision@5 for keyword / semantic / hybrid on the 10 original + 5 paraphrase queries.

uv run python scripts/eval_modes.py index.pkl data/embeddings
"""

import sys
from pathlib import Path

from findex.evaluation import PRECISION_K, LabeledQuery, load_labeled_queries
from findex.semantic import SearchMode, load_semantic_backend, mode_search
from findex.store import load


def precision(results_books: list[str], labeled: LabeledQuery) -> float:
    return sum(book in labeled.relevant_books for book in results_books[:PRECISION_K]) / PRECISION_K


def main(index_path: Path, embeddings: Path) -> None:
    index = load(index_path)
    store, encoder = load_semantic_backend(embeddings)
    groups = {
        "original": load_labeled_queries(Path("eval/queries.json")),
        "paraphrase": load_labeled_queries(Path("eval/paraphrase_queries.json")),
    }
    print("| group | query | keyword | semantic | hybrid |\n|---|---|---|---|---|")
    totals: dict[str, dict[SearchMode, float]] = {}
    for group, queries in groups.items():
        totals[group] = dict.fromkeys(SearchMode, 0.0)
        for labeled in queries:
            row: list[str] = []
            for mode in SearchMode:
                results = mode_search(index, labeled.query, mode, PRECISION_K, store=store, encoder=encoder)
                value = precision([Path(index.doc_meta[r.doc_id].path).stem for r in results], labeled)
                totals[group][mode] += value / len(queries)
                row.append(f"{value:.1f}")
            print(f"| {group} | {labeled.query} | {' | '.join(row)} |")
    for group, means in totals.items():
        print(f"| **{group} mean** | | " + " | ".join(f"**{means[mode]:.2f}**" for mode in SearchMode) + " |")
    overall = [sum(totals[g][mode] * len(groups[g]) for g in groups) / 15 for mode in SearchMode]
    print("| **all 15** | | " + " | ".join(f"**{value:.2f}**" for value in overall) + " |")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
