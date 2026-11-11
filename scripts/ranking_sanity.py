"""Three ranking sanity checks on the real corpus:  uv run python scripts/ranking_sanity.py index.pkl"""

import sys
from pathlib import Path

from findex.index import Index, Posting
from findex.scoring import BM25, SCORERS
from findex.store import load


def rare_beats_common(index: Index) -> None:
    print("1) rare term outranks common term")
    for name, scorer in SCORERS.items():
        bm25_common = scorer.score("the", Posting(0, 1), index)
        rare_term = "ahab"
        rare = scorer.score(rare_term, Posting(0, 1), index)
        print(
            f"   {name}: score(tf=1) 'the' (df={index.df('the')}) = {bm25_common:.3f}, "
            f"'{rare_term}' (df={index.df(rare_term)}) = {rare:.3f}"
        )


def saturation(index: Index) -> None:
    print("2) BM25 saturation: marginal gain of extra repetitions (average-length document)")
    average = round(index.avg_doc_length)
    doc_id = min(index.doc_lengths, key=lambda d: abs(index.doc_lengths[d] - average))
    scorer = BM25()
    scores = {tf: scorer.score("whale", Posting(doc_id, tf), index) for tf in (1, 2, 10, 20, 21, 40)}
    for tf, value in scores.items():
        print(f"   tf={tf:2d}: {value:.3f}")
    print(f"   gain 1->2: {scores[2] - scores[1]:.3f}   gain 20->21: {scores[21] - scores[20]:.3f}")


def short_beats_long(index: Index) -> None:
    print("3) one match in a short document outranks one match in a long one")
    postings = [p for p in index["whale"] if p.tf == 1]
    short = min(postings, key=lambda p: index.doc_length(p.doc_id))
    long = max(postings, key=lambda p: index.doc_length(p.doc_id))
    scorer = BM25()
    for label, posting in (("short", short), ("long ", long)):
        length = index.doc_length(posting.doc_id)
        print(f"   {label} doc {posting.doc_id} ({length} tokens): {scorer.score('whale', posting, index):.3f}")


if __name__ == "__main__":
    loaded = load(Path(sys.argv[1]))
    rare_beats_common(loaded)
    saturation(loaded)
    short_beats_long(loaded)
