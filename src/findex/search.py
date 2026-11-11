import heapq
import logging
from dataclasses import dataclass, field
from functools import lru_cache

from findex.corpus import load_document_text
from findex.index import Index
from findex.query import parse
from findex.scoring import BM25, Scorer
from findex.setops import Engine
from findex.snippets import make_snippet
from findex.timing import timed

log = logging.getLogger(__name__)

DEFAULT_SCORER = BM25()


@dataclass(frozen=True, order=True, slots=True)
class SearchResult:
    """Sorting a list of results puts the best first (higher score, then lower doc_id)."""

    sort_key: tuple[float, int] = field(init=False, repr=False)
    doc_id: int = field(compare=False)
    score: float = field(compare=False)
    title: str = field(compare=False)
    snippet: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_key", (-self.score, self.doc_id))


@lru_cache(maxsize=256)
def matching_doc_ids(index: Index, query: str, engine: Engine) -> tuple[int, ...]:
    return tuple(parse(query).evaluate(index, engine))


def score_documents(index: Index, query: str, scorer: Scorer, engine: Engine) -> dict[int, float]:
    matched = set(matching_doc_ids(index, query, engine))
    scores = dict.fromkeys(matched, 0.0)
    for term in sorted(parse(query).terms()):
        for posting in index.postings.get(term, ()):
            if posting.doc_id in matched:
                scores[posting.doc_id] += scorer.score(term, posting, index)
    return scores


def load_snippet(index: Index, doc_id: int, terms: frozenset[str]) -> str:
    meta = index.doc_meta[doc_id]
    try:
        return make_snippet(load_document_text(meta.path, meta.locator), terms)
    except (OSError, IndexError):
        log.warning("cannot build a snippet for document %d (%s)", doc_id, meta.path)
        return ""


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer = DEFAULT_SCORER,
    k: int = 10,
    engine: Engine = "merge",
    with_snippets: bool = False,
) -> list[SearchResult]:
    scores = score_documents(index, query, scorer, engine)
    top = heapq.nlargest(k, scores.items(), key=lambda item: (item[1], -item[0]))
    terms = parse(query).terms()
    return [
        SearchResult(
            doc_id,
            score,
            index.doc_meta[doc_id].title,
            load_snippet(index, doc_id, terms) if with_snippets else "",
        )
        for doc_id, score in top
    ]
