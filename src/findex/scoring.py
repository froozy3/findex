import math
from dataclasses import dataclass
from typing import Protocol

from findex.index import Index, Posting


class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    def score(self, term: str, posting: Posting, index: Index) -> float:
        inverse_document_frequency = math.log(index.num_docs / index.df(term))
        return (1 + math.log(posting.tf)) * inverse_document_frequency


@dataclass(frozen=True, slots=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: Posting, index: Index) -> float:
        document_frequency = index.df(term)
        inverse_document_frequency = math.log(
            1 + (index.num_docs - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        length_ratio = index.doc_length(posting.doc_id) / index.avg_doc_length
        saturated_tf = posting.tf * (self.k1 + 1) / (posting.tf + self.k1 * (1 - self.b + self.b * length_ratio))
        return inverse_document_frequency * saturated_tf


SCORERS: dict[str, Scorer] = {"bm25": BM25(), "tfidf": TfIdf()}
