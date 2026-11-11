import pytest

from findex.index import Index, Posting
from findex.query import QuerySyntaxError
from findex.scoring import BM25, SCORERS, Scorer, TfIdf
from findex.search import SearchResult, matching_doc_ids, search
from findex.snippets import HIGHLIGHT_END, HIGHLIGHT_START, make_snippet


@pytest.mark.parametrize("scorer", list(SCORERS.values()), ids=list(SCORERS))
def test_rare_term_outranks_common_term(fixture_index: Index, scorer: Scorer) -> None:
    common = scorer.score("the", Posting(0, 1), fixture_index)
    rare = scorer.score("python", Posting(0, 1), fixture_index)
    assert rare > common


def test_bm25_saturates_repeated_terms(fixture_index: Index) -> None:
    scorer = BM25()
    score = {tf: scorer.score("fox", Posting(0, tf), fixture_index) for tf in (1, 2, 20, 21)}
    assert score[2] - score[1] > 10 * (score[21] - score[20])


def test_short_document_outranks_long_document_with_same_tf(fixture_index: Index) -> None:
    scorer = BM25()
    short = scorer.score("quick", Posting(1, 1), fixture_index)  # 4 tokens
    long = scorer.score("quick", Posting(0, 1), fixture_index)  # 9 tokens
    assert short > long


def test_tfidf_ignores_length_but_not_tf(fixture_index: Index) -> None:
    scorer = TfIdf()
    assert scorer.score("fox", Posting(0, 3), fixture_index) > scorer.score("fox", Posting(0, 1), fixture_index)


def test_search_ranks_and_limits(fixture_index: Index) -> None:
    results = search(fixture_index, "quick OR lazy", k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_search_best_document_first(fixture_index: Index) -> None:
    assert search(fixture_index, "python")[0].title == "d"


def test_results_sort_best_first() -> None:
    low, high = SearchResult(1, 1.0, "low"), SearchResult(2, 5.0, "high")
    assert sorted([low, high]) == [high, low]


@pytest.mark.parametrize("scorer", list(SCORERS.values()), ids=list(SCORERS))
def test_merge_and_set_engines_agree(fixture_index: Index, scorer: Scorer) -> None:
    query = "(quick OR lazy) NOT red"
    assert search(fixture_index, query, scorer, engine="merge") == search(fixture_index, query, scorer, engine="set")


def test_repeated_query_hits_the_cache(fresh_index: Index) -> None:
    matching_doc_ids.cache_clear()
    search(fresh_index, "quick fox")
    search(fresh_index, "quick fox")
    assert matching_doc_ids.cache_info().hits >= 1


def test_bad_query_raises(fixture_index: Index) -> None:
    with pytest.raises(QuerySyntaxError):
        search(fixture_index, "(")


def test_snippet_highlights_matches() -> None:
    snippet = make_snippet("The quick brown fox jumps", {"fox"}, radius=10)
    assert f"{HIGHLIGHT_START}fox{HIGHLIGHT_END}" in snippet


def test_snippet_without_match_is_a_text_prefix() -> None:
    assert make_snippet("hello   world", {"zzz"}, radius=50) == "hello world"


def test_snippet_ellipsis_on_long_text() -> None:
    text = "filler " * 100 + "needle " + "filler " * 100
    snippet = make_snippet(text, {"needle"}, radius=20)
    assert snippet.startswith("…")
    assert snippet.endswith("…")


@pytest.mark.parametrize(
    "text",
    [
        "The quick brown fox, the QUICK red fox; quick-witted fox's den; don't stop fox_trot.",
        "Привіт світе, ПРИВІТ! Їжак and café café Straße STRASSE",
        "no matches here at all",
        "fox-like foxes fox. 'fox' (fox) fox’s fox’s-x",
    ],
)
@pytest.mark.parametrize(
    "terms", [{"fox"}, {"quick", "fox", "the"}, {"привіт", "їжак"}, {"café", "strasse"}, {"don't"}]
)
def test_fast_span_search_agrees_with_tokenizing(text: str, terms: set[str]) -> None:
    from findex.snippets import find_term_spans, find_term_spans_by_tokenizing
    from findex.tokenize import normalize_text

    normalized = normalize_text(text)
    assert find_term_spans(normalized, terms) == find_term_spans_by_tokenizing(normalized, terms)
