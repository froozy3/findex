import pytest

from findex.index import Index
from findex.query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse
from findex.setops import Engine


def test_or_binds_looser_than_implicit_and() -> None:
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))


def test_implicit_and_is_left_associative() -> None:
    assert parse("a b c") == And(And(Term("a"), Term("b")), Term("c"))


def test_explicit_and_equals_implicit() -> None:
    assert parse("a AND b") == parse("a b")


def test_not_and_parentheses() -> None:
    assert parse("a NOT b") == And(Term("a"), Not(Term("b")))
    assert parse("(a OR b) c") == And(Or(Term("a"), Term("b")), Term("c"))


def test_phrase_and_single_word_phrase() -> None:
    assert parse('"quick brown fox"') == Phrase(("quick", "brown", "fox"))
    assert parse('"Fox"') == Term("fox")


def test_terms_are_normalized() -> None:
    assert parse("QUICK") == Term("quick")


def test_node_operators_build_the_same_trees() -> None:
    assert Term("a") & Term("b") == And(Term("a"), Term("b"))
    assert Term("a") | Term("b") == Or(Term("a"), Term("b"))
    assert ~Term("a") == Not(Term("a"))


@pytest.mark.parametrize("text", ["", "   ", "(a", "a)", "OR a", "a OR", "NOT", '""', "!!!"])
def test_syntax_errors(text: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse(text)


@pytest.mark.parametrize("engine", ["merge", "set"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("quick fox", [0, 1]),
        ("fox OR dog", [0, 1, 4]),
        ("quick NOT brown", [1]),
        ("(lazy OR red) fox", [0, 1]),
        ('"quick brown"', [0, 4]),
        ('"brown quick"', []),
        ("missing", []),
    ],
)
def test_evaluate(fixture_index: Index, engine: Engine, query: str, expected: list[int]) -> None:
    assert parse(query).evaluate(fixture_index, engine) == expected


def test_phrase_requires_positions(fresh_index: Index) -> None:
    without_positions = Index(fresh_index.postings, fresh_index.doc_lengths, fresh_index.doc_meta, has_positions=False)
    with pytest.raises(QuerySyntaxError, match="positions"):
        parse('"quick fox"').evaluate(without_positions)


def test_terms_exclude_negated_words() -> None:
    assert parse("a NOT b (c OR d)").terms() == {"a", "c", "d"}
