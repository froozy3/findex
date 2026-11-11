import pytest

from findex.tokenize import iter_token_spans, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello World", ["hello", "world"]),
        ("Привіт, Світе! Їжак", ["привіт", "світе", "їжак"]),
        ("café", ["café"]),
        ("café", ["café"]),
        ("Hello, world... again?!", ["hello", "world", "again"]),
        ("don't stop; it’s well-known", ["don't", "stop", "it’s", "well-known"]),
        ("'quoted' -dash-", ["quoted", "dash"]),
        ("year 1984 and 3d", ["year", "1984", "and", "3d"]),
        ("snake_case", ["snake", "case"]),
        ("Straße", ["strasse"]),
        ("", []),
        ("   \n\t ", []),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_tokenize_is_lazy() -> None:
    tokens = tokenize("a b c")
    assert next(tokens) == "a"


def test_token_spans_point_into_normalized_text() -> None:
    text = "Hello, World"
    assert [(token, text[start:end]) for token, start, end in iter_token_spans(text)] == [
        ("hello", "Hello"),
        ("world", "World"),
    ]
