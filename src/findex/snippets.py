"""Snippets: a window around the densest cluster of matches, with matches wrapped in markers."""

import re
from bisect import bisect_left, bisect_right
from collections.abc import Collection
from functools import lru_cache

from findex.tokenize import iter_token_spans, normalize_text

HIGHLIGHT_START = "\x02"
HIGHLIGHT_END = "\x03"
SNIPPET_RADIUS = 80


TOKEN_CHAR = r"[^\W_]"
JOINERS = "'\u2019\\-"


@lru_cache(maxsize=512)
def compile_terms_pattern(terms: frozenset[str]) -> re.Pattern[str]:
    """One regex for all query terms that matches exactly where the tokenizer would produce that term."""
    alternation = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
    before = rf"(?<!{TOKEN_CHAR})(?<!{TOKEN_CHAR}[{JOINERS}])"
    after = rf"(?!{TOKEN_CHAR})(?![{JOINERS}]{TOKEN_CHAR})"
    return re.compile(f"{before}(?:{alternation}){after}", re.IGNORECASE)


def find_term_spans(text: str, terms: Collection[str]) -> list[tuple[int, int]]:
    """Fast path: scan with one compiled regex instead of tokenizing the whole document."""
    if not terms:
        return []
    if text.casefold() != text.lower():  # full case folding (ß -> ss) is beyond re.IGNORECASE: take the exact path
        return find_term_spans_by_tokenizing(text, terms)
    return [match.span() for match in compile_terms_pattern(frozenset(terms)).finditer(text)]


def find_term_spans_by_tokenizing(text: str, terms: Collection[str]) -> list[tuple[int, int]]:
    """Reference implementation (slow): tokenize everything, keep tokens that are query terms."""
    return [(start, end) for token, start, end in iter_token_spans(text) if token in terms]


def make_snippet(text: str, terms: Collection[str], radius: int = SNIPPET_RADIUS) -> str:
    normalized = normalize_text(text)
    spans = find_term_spans(normalized, terms)
    if not spans:
        return " ".join(normalized[: radius * 2].split())

    starts = [start for start, _ in spans]
    best_start = max(
        starts,
        key=lambda start: bisect_right(starts, start + radius) - bisect_left(starts, start - radius),
    )
    window_start = max(0, best_start - radius)
    window_end = min(len(normalized), best_start + radius)

    pieces: list[str] = []
    cursor = window_start
    for start, end in spans:
        if start < window_start or end > window_end:
            continue
        pieces.append(normalized[cursor:start])
        pieces.append(f"{HIGHLIGHT_START}{normalized[start:end]}{HIGHLIGHT_END}")
        cursor = end
    pieces.append(normalized[cursor:window_end])

    snippet = " ".join("".join(pieces).split())
    return f"{'…' if window_start > 0 else ''}{snippet}{'…' if window_end < len(normalized) else ''}"
