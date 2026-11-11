"""Streaming tokenizer.

Policy:
  * text is NFC-normalized, tokens are casefolded;
  * a token is a run of letters/digits (Unicode-aware, `_` excluded);
  * an apostrophe or hyphen between two letters/digits stays inside the token
    (`don't`, `well-known`); leading/trailing ones are dropped;
  * digits are tokens (`1984`, `3d`).
"""

import re
import unicodedata
from collections.abc import Iterator

APOSTROPHES = "'\u2019"
WORD_PATTERN = re.compile(rf"[^\W_]+(?:[{APOSTROPHES}\-][^\W_]+)*")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def iter_token_spans(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield `(token, start, end)`; offsets point into the NFC-normalized text."""
    for match in WORD_PATTERN.finditer(text):
        yield match.group().casefold(), match.start(), match.end()


def tokenize(text: str) -> Iterator[str]:
    for match in WORD_PATTERN.finditer(normalize_text(text)):
        yield match.group().casefold()
