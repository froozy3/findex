"""Query language: recursive-descent parser and an expression tree.

or_expr  := and_expr ("OR" and_expr)*
and_expr := not_expr (["AND"] not_expr)*      # implicit AND between neighbours
not_expr := "NOT" not_expr | atom
atom     := "(" or_expr ")" | "quoted phrase" | word
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from findex.index import Index
from findex.setops import OPERATIONS, Engine
from findex.tokenize import tokenize

QUERY_TOKEN_PATTERN = re.compile(r'"[^"]*"?|\(|\)|[^\s()"]+')


class QuerySyntaxError(ValueError):
    pass


class Node(ABC):
    @abstractmethod
    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]: ...

    @abstractmethod
    def terms(self) -> frozenset[str]:
        """Terms that count towards ranking (everything outside `NOT`)."""

    def __and__(self, other: "Node") -> "And":
        return And(self, other)

    def __or__(self, other: "Node") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)


@dataclass(frozen=True, slots=True)
class Term(Node):
    term: str

    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]:
        return [posting.doc_id for posting in index.postings.get(self.term, ())]

    def terms(self) -> frozenset[str]:
        return frozenset({self.term})


@dataclass(frozen=True, slots=True)
class Phrase(Node):
    words: tuple[str, ...]

    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]:
        if not index.has_positions:
            raise QuerySyntaxError("phrase queries need an index built with positions (--positions)")
        and_operation = OPERATIONS[engine][0]
        candidates = Term(self.words[0]).evaluate(index, engine)
        for word in self.words[1:]:
            candidates = and_operation(candidates, Term(word).evaluate(index, engine))

        positions_by_word = [
            {posting.doc_id: posting.positions for posting in index.postings.get(word, ())} for word in self.words
        ]
        return [
            doc_id
            for doc_id in candidates
            if any(
                all(start + offset in set(positions_by_word[offset][doc_id]) for offset in range(1, len(self.words)))
                for start in positions_by_word[0][doc_id]
            )
        ]

    def terms(self) -> frozenset[str]:
        return frozenset(self.words)


@dataclass(frozen=True, slots=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]:
        return OPERATIONS[engine][0](self.left.evaluate(index, engine), self.right.evaluate(index, engine))

    def terms(self) -> frozenset[str]:
        return self.left.terms() | self.right.terms()


@dataclass(frozen=True, slots=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]:
        return OPERATIONS[engine][1](self.left.evaluate(index, engine), self.right.evaluate(index, engine))

    def terms(self) -> frozenset[str]:
        return self.left.terms() | self.right.terms()


@dataclass(frozen=True, slots=True)
class Not(Node):
    child: Node

    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]:
        return OPERATIONS[engine][2](index.all_doc_ids(), self.child.evaluate(index, engine))

    def terms(self) -> frozenset[str]:
        return frozenset()


class Parser:
    def __init__(self, text: str) -> None:
        self.tokens = QUERY_TOKEN_PATTERN.findall(text)
        self.position = 0

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def advance(self) -> str:
        token = self.tokens[self.position]
        self.position += 1
        return token

    def parse(self) -> Node:
        if not self.tokens:
            raise QuerySyntaxError("empty query")
        node = self.parse_or()
        if self.peek() is not None:
            raise QuerySyntaxError(f"unexpected {self.peek()!r}")
        return node

    def parse_or(self) -> Node:
        node = self.parse_and()
        while self.peek() == "OR":
            self.advance()
            node = Or(node, self.parse_and())
        return node

    def parse_and(self) -> Node:
        node = self.parse_not()
        while self.peek() is not None and self.peek() not in {"OR", ")"}:
            if self.peek() == "AND":
                self.advance()
            node = And(node, self.parse_not())
        return node

    def parse_not(self) -> Node:
        if self.peek() == "NOT":
            self.advance()
            return Not(self.parse_not())
        return self.parse_atom()

    def parse_atom(self) -> Node:
        token = self.peek()
        if token is None:
            raise QuerySyntaxError("query ends unexpectedly")
        if token == "(":
            self.advance()
            node = self.parse_or()
            if self.peek() != ")":
                raise QuerySyntaxError("missing closing parenthesis")
            self.advance()
            return node
        if token in {")", "AND", "OR"}:
            raise QuerySyntaxError(f"unexpected {token!r}")

        self.advance()
        if token.startswith('"'):
            words = tuple(tokenize(token.strip('"')))
            if not words:
                raise QuerySyntaxError("empty phrase")
            return Term(words[0]) if len(words) == 1 else Phrase(words)

        words = tuple(tokenize(token))
        if not words:
            raise QuerySyntaxError(f"no searchable word in {token!r}")
        node: Node = Term(words[0])
        for extra_word in words[1:]:
            node = And(node, Term(extra_word))
        return node


def parse(text: str) -> Node:
    return Parser(text).parse()
