import inspect
import json
from pathlib import Path

from findex.corpus import (
    extract_title,
    iter_documents,
    load_document_text,
    split_into_chunks,
    strip_gutenberg_boilerplate,
)


def test_iter_documents_is_a_generator(corpus_dir: Path) -> None:
    assert inspect.isgeneratorfunction(iter_documents)
    assert inspect.isgenerator(iter_documents(corpus_dir))


def test_one_document_per_small_file(corpus_dir: Path) -> None:
    documents = list(iter_documents(corpus_dir))
    assert [d.doc_id for d in documents] == [0, 1, 2, 3, 4]
    assert documents[0].title == "a"
    assert documents[1].text == "The quick red fox"


def test_jsonl_corpus_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "pages.jsonl"
    lines = [json.dumps({"title": "One", "text": "first"}), "not json", json.dumps({"url": "u", "text": "second"})]
    path.write_text("\n".join(lines), encoding="utf-8")
    documents = list(iter_documents(path))
    assert [(d.title, d.text, d.locator) for d in documents] == [("One", "first", 0), ("u", "second", 1)]
    assert load_document_text(str(path), 1) == "second"


def test_bad_encoding_does_not_crash(tmp_path: Path) -> None:
    (tmp_path / "latin.txt").write_bytes("caf\xe9 au lait".encode("latin-1"))
    assert [d.text for d in iter_documents(tmp_path)] == ["café au lait"]


def test_split_into_chunks_groups_paragraphs() -> None:
    text = "\n\n".join(["x" * 40] * 5)
    assert [len(chunk) for chunk in split_into_chunks(text, target_chars=100)] == [124, 82]


def test_strip_gutenberg_boilerplate_and_title() -> None:
    raw = "Title: My Book\n\n*** START OF THE PROJECT ***\nbody\n*** END OF THE PROJECT ***\nlicense"
    assert strip_gutenberg_boilerplate(raw).strip() == "body"
    assert extract_title(raw, "fallback") == "My Book"
    assert extract_title("no title here", "fallback") == "fallback"
