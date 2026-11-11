import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import Index
from findex.search import matching_doc_ids
from findex.store import load, open_index, save

runner = CliRunner()


def as_plain_data(index: Index) -> tuple[object, object, object, bool]:
    return index.postings, index.doc_lengths, index.doc_meta, index.has_positions


@pytest.mark.parametrize("suffix", [".pkl", ".json"])
def test_save_load_round_trip(fixture_index: Index, tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"index{suffix}"
    save(fixture_index, path)
    assert as_plain_data(load(path)) == as_plain_data(fixture_index)


def test_open_index_context_manager_releases_cache_on_error(fresh_index: Index, tmp_path: Path) -> None:
    path = tmp_path / "index.pkl"
    save(fresh_index, path)
    with pytest.raises(RuntimeError), open_index(path) as index:
        assert "fox" in index
        raise RuntimeError("boom")
    assert matching_doc_ids.cache_info().currsize == 0


@pytest.fixture(scope="module")
def cli_index_path(corpus_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cli") / "index.pkl"
    result = runner.invoke(app, ["index", str(corpus_dir), "--out", str(path), "--positions"])
    assert result.exit_code == 0, result.output
    return path


def test_cli_search_table(cli_index_path: Path) -> None:
    result = runner.invoke(app, ["search", str(cli_index_path), "python"])
    assert result.exit_code == 0
    assert "python" in result.output.lower()


def test_cli_search_json_lines_are_clean(cli_index_path: Path) -> None:
    result = runner.invoke(app, ["search", str(cli_index_path), "quick", "--json", "--k", "2"])
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 2
    assert set(rows[0]) == {"doc_id", "score", "title", "snippet"}
    assert "\x02" not in rows[0]["snippet"]


def test_cli_search_tfidf_phrase(cli_index_path: Path) -> None:
    result = runner.invoke(app, ["search", str(cli_index_path), '"quick brown"', "--scorer", "tfidf", "--json"])
    assert result.exit_code == 0
    assert len(result.stdout.splitlines()) == 2


def test_cli_stats(cli_index_path: Path) -> None:
    result = runner.invoke(app, ["stats", str(cli_index_path)])
    assert result.exit_code == 0
    assert "documents" in result.output


def test_cli_missing_index_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["stats", str(tmp_path / "nope.pkl")])
    assert result.exit_code == 1
    assert "index not found" in result.output


def test_cli_bad_query_exits_nonzero(cli_index_path: Path) -> None:
    result = runner.invoke(app, ["search", str(cli_index_path), "(broken"])
    assert result.exit_code == 1
    assert "bad query" in result.output


def test_cli_missing_corpus_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", str(tmp_path / "nope"), "--out", str(tmp_path / "i.pkl")])
    assert result.exit_code == 1
