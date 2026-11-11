# Starlette's TestClient is only partially typed, so strict checking of this one file is relaxed
# pyright: basic
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from findex.index import Index
from findex.web.app import create_app, get_index, get_optional_index


@pytest.fixture
def client(fixture_index: Index) -> Iterator[TestClient]:
    app = create_app()
    app.state.started = time.monotonic()
    app.dependency_overrides[get_index] = lambda: fixture_index
    app.dependency_overrides[get_optional_index] = lambda: fixture_index
    yield TestClient(app)


@pytest.fixture
def client_without_index() -> TestClient:
    app = create_app()
    app.state.index = None
    return TestClient(app)


def test_search_returns_ranked_results_with_snippets(client: TestClient) -> None:
    response = client.get("/search", params={"q": "python"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["results"][0]["title"] == "d"
    assert "\x02" in body["results"][0]["snippet"]
    assert body["took_ms"] >= 0


def test_search_pagination(client: TestClient) -> None:
    first = client.get("/search", params={"q": "quick OR lazy", "k": 2, "page": 1}).json()
    second = client.get("/search", params={"q": "quick OR lazy", "k": 2, "page": 2}).json()
    assert first["total"] == second["total"] == 4
    assert len(first["results"]) == 2
    assert len(second["results"]) == 2
    assert not {r["doc_id"] for r in first["results"]} & {r["doc_id"] for r in second["results"]}


def test_search_tfidf_scorer(client: TestClient) -> None:
    assert client.get("/search", params={"q": "fox", "scorer": "tfidf"}).status_code == 200


@pytest.mark.parametrize(
    "params",
    [{}, {"q": ""}, {"q": "x" * 300}, {"q": "a", "k": 0}, {"q": "a", "page": 0}, {"q": "a", "scorer": "nope"}],
)
def test_search_validation_errors_are_422(client: TestClient, params: dict[str, object]) -> None:
    assert client.get("/search", params=params).status_code == 422  # type: ignore[arg-type]  # httpx accepts mixed values


def test_bad_query_syntax_is_422_with_detail(client: TestClient) -> None:
    response = client.get("/search", params={"q": "(broken"})
    assert response.status_code == 422
    assert "bad query" in response.json()["detail"]


def test_document_found_and_not_found(client: TestClient) -> None:
    found = client.get("/docs/3")
    assert found.status_code == 200
    assert "Python is a programming language" in found.json()["text"]
    missing = client.get("/docs/999")
    assert missing.status_code == 404
    assert "detail" in missing.json()


def test_stats(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["documents"] == 5
    assert body["has_positions"] is True
    assert body["uptime_seconds"] >= 0


def test_health_ok_and_503_when_index_missing(client: TestClient, client_without_index: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client_without_index.get("/health").status_code == 503
    assert client_without_index.get("/search", params={"q": "a"}).status_code == 503


def test_openapi_docs_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert "/search" in client.get("/openapi.json").json()["paths"]


def test_html_ui_highlights_and_links(client: TestClient) -> None:
    page = client.get("/", params={"q": "python"})
    assert page.status_code == 200
    assert "<mark>" in page.text
    assert "/doc/3?q=python" in page.text
    assert client.get("/").status_code == 200


def test_html_ui_shows_query_errors(client: TestClient) -> None:
    assert "bad query" in client.get("/", params={"q": "("}).text


def test_document_page(client: TestClient) -> None:
    assert client.get("/doc/0").status_code == 200
    assert client.get("/doc/999").status_code == 404


def test_request_id_header_and_clean_500(fixture_index: Index) -> None:
    app = create_app()

    def exploding_index() -> Index:
        raise RuntimeError("boom")

    app.dependency_overrides[get_index] = exploding_index
    quiet_client = TestClient(app, raise_server_exceptions=False)
    response = quiet_client.get("/search", params={"q": "a"}, headers={"X-Request-ID": "abc123"})
    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error", "request_id": "abc123"}
    assert response.headers["X-Request-ID"] == "abc123"


def test_async_variant_returns_the_same_hits(client: TestClient) -> None:
    sync_ids = [r["doc_id"] for r in client.get("/search", params={"q": "quick"}).json()["results"]]
    async_ids = [r["doc_id"] for r in client.get("/search-async", params={"q": "quick"}).json()["results"]]
    assert sync_ids == async_ids


def test_semantic_mode_503_without_embeddings(client: TestClient) -> None:
    response = client.get("/search", params={"q": "fox", "mode": "semantic"})
    assert response.status_code == 503
    assert "not available" in response.json()["detail"]


def test_semantic_and_hybrid_modes_with_a_backend(fixture_index: Index) -> None:
    from findex.semantic import build_embeddings
    from findex.web.app import get_semantic
    from tests.test_semantic import HashingEncoder

    encoder = HashingEncoder()
    backend = (build_embeddings(fixture_index, encoder), encoder)
    app = create_app()
    app.state.started = time.monotonic()
    app.dependency_overrides[get_index] = lambda: fixture_index
    app.dependency_overrides[get_semantic] = lambda: backend
    test_client = TestClient(app)

    for mode in ("semantic", "hybrid"):
        body = test_client.get("/search", params={"q": "python programming", "mode": mode}).json()
        assert body["results"][0]["title"] == "d"
    assert "Semantic" in test_client.get("/").text
    assert test_client.get("/", params={"q": "python", "mode": "hybrid"}).status_code == 200
