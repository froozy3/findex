"""FastAPI application: JSON API under /search /docs/{id} /stats /health, HTML UI under / and /doc/{id}."""

import html
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from findex.corpus import load_document_text
from findex.index import Index
from findex.query import QuerySyntaxError
from findex.scoring import SCORERS
from findex.search import load_snippet, matching_doc_ids
from findex.semantic import (
    RANKING_DEPTH,
    EmbeddingStore,
    Encoder,
    SearchMode,
    load_semantic_backend,
    mode_search,
)
from findex.snippets import HIGHLIGHT_END, HIGHLIGHT_START
from findex.store import load
from findex.tokenize import tokenize
from findex.web.models import DocumentResponse, HealthResponse, SearchHit, SearchResponse, StatsResponse
from findex.web.settings import Settings, get_settings

log = logging.getLogger("findex.web")
WEB_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=WEB_DIR / "templates")


class ScorerName(StrEnum):
    BM25 = "bm25"
    TFIDF = "tfidf"


def highlight(snippet: str) -> Markup:
    escaped = html.escape(snippet)
    return Markup(escaped.replace(HIGHLIGHT_START, "<mark>").replace(HIGHLIGHT_END, "</mark>"))


templates.env.filters["highlight"] = highlight  # pyright: ignore[reportUnknownMemberType]


def get_optional_index(request: Request) -> Index | None:
    index: Index | None = getattr(request.app.state, "index", None)
    return index


def get_index(index: Annotated[Index | None, Depends(get_optional_index)]) -> Index:
    if index is None:
        raise HTTPException(status_code=503, detail="index is not loaded")
    return index


QueryParam = Annotated[str, Query(min_length=1, max_length=200, description="query, e.g. whale (harpoon OR ahab)")]
PageParam = Annotated[int, Query(ge=1, le=1000)]
SemanticBackend = tuple[EmbeddingStore, Encoder]


def get_semantic(request: Request) -> SemanticBackend | None:
    backend: SemanticBackend | None = getattr(request.app.state, "semantic", None)
    return backend


IndexDependency = Annotated[Index, Depends(get_index)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]
SemanticDependency = Annotated[SemanticBackend | None, Depends(get_semantic)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    app.state.started = time.monotonic()
    app.state.index = None
    app.state.semantic = None
    if settings.embeddings_path is not None:
        try:
            app.state.semantic = load_semantic_backend(settings.embeddings_path)
            log.info("semantic search enabled (%s)", settings.embeddings_path)
        except (OSError, ImportError):
            log.exception("cannot enable semantic search from %s", settings.embeddings_path)
    try:
        app.state.index = load(settings.index_path)
        log.info("loaded %r from %s", app.state.index, settings.index_path)
    except OSError:
        log.exception("cannot load the index from %s", settings.index_path)
    yield
    matching_doc_ids.cache_clear()


def run_search(
    index: Index,
    q: str,
    scorer: ScorerName,
    page: int,
    page_size: int,
    with_snippets: bool = True,
    mode: SearchMode = SearchMode.KEYWORD,
    semantic: SemanticBackend | None = None,
) -> SearchResponse:
    started = time.perf_counter()
    try:
        if mode is SearchMode.KEYWORD:
            total = len(matching_doc_ids(index, q, "merge"))
            ranked = mode_search(index, q, mode, page * page_size, SCORERS[scorer.value])
            terms = _query_terms(q)
        else:
            if semantic is None:
                raise HTTPException(status_code=503, detail=f"{mode.value} search is not available on this server")
            store, encoder = semantic
            ranked = mode_search(index, q, mode, RANKING_DEPTH, SCORERS[scorer.value], store, encoder)
            total = len(ranked)
            terms = frozenset(tokenize(q))
    except QuerySyntaxError as error:
        raise HTTPException(status_code=422, detail=f"bad query: {error}") from None

    hits = [
        SearchHit(
            doc_id=result.doc_id,
            title=result.title,
            score=round(result.score, 4),
            snippet=load_snippet(index, result.doc_id, terms) if with_snippets else "",
        )
        for result in ranked[(page - 1) * page_size : page * page_size]
    ]
    return SearchResponse(
        query=q,
        total=total,
        page=page,
        page_size=page_size,
        took_ms=round((time.perf_counter() - started) * 1000, 2),
        results=hits,
    )


def _query_terms(q: str) -> frozenset[str]:
    from findex.query import parse

    return parse(q).terms()


def create_app() -> FastAPI:
    app = FastAPI(title="findex", version="1.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error (request id %s) on %s", request_id, request.url.path)
            response = JSONResponse({"detail": "internal server error", "request_id": request_id}, status_code=500)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/search", response_model=SearchResponse)
    def search_endpoint(
        index: IndexDependency,
        settings: SettingsDependency,
        semantic: SemanticDependency,
        q: QueryParam,
        k: Annotated[int | None, Query(ge=1, description="page size")] = None,
        scorer: ScorerName = ScorerName.BM25,
        page: PageParam = 1,
        mode: SearchMode = SearchMode.KEYWORD,
    ) -> SearchResponse:
        page_size = min(k or settings.default_page_size, settings.max_page_size)
        return run_search(index, q, scorer, page, page_size, mode=mode, semantic=semantic)

    @app.get("/search-async", response_model=SearchResponse, include_in_schema=False)
    async def search_async_endpoint(
        index: IndexDependency, settings: SettingsDependency, q: QueryParam, page: PageParam = 1
    ) -> SearchResponse:
        """Same work as /search but run on the event loop; exists only to measure `def` vs `async def`."""
        return run_search(index, q, ScorerName.BM25, page, settings.default_page_size)

    @app.get("/docs/{doc_id}", response_model=DocumentResponse)
    def document_endpoint(doc_id: int, index: IndexDependency) -> DocumentResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"document {doc_id} not found")
        return DocumentResponse(
            doc_id=doc_id, title=meta.title, path=Path(meta.path).name, text=load_document_text(meta.path, meta.locator)
        )

    @app.get("/stats", response_model=StatsResponse)
    def stats_endpoint(request: Request, index: IndexDependency) -> StatsResponse:
        return StatsResponse(
            documents=index.num_docs,
            terms=len(index),
            tokens=sum(index.doc_lengths.values()),
            has_positions=index.has_positions,
            uptime_seconds=round(time.monotonic() - request.app.state.started, 1),
        )

    @app.get("/health", response_model=HealthResponse)
    def health_endpoint(index: Annotated[Index | None, Depends(get_optional_index)]) -> HealthResponse:
        if index is None:
            raise HTTPException(status_code=503, detail="index is not loaded")
        return HealthResponse(status="ok")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(
        request: Request,
        index: IndexDependency,
        settings: SettingsDependency,
        semantic: SemanticDependency,
        q: str = "",
        scorer: ScorerName = ScorerName.BM25,
        page: PageParam = 1,
        mode: SearchMode = SearchMode.KEYWORD,
    ) -> Response:
        response: SearchResponse | None = None
        error: str | None = None
        if q.strip():
            try:
                response = run_search(index, q, scorer, page, settings.default_page_size, mode=mode, semantic=semantic)
            except HTTPException as exception:
                error = str(exception.detail)
        context = {
            "q": q,
            "scorer": scorer.value,
            "mode": mode.value,
            "semantic_available": semantic is not None,
            "response": response,
            "error": error,
            "stats": index.num_docs,
        }
        return templates.TemplateResponse(request, "index.html", context)

    @app.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def document_page(request: Request, doc_id: int, index: IndexDependency, q: str = "") -> Response:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"document {doc_id} not found")
        text = load_document_text(meta.path, meta.locator)
        return templates.TemplateResponse(request, "document.html", {"meta": meta, "text": text, "q": q})

    return app


app = create_app()
