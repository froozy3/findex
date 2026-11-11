"""Command-line interface:  findex index | search | stats"""

import asyncio
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from findex.corpus import Document, iter_documents
from findex.crawler.crawl import CrawlStats, crawl, make_client
from findex.crawler.fetch import DEFAULT_USER_AGENT
from findex.crawler.urls import canonicalize
from findex.index import Index, build_index
from findex.parallel import ExecutorKind, build_parallel
from findex.query import QuerySyntaxError
from findex.scoring import SCORERS
from findex.search import SearchResult, load_snippet, search
from findex.semantic import (
    DEFAULT_MODEL,
    SearchMode,
    SentenceTransformerEncoder,
    build_embeddings,
    load_semantic_backend,
    mode_search,
)
from findex.snippets import HIGHLIGHT_END, HIGHLIGHT_START
from findex.stats import limited
from findex.store import load, save
from findex.tokenize import tokenize
from findex.vector import VectorIndex, vector_search

log = logging.getLogger("findex")
app = typer.Typer(no_args_is_help=True, add_completion=False, help="Full-text search over your own corpus.")
console = Console()
error_console = Console(stderr=True)

STATS_TOP_TERMS = 10


class ScorerName(StrEnum):
    BM25 = "bm25"
    TFIDF = "tfidf"


def fail(message: str) -> typer.Exit:
    error_console.print(f"[red]error:[/red] {message}")
    return typer.Exit(code=1)


@app.callback()
def configure_logging(
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="-v: info, -vv: debug")] = 0,
) -> None:
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logging.basicConfig(level=level, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s", force=True)


def snippet_to_text(snippet: str) -> Text:
    text = Text()
    for piece in snippet.split(HIGHLIGHT_START):
        highlighted, separator, plain = piece.partition(HIGHLIGHT_END)
        if separator:
            text.append(highlighted, style="bold yellow")
            text.append(plain)
        else:
            text.append(highlighted)
    return text


def with_progress(documents: Iterator[Document], progress: Progress) -> Iterator[Document]:
    task = progress.add_task("indexing", total=None, unit="docs")
    for document in documents:
        progress.advance(task)
        yield document


@app.command("index")
def index_command(
    corpus: Annotated[Path, typer.Argument(help="directory of .txt files or a .jsonl file")],
    out: Annotated[Path, typer.Option("--out", help="where to save the index (.pkl or .json)")],
    positions: Annotated[bool, typer.Option(help="store word positions (needed for phrase queries)")] = False,
    limit: Annotated[int | None, typer.Option(help="index only the first N documents (serial only)")] = None,
    workers: Annotated[int | None, typer.Option(min=1, help="parallel workers (default: all CPUs)")] = None,
    executor: Annotated[ExecutorKind, typer.Option(help="how to run the workers")] = ExecutorKind.PROCESSES,
) -> None:
    """Build an index from a corpus and save it."""
    if not corpus.exists():
        raise fail(f"corpus not found: {corpus}")

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description} {task.completed} {task.fields[unit]}"),
        TimeElapsedColumn(),
        console=error_console,
    ) as progress:
        if limit is not None:
            documents = with_progress(limited(iter_documents(corpus), limit), progress)
            index = build_index(documents, positions)
        else:
            task = progress.add_task("indexing", total=None, unit="chunks")
            index, _ = build_parallel(corpus, positions, executor, workers, lambda: progress.advance(task))

    save(index, out)
    log.info("saved %r to %s", index, out)
    console.print(f"{index!r} -> {out}")


def load_index_or_fail(path: Path) -> Index:
    try:
        return load(path)
    except FileNotFoundError:
        raise fail(f"index not found: {path}") from None


@app.command("search")
def search_command(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX")],
    query: Annotated[str, typer.Argument(help="e.g. 'whale (harpoon OR ahab) NOT pequod \"white whale\"'")],
    k: Annotated[int, typer.Option(min=1, help="number of results")] = 10,
    scorer: Annotated[ScorerName, typer.Option(help="ranking function")] = ScorerName.BM25,
    as_json: Annotated[bool, typer.Option("--json", help="emit JSON lines on stdout")] = False,
    mode: Annotated[SearchMode, typer.Option(help="keyword, semantic (embeddings) or hybrid (RRF of both)")] = (
        SearchMode.KEYWORD
    ),
    embeddings: Annotated[Path, typer.Option(help="directory made by `findex embed`")] = Path("data/embeddings"),
) -> None:
    """Search a saved index (a file) or a vectorized index (a directory made by `findex vectorize`)."""
    try:
        if index_path.is_dir():
            results = vector_search(VectorIndex.load(index_path), query, scorer.value, k)
        elif mode is SearchMode.KEYWORD:
            results = search(load_index_or_fail(index_path), query, SCORERS[scorer.value], k, with_snippets=True)
        else:
            index = load_index_or_fail(index_path)
            store, encoder = load_semantic_backend(embeddings)
            results = mode_search(index, query, mode, k, SCORERS[scorer.value], store, encoder)
            terms = frozenset(tokenize(query))
            results = [SearchResult(r.doc_id, r.score, r.title, load_snippet(index, r.doc_id, terms)) for r in results]
    except QuerySyntaxError as error:
        raise fail(f"bad query: {error}") from None
    except FileNotFoundError:
        raise fail(f"index not found: {index_path}") from None

    if as_json:
        for result in results:
            clean_snippet = result.snippet.replace(HIGHLIGHT_START, "").replace(HIGHLIGHT_END, "")
            print(json.dumps({**result_as_dict(result), "snippet": clean_snippet}, ensure_ascii=False))
        return

    table = Table(show_lines=True)
    for column in ("#", "score", "title", "snippet"):
        table.add_column(column)
    for rank, result in enumerate(results, start=1):
        table.add_row(str(rank), f"{result.score:.2f}", result.title, snippet_to_text(result.snippet))
    console.print(table if results else "[yellow]no results[/yellow]")


def result_as_dict(result: SearchResult) -> dict[str, object]:
    return {"doc_id": result.doc_id, "score": round(result.score, 4), "title": result.title, "snippet": result.snippet}


@app.command("stats")
def stats_command(index_path: Annotated[Path, typer.Argument(metavar="INDEX")]) -> None:
    """Show corpus numbers for a saved index."""
    index = load_index_or_fail(index_path)
    term_totals = Counter({term: sum(posting.tf for posting in postings) for term, postings in index.items()})

    table = Table(show_header=False)
    table.add_row("documents", f"{index.num_docs:,}")
    table.add_row("tokens", f"{sum(index.doc_lengths.values()):,}")
    table.add_row("vocabulary", f"{len(index):,}")
    table.add_row("avg document length", f"{index.avg_doc_length:.0f}")
    table.add_row("positions", "yes" if index.has_positions else "no")
    table.add_row(
        "top terms", ", ".join(f"{term} ({count:,})" for term, count in term_totals.most_common(STATS_TOP_TERMS))
    )
    console.print(table)


def crawl_table(stats: CrawlStats) -> Table:
    table = Table(title="crawling", show_header=False)
    table.add_row("pages", str(stats.pages))
    table.add_row("pages/s", f"{stats.pages_per_second:.1f}")
    table.add_row("in flight", str(stats.in_flight))
    table.add_row("errors", str(stats.errors))
    table.add_row("queue depth", str(stats.queue_depth))
    return table


async def crawl_to_jsonl(
    seed: str,
    out: Path,
    max_pages: int,
    concurrency: int,
    per_host: int,
    delay: float,
    user_agent: str,
    domains: list[str],
) -> CrawlStats:
    stats = CrawlStats()
    out.parent.mkdir(parents=True, exist_ok=True)
    allowed = domains or [canonicalize(seed).split("//", 1)[1].split("/", 1)[0].split(":")[0]]
    async with make_client(user_agent, concurrency) as client:
        with (
            out.open("w", encoding="utf-8") as sink,
            Live(crawl_table(stats), console=error_console, refresh_per_second=4) as live,
        ):
            async for page in crawl(
                [seed],
                client=client,
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                allowed_domains=allowed,
                min_delay=delay,
                user_agent=user_agent,
                stats=stats,
            ):
                sink.write(
                    json.dumps(
                        {"url": page.url, "title": page.title, "text": page.text, "fetched_at": page.fetched_at},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sink.flush()
                live.update(crawl_table(stats))
    return stats


@app.command("crawl")
def crawl_command(
    seed: Annotated[str, typer.Argument(help="start URL; only crawl sites you have permission to crawl")],
    max_pages: Annotated[int, typer.Option(min=1, help="page budget")] = 500,
    concurrency: Annotated[int, typer.Option(min=1, help="simultaneous requests")] = 10,
    per_host: Annotated[int, typer.Option(min=1, help="simultaneous requests per host")] = 2,
    delay: Annotated[float, typer.Option(min=0.0, help="minimum seconds between requests to one host")] = 0.0,
    out: Annotated[Path, typer.Option(help="JSON-lines output")] = Path("data/crawl.jsonl"),
    domain: Annotated[list[str] | None, typer.Option(help="allowed domain (default: the seed's host)")] = None,
    contact: Annotated[str, typer.Option(help="contact URL/e-mail put into the User-Agent")] = "",
    log_file: Annotated[Path, typer.Option(help="per-URL log")] = Path("crawl.log"),
) -> None:
    """Crawl a site politely and write one JSON line per page."""
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    crawler_log = logging.getLogger("findex.crawler")
    crawler_log.setLevel(logging.INFO)
    crawler_log.addHandler(file_handler)

    user_agent = f"findex-crawler/0.6 (+{contact}; student project)" if contact else DEFAULT_USER_AGENT
    stats = asyncio.run(crawl_to_jsonl(seed, out, max_pages, concurrency, per_host, delay, user_agent, domain or []))
    console.print(f"{stats.pages} pages ({stats.errors} errors) in {stats.elapsed:.1f}s -> {out}")


@app.command("serve")
def serve_command(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="saved index to serve")] = Path("index.pkl"),
    host: Annotated[str, typer.Option(help="interface to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="port")] = 8000,
    workers: Annotated[int, typer.Option(min=1, help="uvicorn worker processes")] = 1,
) -> None:
    """Serve the search API and web UI."""
    import uvicorn

    os.environ["INDEX_PATH"] = str(index_path)
    uvicorn.run("findex.web.app:app", host=host, port=port, workers=workers, log_level="info")


@app.command("vectorize")
def vectorize_command(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX")],
    out: Annotated[Path, typer.Option("--out", help="directory for the .npy arrays")],
) -> None:
    """Convert an index to memory-mappable NumPy arrays for fast vectorized search."""
    vector_index = VectorIndex.from_index(load_index_or_fail(index_path))
    vector_index.save(out)
    console.print(f"{len(vector_index.terms):,} terms, {len(vector_index.doc_ids):,} postings -> {out}")


@app.command("embed")
def embed_command(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX")],
    out: Annotated[Path, typer.Option("--out", help="directory for the .npy embeddings")] = Path("data/embeddings"),
    model: Annotated[str, typer.Option(help="sentence-transformers model")] = DEFAULT_MODEL,
) -> None:
    """Embed every passage window (needs `uv sync --extra semantic`) for semantic and hybrid search."""
    index = load_index_or_fail(index_path)
    encoder = SentenceTransformerEncoder(model)
    with Progress(
        SpinnerColumn(), TextColumn("embedding {task.completed} windows"), TimeElapsedColumn(), console=error_console
    ) as progress:
        task = progress.add_task("embed", total=None)
        store = build_embeddings(index, encoder, on_batch=lambda count: progress.advance(task, count))
    store.save(out)
    console.print(f"{len(store.vectors):,} windows x {store.vectors.shape[1]} dims -> {out}")
