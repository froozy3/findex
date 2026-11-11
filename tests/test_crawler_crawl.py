import asyncio
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.crawler.crawl import CrawlStats, Page, crawl

SITE = {
    "/": ["/a", "/b", "/private/x"],
    "/a": ["/b", "/c", "https://other.org/z"],
    "/b": ["/a", "/c/", "/#top"],
    "/c": ["/d"],
    "/d": [],
    "/private/x": ["/hidden"],
    "/hidden": [],
}


def make_site_client(hits: Counter[str], concurrent: list[int] | None = None) -> httpx.AsyncClient:
    active = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active
        path = request.url.path.rstrip("/") or "/"
        hits[path] += 1
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        active += 1
        if concurrent is not None:
            concurrent.append(active)
        await asyncio.sleep(0.01)
        active -= 1
        if path not in SITE:
            return httpx.Response(404)
        links = "".join(f'<a href="{link}">l</a>' for link in SITE[path])
        return httpx.Response(
            200, text=f"<title>{path}</title><p>page {path}</p>{links}", headers={"Content-Type": "text/html"}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


async def collect(client: httpx.AsyncClient, **options: Any) -> list[Page]:
    async with client:
        return [page async for page in crawl(["http://site.test/"], client=client, **options)]


def test_crawl_visits_each_page_once_and_obeys_robots_and_domains() -> None:
    hits: Counter[str] = Counter()
    pages = asyncio.run(collect(make_site_client(hits), max_pages=50, concurrency=4, allowed_domains=["site.test"]))
    assert sorted(urlpath(page.url) for page in pages) == ["/", "/a", "/b", "/c", "/d"]
    assert all(count == 1 for count in hits.values())
    assert "/private/x" not in hits
    assert page_titles(pages)["/c"] == "/c"


def urlpath(url: str) -> str:
    return httpx.URL(url).path or "/"


def page_titles(pages: list[Page]) -> dict[str, str]:
    return {urlpath(page.url): page.title for page in pages}


def test_crawl_stops_at_max_pages() -> None:
    hits: Counter[str] = Counter()
    pages = asyncio.run(collect(make_site_client(hits), max_pages=3, concurrency=2, allowed_domains=["site.test"]))
    assert len(pages) <= 3
    assert sum(count for path, count in hits.items() if path != "/robots.txt") <= 3


def test_per_host_limit_bounds_concurrency() -> None:
    concurrent: list[int] = []
    asyncio.run(
        collect(
            make_site_client(Counter(), concurrent),
            max_pages=50,
            concurrency=6,
            per_host=2,
            allowed_domains=["site.test"],
        )
    )
    assert max(concurrent) <= 2


def test_stats_are_updated() -> None:
    stats = CrawlStats()
    asyncio.run(
        collect(make_site_client(Counter()), max_pages=50, concurrency=3, allowed_domains=["site.test"], stats=stats)
    )
    assert stats.pages == 5
    assert stats.errors == 0
    assert stats.in_flight == 0
    assert stats.pages_per_second > 0


def test_early_exit_of_the_consumer_cancels_workers() -> None:
    async def first_page_only() -> Page:
        client = make_site_client(Counter())
        async with client:
            generator = crawl(["http://site.test/"], client=client, max_pages=50, concurrency=4)
            first = await anext(generator)
            await generator.aclose()
            return first

    assert asyncio.run(first_page_only()).url == "http://site.test/"


class StaticSiteHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/robots.txt":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        links = '<a href="/two">two</a>' if self.path == "/" else ""
        self.wfile.write(f"<title>{self.path}</title><p>searchable zebra {self.path}</p>{links}".encode())

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def static_site_url() -> Any:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StaticSiteHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


def test_cli_crawl_output_feeds_cli_index_and_search(static_site_url: str, tmp_path: Path) -> None:
    runner = CliRunner()
    crawl_file = tmp_path / "crawl.jsonl"
    result = runner.invoke(
        app, ["crawl", static_site_url, "--out", str(crawl_file), "--log-file", str(tmp_path / "crawl.log")]
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in crawl_file.read_text().splitlines()]
    assert len(rows) == 2

    index_path = tmp_path / "index.pkl"
    assert runner.invoke(app, ["index", str(crawl_file), "--out", str(index_path)]).exit_code == 0
    found = runner.invoke(app, ["search", str(index_path), "zebra", "--json"])
    assert len(found.stdout.splitlines()) == 2
    assert "200" in (tmp_path / "crawl.log").read_text()
