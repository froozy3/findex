import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest

from findex.crawler.extract import extract_page
from findex.crawler.fetch import fetch, retry_delay
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, is_domain_allowed, resolve_link


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True, max_redirects=3)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTP://Example.COM/Path/", "http://example.com/Path"),
        ("https://example.com", "https://example.com/"),
        ("https://example.com/a#frag", "https://example.com/a"),
        ("https://example.com/a?b=2&a=1", "https://example.com/a?a=1&b=2"),
        ("http://example.com:80/a", "http://example.com/a"),
        ("http://example.com:8080/a", "http://example.com:8080/a"),
    ],
)
def test_canonicalize(raw: str, expected: str) -> None:
    assert canonicalize(raw) == expected


def test_resolve_link() -> None:
    assert resolve_link("https://a.com/x/y", "../z#top") == "https://a.com/z"
    assert resolve_link("https://a.com/", "mailto:me@a.com") is None
    assert resolve_link("https://a.com/", "/logo.png") is None


def test_domain_allowlist() -> None:
    allowed = frozenset({"example.com"})
    assert is_domain_allowed("https://example.com/a", allowed)
    assert is_domain_allowed("https://docs.example.com/a", allowed)
    assert not is_domain_allowed("https://evil-example.com/a", allowed)
    assert is_domain_allowed("https://anything.org/", frozenset())


def test_fetch_success() -> None:
    result = run(
        fetch(client_for(lambda _: httpx.Response(200, text="hi", headers={"Content-Type": "text/html"})), "http://x/")
    )
    assert result.ok
    assert result.content == b"hi"
    assert result.attempts == 1


def test_fetch_retries_5xx_then_succeeds() -> None:
    statuses = iter([503, 502, 200])
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = run(fetch(client_for(lambda _: httpx.Response(next(statuses))), "http://x/", sleep=fake_sleep))
    assert result.ok
    assert result.attempts == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0] * 0.6  # exponential growth, jitter allowing


def test_fetch_honours_retry_after() -> None:
    responses = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)])
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = run(fetch(client_for(lambda _: next(responses)), "http://x/", sleep=fake_sleep))
    assert result.ok
    assert sleeps == [7.0]


def test_fetch_does_not_retry_other_4xx() -> None:
    calls: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404)

    result = run(fetch(client_for(handler), "http://x/"))
    assert not result.ok
    assert result.status == 404
    assert result.error == "HTTP 404"
    assert len(calls) == 1


def test_fetch_gives_up_after_max_retries() -> None:
    async def fake_sleep(_: float) -> None:
        return None

    result = run(fetch(client_for(lambda _: httpx.Response(500)), "http://x/", max_retries=2, sleep=fake_sleep))
    assert not result.ok
    assert result.attempts == 3
    assert result.error == "HTTP 500"


def test_fetch_retries_timeouts() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200)

    async def fake_sleep(_: float) -> None:
        return None

    assert run(fetch(client_for(handler), "http://x/", sleep=fake_sleep)).ok
    assert len(attempts) == 3


def test_fetch_stops_at_redirect_loop() -> None:
    result = run(fetch(client_for(lambda _: httpx.Response(302, headers={"Location": "http://x/"})), "http://x/"))
    assert result.error == "redirect loop"
    assert result.attempts == 1


def test_retry_delay_caps_and_ignores_garbage() -> None:
    assert retry_delay(0, "3") == 3.0
    assert retry_delay(0, "999") == 60.0
    assert 0.25 <= retry_delay(0, "Wed, 21 Oct 2026 07:28:00 GMT") <= 0.75
    assert retry_delay(20, None) <= 45.0


def test_robots_allow_disallow_and_cache() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text="User-agent: *\nDisallow: /private/\nCrawl-delay: 2\n")

    cache = RobotsCache(client_for(handler), "findex-test")
    assert run(cache.allowed("http://x/public"))
    assert not run(cache.allowed("http://x/private/a"))
    assert run(cache.crawl_delay("http://x/public")) == 2.0
    assert calls == ["http://x/robots.txt"]


@pytest.mark.parametrize(("status", "expected"), [(404, True), (503, False)])
def test_robots_missing_allows_all_but_server_error_disallows(status: int, expected: bool) -> None:
    cache = RobotsCache(client_for(lambda _: httpx.Response(status)), "findex-test")
    assert run(cache.allowed("http://x/anything")) is expected


def test_robots_unreachable_disallows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    assert not run(RobotsCache(client_for(handler), "findex-test").allowed("http://x/a"))


def test_extract_page() -> None:
    html = b"""<html><head><title> My Page </title><style>x{}</style></head><body>
    <nav>menu</nav><p>Hello <b>world</b></p><script>var a=1</script>
    <a href="/next#x">n</a><a href="https://other.org/">o</a><a href="mailto:a@b.c">m</a></body></html>"""
    page = extract_page(html, "https://site.com/dir/")
    assert page.title == "My Page"
    assert page.text == "Hello world n o m"
    assert page.links == ["https://site.com/next", "https://other.org/"]
