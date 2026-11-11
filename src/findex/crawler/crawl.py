"""Async crawler: a queue frontier, TaskGroup workers, per-host politeness, async-generator output."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from findex.crawler.extract import extract_page
from findex.crawler.fetch import DEFAULT_USER_AGENT, fetch
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, host_of, is_domain_allowed

log = logging.getLogger(__name__)

MAX_QUEUED_PER_PAGE_BUDGET = 20


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass(slots=True)
class CrawlStats:
    started: float = field(default_factory=time.perf_counter)
    claimed: int = 0
    pages: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def pages_per_second(self) -> float:
        return self.pages / self.elapsed if self.elapsed else 0.0


class Crawler:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        max_pages: int,
        concurrency: int,
        per_host: int,
        allowed_domains: Iterable[str] = (),
        min_delay: float = 0.0,
        user_agent: str = DEFAULT_USER_AGENT,
        stats: CrawlStats | None = None,
    ) -> None:
        self.client = client
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.per_host = per_host
        self.allowed_domains = frozenset(allowed_domains)
        self.min_delay = min_delay
        self.stats = stats or CrawlStats()
        self.robots = RobotsCache(client, user_agent)
        self.frontier: asyncio.Queue[str] = asyncio.Queue()
        self.output: asyncio.Queue[Page | None] = asyncio.Queue()
        self.seen: set[str] = set()
        self.global_slots = asyncio.Semaphore(concurrency)
        self.host_slots: dict[str, asyncio.Semaphore] = {}
        self.host_locks: dict[str, asyncio.Lock] = {}
        self.host_next_start: dict[str, float] = {}

    def enqueue(self, url: str) -> None:
        canonical = canonicalize(url)
        if (
            canonical in self.seen
            or not is_domain_allowed(canonical, self.allowed_domains)
            or len(self.seen) >= self.max_pages * MAX_QUEUED_PER_PAGE_BUDGET
        ):
            return
        self.seen.add(canonical)
        self.frontier.put_nowait(canonical)

    async def wait_for_host_turn(self, url: str, delay: float) -> None:
        host = host_of(url)
        async with self.host_locks.setdefault(host, asyncio.Lock()):
            loop = asyncio.get_running_loop()
            pause = self.host_next_start.get(host, 0.0) - loop.time()
            if pause > 0:
                await asyncio.sleep(pause)
            self.host_next_start[host] = loop.time() + delay

    async def process(self, url: str) -> None:
        if not await self.robots.allowed(url):
            log.info("SKIP robots.txt disallows %s", url)
            return

        delay = max(self.min_delay, await self.robots.crawl_delay(url))
        host_slot = self.host_slots.setdefault(host_of(url), asyncio.Semaphore(self.per_host))
        async with self.global_slots, host_slot:
            await self.wait_for_host_turn(url, delay)
            self.stats.in_flight += 1
            try:
                result = await fetch(self.client, url)
            finally:
                self.stats.in_flight -= 1

        log.info(
            "%s %s %.0fms attempts=%d%s",
            result.status or "ERR",
            url,
            result.elapsed * 1000,
            result.attempts,
            f" ({result.error})" if result.error else "",
        )
        if not result.ok:
            self.stats.errors += 1
            return
        if "html" not in result.content_type.lower():
            return

        self.seen.add(canonicalize(result.final_url))
        extracted = await asyncio.to_thread(extract_page, result.content, result.final_url)
        for link in extracted.links:
            self.enqueue(link)
        if extracted.text:
            self.stats.pages += 1
            await self.output.put(
                Page(result.final_url, extracted.title, extracted.text, datetime.now(UTC).isoformat())
            )

    async def worker(self) -> None:
        while True:
            url = await self.frontier.get()
            try:
                if self.stats.claimed < self.max_pages:
                    self.stats.claimed += 1
                    await self.process(url)
            except Exception:
                log.exception("unexpected error while crawling %s", url)
                self.stats.errors += 1
            finally:
                self.stats.queue_depth = self.frontier.qsize()
                self.frontier.task_done()

    async def finish_when_drained(self) -> None:
        await self.frontier.join()
        await self.output.put(None)

    async def run(self, seeds: Iterable[str]) -> AsyncGenerator[Page]:
        for seed in seeds:
            self.enqueue(seed)

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self.worker()) for _ in range(self.concurrency)]
            tasks.append(group.create_task(self.finish_when_drained()))
            try:
                while (page := await self.output.get()) is not None:
                    yield page
            finally:
                for task in tasks:
                    task.cancel()


async def crawl(
    seeds: Iterable[str],
    *,
    client: httpx.AsyncClient,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    allowed_domains: Iterable[str] = (),
    min_delay: float = 0.0,
    user_agent: str = DEFAULT_USER_AGENT,
    stats: CrawlStats | None = None,
) -> AsyncGenerator[Page]:
    crawler = Crawler(
        client,
        max_pages=max_pages,
        concurrency=concurrency,
        per_host=per_host,
        allowed_domains=allowed_domains,
        min_delay=min_delay,
        user_agent=user_agent,
        stats=stats,
    )
    async for page in crawler.run(seeds):
        yield page


def make_client(user_agent: str = DEFAULT_USER_AGENT, concurrency: int = 10) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": user_agent},
        follow_redirects=True,
        max_redirects=5,
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        timeout=httpx.Timeout(10.0),
    )
