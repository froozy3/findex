"""One polite fetch: timeout, bounded retries with exponential backoff + jitter, `Retry-After`."""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "findex-crawler/0.6 (+https://github.com/your-name/findex; student project)"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.5
BACKOFF_CAP_SECONDS = 30.0
MAX_RETRY_AFTER_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status: int | None
    content: bytes
    content_type: str
    elapsed: float
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is not None and 200 <= self.status < 300


def retry_delay(attempt: int, retry_after: str | None) -> float:
    """Seconds to wait before retry number `attempt` (0-based): honour `Retry-After`, else backoff + jitter."""
    if retry_after is not None and retry_after.strip().isdigit():
        return min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
    return min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2**attempt) * random.uniform(0.5, 1.5)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = MAX_RETRIES,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> FetchResult:
    started = time.perf_counter()
    error = "no attempt made"

    for attempt in range(max_retries + 1):
        retry_after: str | None = None
        try:
            async with asyncio.timeout(timeout):
                response = await client.get(url)
        except httpx.TooManyRedirects:
            return FetchResult(url, url, None, b"", "", time.perf_counter() - started, attempt + 1, "redirect loop")
        except (TimeoutError, httpx.TransportError) as exc:
            error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        else:
            retry_after = response.headers.get("Retry-After")
            if response.status_code not in RETRY_STATUSES:
                failure = None if response.is_success else f"HTTP {response.status_code}"
                return FetchResult(
                    url,
                    str(response.url),
                    response.status_code,
                    response.content,
                    response.headers.get("Content-Type", ""),
                    time.perf_counter() - started,
                    attempt + 1,
                    failure,
                )
            error = f"HTTP {response.status_code}"

        if attempt < max_retries:
            delay = retry_delay(attempt, retry_after)
            log.warning("retry %d/%d for %s in %.1fs (%s)", attempt + 1, max_retries, url, delay, error)
            await sleep(delay)

    return FetchResult(url, url, None, b"", "", time.perf_counter() - started, max_retries + 1, error)
