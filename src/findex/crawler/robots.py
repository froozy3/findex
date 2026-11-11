import asyncio
import logging
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.urls import host_of

log = logging.getLogger(__name__)


class RobotsCache:
    """Fetches and caches `robots.txt` once per host."""

    def __init__(self, client: httpx.AsyncClient, user_agent: str) -> None:
        self.client = client
        self.user_agent = user_agent
        self.parsers: dict[str, RobotFileParser] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def parser_for(self, url: str) -> RobotFileParser:
        host = host_of(url)
        scheme = url.split(":", 1)[0]
        async with self.locks.setdefault(host, asyncio.Lock()):
            if host not in self.parsers:
                self.parsers[host] = await self.download(f"{scheme}://{host}/robots.txt")
        return self.parsers[host]

    async def download(self, robots_url: str) -> RobotFileParser:
        parser = RobotFileParser()
        try:
            response = await self.client.get(robots_url)
        except httpx.HTTPError as error:
            log.warning("robots.txt unreachable (%s): treating the host as disallowed", error)
            parser.parse(["User-agent: *", "Disallow: /"])
            return parser
        if response.status_code >= 500:
            parser.parse(["User-agent: *", "Disallow: /"])
        elif response.status_code >= 400:
            parser.parse([])
        else:
            parser.parse(response.text.splitlines())
        return parser

    async def allowed(self, url: str) -> bool:
        return (await self.parser_for(url)).can_fetch(self.user_agent, url)

    async def crawl_delay(self, url: str) -> float:
        delay = (await self.parser_for(url)).crawl_delay(self.user_agent)
        return float(delay) if delay is not None else 0.0
