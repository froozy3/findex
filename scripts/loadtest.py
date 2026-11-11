"""Tiny HTTP load tester (rps, p50/p95/p99):  uv run python scripts/loadtest.py URL --concurrency 50 --requests 2000"""

import argparse
import asyncio
import itertools
import statistics
import time

import httpx

QUERIES = [
    "white whale",
    "sherlock holmes",
    "love OR marriage",
    "ship captain sea",
    "dracula OR vampire",
    '"my dear"',
    "king NOT queen",
    "monster laboratory",
    "treasure island",
    "natural selection",
]


async def run(base_url: str, route: str, concurrency: int, total_requests: int) -> None:
    queries = itertools.cycle(QUERIES)
    latencies: list[float] = []
    failures = 0

    async def worker(client: httpx.AsyncClient, budget: list[int]) -> None:
        nonlocal failures
        while budget[0] > 0:
            budget[0] -= 1
            started = time.perf_counter()
            response = await client.get(route, params={"q": next(queries)})
            latencies.append((time.perf_counter() - started) * 1000)
            failures += response.status_code != 200

    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, limits=limits, timeout=60) as client:
        await client.get(route, params={"q": "warm up"})
        started = time.perf_counter()
        budget = [total_requests]
        await asyncio.gather(*(worker(client, budget) for _ in range(concurrency)))
        wall = time.perf_counter() - started

    cuts = statistics.quantiles(latencies, n=100)
    print(
        f"| {route} | {concurrency} | {len(latencies) / wall:.0f} | {cuts[49]:.0f} | {cuts[94]:.0f} | {cuts[98]:.0f} "
        f"| {failures} |"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--route", default="/search")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--requests", type=int, default=2000)
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.route, args.concurrency, args.requests))
