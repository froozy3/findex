"""Run the crawler with asyncio debug mode on and report any coroutine step slower than 100 ms (a blocking call).

uv run python scripts/mirror_site.py --port 8765 &   uv run python scripts/crawl_debug_check.py 8765
"""

import asyncio
import logging
import sys

from findex.crawler.crawl import crawl, make_client


async def main(port: int) -> None:
    asyncio.get_running_loop().slow_callback_duration = 0.1
    async with make_client(concurrency=10) as client:
        pages = [
            page
            async for page in crawl(
                [f"http://127.0.0.1:{port}/"], client=client, max_pages=100, concurrency=10, per_host=10
            )
        ]
    print(f"{len(pages)} pages crawled with debug=True")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(int(sys.argv[1])), debug=True)
