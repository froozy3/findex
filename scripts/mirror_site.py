"""A local mirror site to crawl politely (no third-party load): `uv run python scripts/mirror_site.py --port 8765`.

Routes: /page/N (linked pages with fixed latency), /robots.txt (disallows /private/),
and, with --faults: /flaky (503 twice), /limited (429 + Retry-After), /loop (redirect loop), /slow (timeout).
"""

import argparse
import random
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

WORDS = [
    "index",
    "search",
    "engine",
    "query",
    "ranking",
    "document",
    "term",
    "frequency",
    "inverse",
    "corpus",
    "token",
    "stream",
    "generator",
    "python",
    "thread",
    "process",
    "async",
    "event",
    "loop",
    "network",
    "request",
    "response",
    "crawler",
    "robot",
    "polite",
    "delay",
    "queue",
    "merge",
    "posting",
    "list",
    "memory",
    "array",
    "vector",
    "score",
    "model",
    "cache",
    "latency",
    "throughput",
    "benchmark",
    "profile",
]


class ReliableServer(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


class MirrorHandler(BaseHTTPRequestHandler):
    page_count = 600
    latency = 0.08
    with_faults = False
    hits: ClassVar[Counter[str]] = Counter()
    lock = threading.Lock()

    def reply(self, status: int, body: str = "", headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body.encode())

    def page_html(self, number: int) -> str:
        generator = random.Random(number)
        text = " ".join(generator.choices(WORDS, k=80))
        targets = {
            (number + 1) % self.page_count,
            generator.randrange(self.page_count),
            generator.randrange(self.page_count),
        }
        links = "".join(f'<a href="/page/{target}">page {target}</a> ' for target in targets)
        head = f"<title>Mirror page {number}</title>"
        return f"<html><head>{head}</head><body><h1>Page {number}</h1><p>{text}</p>{links}</body></html>"

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        with self.lock:
            self.hits[path] += 1
            hit_number = self.hits[path]
        if path == "/robots.txt":
            self.reply(200, "User-agent: *\nDisallow: /private/\n")
        elif path == "/":
            faults = '<a href="/flaky">f</a> <a href="/limited">l</a> <a href="/loop">r</a> <a href="/slow">s</a>'
            self.reply(
                200,
                '<title>Home</title><a href="/page/0">start</a> <a href="/private/secret">no</a> '
                + (faults if self.with_faults else ""),
            )
        elif path.startswith("/page/"):
            time.sleep(self.latency)
            self.reply(200, self.page_html(int(path.rsplit("/", 1)[1]) % self.page_count))
        elif path.startswith("/private/"):
            self.reply(200, "<title>private</title>secret")
        elif path == "/flaky" and self.with_faults:
            self.reply(503) if hit_number <= 2 else self.reply(200, "<title>Flaky</title>finally worked")
        elif path == "/limited" and self.with_faults:
            self.reply(429, headers={"Retry-After": "1"}) if hit_number == 1 else self.reply(
                200, "<title>Limited</title>ok now"
            )
        elif path == "/loop" and self.with_faults:
            self.reply(302, headers={"Location": "/loop"})
        elif path == "/slow" and self.with_faults:
            time.sleep(12)
            self.reply(200, "<title>Slow</title>too late")
        else:
            self.reply(404, "not found")

    def log_message(self, format: str, *args: Any) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pages", type=int, default=600)
    parser.add_argument("--latency", type=float, default=0.08)
    parser.add_argument("--faults", action="store_true")
    args = parser.parse_args()
    MirrorHandler.page_count, MirrorHandler.latency, MirrorHandler.with_faults = args.pages, args.latency, args.faults
    print(f"mirror site on http://127.0.0.1:{args.port}/", flush=True)
    ReliableServer(("127.0.0.1", args.port), MirrorHandler).serve_forever()
