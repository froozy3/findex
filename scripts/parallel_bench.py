# matplotlib ships no type information
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Benchmark of parallel indexing (lab 5).

uv run python scripts/parallel_bench.py run data/gutenberg --runs 3 --out docs/parallel_bench.json
uv run python scripts/parallel_bench.py report docs/parallel_bench.json
uv run python scripts/parallel_bench.py io-bound
"""

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.request import urlopen

from findex.parallel import ExecutorKind, build_parallel

BYTES_PER_MB = 1024 * 1024 if sys.platform == "darwin" else 1024  # ru_maxrss is bytes on macOS, KiB on Linux
WORKER_COUNTS = [1, 2, 4, 8, os.cpu_count() or 1]


def run_single(corpus: Path, kind: ExecutorKind, workers: int) -> dict[str, Any]:
    """Executed in a fresh interpreter so peak RSS belongs to this configuration only."""
    started = time.perf_counter()
    _, merge_seconds = build_parallel(corpus, kind=kind, workers=workers)
    wall = time.perf_counter() - started
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "executor": kind.value,
        "gil_enabled": sys._is_gil_enabled(),  # pyright: ignore[reportPrivateUsage]
        "workers": workers,
        "wall": wall,
        "cpu": own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime,
        "parent_rss_mb": own.ru_maxrss / BYTES_PER_MB,
        "child_rss_mb": children.ru_maxrss / BYTES_PER_MB,
        "merge": merge_seconds,
    }


def run_all(corpus: Path, runs: int, out: Path, executors: list[ExecutorKind]) -> None:
    configurations = [(ExecutorKind.SERIAL, 1)]
    for kind in executors:
        configurations += [(kind, workers) for workers in dict.fromkeys(WORKER_COUNTS) if workers > 1]

    results: list[dict[str, Any]] = []
    for kind, workers in configurations:
        for run_number in range(runs):
            completed = subprocess.run(
                [sys.executable, __file__, "single", str(corpus), kind.value, str(workers)],
                capture_output=True,
                text=True,
                check=True,
            )
            result = json.loads(completed.stdout.splitlines()[-1])
            results.append(result)
            print(f"{kind.value:9s} w={workers:2d} run {run_number + 1}: wall {result['wall']:.1f}s", flush=True)
    machine = {
        "machine": platform.platform(),
        "cpus": os.cpu_count(),
        "python": sys.version.split()[0],
        "gil_enabled": sys._is_gil_enabled(),  # pyright: ignore[reportPrivateUsage]
    }
    out.write_text(json.dumps({"machine": machine, "results": results}, indent=2))


def report(path: Path) -> None:
    data = json.loads(path.read_text())
    rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for result in data["results"]:
        rows.setdefault((result["executor"], result["workers"]), []).append(result)

    baseline = statistics.median(r["wall"] for r in rows[("serial", 1)])
    print(f"machine: {data['machine']}")
    print("| Executor | Workers | Wall (s, median) | CPU (s) | Peak RSS parent / worker (MB) | Merge (s) | Speedup |")
    print("|---|---|---|---|---|---|---|")
    for (executor, workers), group in rows.items():
        wall = statistics.median(r["wall"] for r in group)
        cpu = statistics.median(r["cpu"] for r in group)
        parent = statistics.median(r["parent_rss_mb"] for r in group)
        child = statistics.median(r["child_rss_mb"] for r in group)
        merge = statistics.median(r["merge"] for r in group)
        memory = f"{parent:.0f} / {child:.0f}"
        print(f"| {executor} | {workers} | {wall:.1f} | {cpu:.1f} | {memory} | {merge:.1f} | {baseline / wall:.2f}x |")
    plot(rows, baseline, path.with_suffix(".png"))


def plot(rows: dict[tuple[str, int], list[dict[str, Any]]], baseline: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 4))
    for executor in ("threads", "processes"):
        points = sorted(
            (w, baseline / statistics.median(r["wall"] for r in g)) for (e, w), g in rows.items() if e == executor
        )
        axes.plot(*zip(*points, strict=True), marker="o", label=executor)
    max_workers = max(w for _, w in rows)
    axes.plot([1, max_workers], [1, max_workers], "k--", label="ideal linear")
    axes.set_xlabel("workers")
    axes.set_ylabel("speedup vs serial")
    axes.set_title("Indexing speedup")
    axes.legend()
    figure.tight_layout()
    figure.savefig(out, dpi=140)
    print(f"plot: {out}")


class ReliableServer(ThreadingHTTPServer):
    request_queue_size = 128  # the default of 5 drops connections under a burst


class SlowHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        time.sleep(0.1)  # network latency
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: Any) -> None:
        pass


def io_bound() -> None:
    """Fetch 50 URLs from a local server that answers after 100 ms: threads scale because they wait, not compute."""
    server = ReliableServer(("127.0.0.1", 0), SlowHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"

    def fetch(_: int) -> int:
        with urlopen(url) as response:
            return len(response.read())

    print("| Threads | Wall for 50 requests (s) | Speedup |")
    print("|---|---|---|")
    baseline = 0.0
    for workers in (1, 2, 4, 8, 16, 32):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, range(50)))
        wall = time.perf_counter() - started
        baseline = baseline or wall
        print(f"| {workers} | {wall:.2f} | {baseline / wall:.1f}x |")
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("corpus", type=Path)
    run_parser.add_argument("--runs", type=int, default=3)
    run_parser.add_argument("--out", type=Path, default=Path("docs/parallel_bench.json"))
    run_parser.add_argument(
        "--executors", nargs="+", type=ExecutorKind, default=[ExecutorKind.THREADS, ExecutorKind.PROCESSES]
    )
    single_parser = sub.add_parser("single")
    single_parser.add_argument("corpus", type=Path)
    single_parser.add_argument("kind", type=ExecutorKind)
    single_parser.add_argument("workers", type=int)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("path", type=Path)
    sub.add_parser("io-bound")
    args = parser.parse_args()

    if args.command == "single":
        print(json.dumps(run_single(args.corpus, args.kind, args.workers)))
    elif args.command == "run":
        run_all(args.corpus, args.runs, args.out, args.executors)
    elif args.command == "report":
        report(args.path)
    else:
        io_bound()
