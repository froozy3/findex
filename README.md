# findex — a search engine over your own corpus

[![CI](https://github.com/froozy3/findex/actions/workflows/ci.yml/badge.svg)](https://github.com/froozy3/findex/actions)

Full-text search over ~87 public-domain books (Project Gutenberg, 75 MB, 12 M tokens, split into 10 403 passages).
Streaming indexer → inverted index with positions → BM25/TF-IDF ranking → boolean + phrase query language →
typed, tested CLI → parallel indexing → async crawler → FastAPI web service → NumPy-vectorized scorer.

**Live URL:** _not deployed yet — see "Deploying"_ · **Python** 3.13 · **uv** · **ruff** · **pyright --strict** · **207 tests**

```bash
uv sync
uv run findex index data/gutenberg --out index.pkl --positions
uv run findex search index.pkl 'whale (harpoon OR ahab) NOT pequod "white whale"'
uv run findex serve index.pkl            # http://127.0.0.1:8000  (API docs: /docs)
```

The corpus is downloaded from <https://www.gutenberg.org> into `data/gutenberg/` (git-ignored): 87 books by id,
`https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt`. A passage = ~6 000 characters of consecutive paragraphs.

## Layout

```
src/findex/
  tokenize.py corpus.py stats.py            lab 1   streaming pipeline
  index.py setops.py store.py               lab 2   inverted index, merges, persistence
  scoring.py query.py search.py snippets.py timing.py evaluation.py   lab 3
  cli.py                                    lab 4   typer CLI: index | search | stats | crawl | serve | vectorize
  parallel.py                               lab 5   partial indexes + merge, 3 executors
  crawler/                                  lab 6   fetch, robots, urls, extract, crawl
  web/                                      lab 7   FastAPI app, Jinja2 UI, settings
  vector.py semantic.py                     lab 8   CSR int32 postings + NumPy BM25 + memmap; embeddings, hybrid RRF
scripts/   benchmarks and measurements (everything below was produced by these)
docs/      raw benchmark outputs, plots, crawl log
```

Machine for all numbers: Apple M5, 10 cores, 16 GB, macOS, CPython 3.13.12.

---

## Lab 1 — streams, not lists

`iter_documents` and `tokenize` are generators; `python -m findex.stats data/ [--limit N] [--eager]` runs the pipeline in one pass.
Tokenizer policy (documented in `tokenize.py`, tested): NFC + `casefold`; token = run of letters/digits; an inner apostrophe
or hyphen stays inside the token (`don't`, `well-known`); `_` splits; digits are tokens.

| Version | Documents | Peak memory (tracemalloc) | Elapsed |
|---|---|---|---|
| eager (lists of lists) | 10 403 | 792.2 MB | 11.21 s |
| lazy (generators) | 10 403 | 35.1 MB | 12.07 s |

The eager version keeps the whole token stream alive (12 M `str` objects plus list slots ≈ 790 MB); the lazy one holds one
passage, its token list and the `Counter` (141 k terms) ≈ 35 MB — not zero because of the `Counter` and one passage.

## Lab 2 — the inverted index

`Posting` / `DocMeta` are `@dataclass(frozen=True, slots=True)` (hashable). Index built in one streaming pass with `defaultdict`.
Positions are optional (`--positions`).

**Merge vs set** (µs per operation, median of 200 runs, `scripts/lab_measurements.py merge`):

| Pair | Postings sizes | merge AND | set AND | merge OR | set OR | AND winner |
|---|---|---|---|---|---|---|
| two most common: `in`, `a` | 10078 / 10396 | 622 | 275 | 641 | 294 | set |
| two rarest (df≥5) | 5 / 5 | 0 | 0 | 1 | 0 | tie |
| common + rare: `in`, `labour’s` | 10078 / 5 | 412 | 53 | 473 | 124 | set |

Both are O(n+m), but `set` runs in C while the two-pointer merge is a Python loop, so `set` wins everywhere in CPython.
The merge's real advantage — no hashing, sorted output for free, galloping/skip pointers possible — doesn't show up in pure Python.

**Persistence** (`scripts/lab_measurements.py formats`):

| Format | File size | Save | Load |
|---|---|---|---|
| pickle | 120 MB | 7.4 s | 7.6 s |
| json | 127 MB | 5.9 s | 8.3 s |

`pickle.load` executes arbitrary code embedded in the file (a crafted pickle runs `os.system` via `__reduce__`),
so it must never be used on a file you did not build yourself — see the comment in `store.py`.

**Memory study** (postings only, no positions, tracemalloc):

| Postings representation | Peak build memory | Pickle size | Pickle load |
|---|---|---|---|
| `list[Posting]`, plain `@dataclass` | 510 MB | 95 MB | 1.06 s |
| `list[Posting]`, `slots=True` | 313 MB | 110 MB | 0.90 s |
| `array('I')` pairs, no objects | 99 MB | 46 MB | 0.07 s |

A plain instance carries a ~100-byte `__dict__`; `slots=True` removes it (−39 %). Each 5 M postings then still costs an
object header + two `int` pointers (~56 B); `array('I')` stores 8 bytes per posting with no per-element objects (−81 %).

## Lab 3 — object model and ranking

`Index` is a `Mapping` (`len`, `in`, `[]`, iteration, `repr`, `cached_property avg_doc_length`, `df`), `open_index()` is a
context manager. `Scorer` is a `Protocol` with `TfIdf` and `BM25(k1=1.5, b=0.75)`; top-k via `heapq.nlargest`.
Query language: recursive-descent parser → `Term / Phrase / And / Or / Not` dataclass tree (`&`, `|`, `~` overloaded,
tree-equality tests): `parse('a OR b c') == Or(Term('a'), And(Term('b'), Term('c')))` — `OR` binds looser than the implicit AND.
`@timed` (with `functools.wraps`) logs elapsed ms; `lru_cache` caches query → matching doc ids; snippets highlight matches.

Sanity checks on the real corpus (`scripts/ranking_sanity.py`):

1. Rare beats common: BM25 `score(tf=1)` for `the` (df=10 077) = 0.033, for `ahab` (df=88) = 4.901.
2. Saturation (average-length passage, `whale`): tf 1→2 adds 1.728, tf 20→21 adds 0.031; tf 40 is only 3 % above tf 20.
3. Length normalization: the same single match scores 4.241 in a 1 032-token passage and 2.549 in a 2 657-token passage.

**precision@5** (10 hand-written queries; a hit = the passage comes from a book that is the right answer; `python -m findex.evaluation`):

| query | BM25 | TF-IDF |
|---|---|---|
| white whale Ahab | 1.0 | 1.0 |
| Elizabeth Darcy proposal | 1.0 | 1.0 |
| Sherlock Holmes Watson | 1.0 | 1.0 |
| vampire OR dracula OR transylvania | 1.0 | 1.0 |
| frankenstein OR monster OR creature | 1.0 | 1.0 |
| alice (rabbit OR hatter OR queen) | 1.0 | 1.0 |
| raskolnikov murder pawnbroker | 0.8 | 0.8 |
| treasure OR pirates OR silver | 0.8 | 0.8 |
| natural selection variation species | 1.0 | 1.0 |
| odysseus OR suitors OR penelope | 1.0 | 1.0 |
| **mean** | **0.96** | **0.96** |

BM25 and TF-IDF tie on this set: the queries are distinctive names, which both rank well. The set is small and labels are at book
level; it cannot separate the scorers. (My first version of the queries used 5 plain words and scored P@5 = 0 for four of them — implicit
AND over five words rarely co-occurs in one 1 000-word passage — which is why the final queries use explicit `OR`.)

## Lab 4 — a real tool

`pyright --strict`: 0 errors (one test file, `tests/test_web.py`, is checked in basic mode because Starlette's `TestClient`
is only partially typed). `ruff check` + `ruff format --check` clean. **207 tests** (pytest, parametrized, 3+ Hypothesis properties on
the merges, CLI via `CliRunner`), **93 % coverage**; uncovered: `__main__`, the `argparse` entry of `stats`/`evaluation`, a few
defensive `except` branches in the crawler/web layers and the `--limit` progress path.

CLI: `findex index|search|stats|crawl|serve|vectorize|embed`; `-v/-vv` set the log level, diagnostics go to stderr through `logging`;
`search --json` emits clean JSON lines on stdout; expected errors exit with code 1 and one line. `uv build` produces the wheel
(`dist/findex-1.0.0-py3-none-any.whl`); `uv tool install .` puts `findex` on PATH. CI: `.github/workflows/ci.yml` (ruff, pyright, pytest).

## Lab 5 — concurrency and the GIL

`build_partial(paths) -> PartialIndex` (picklable, module-level) + `merge(partials)`; the serial build is `merge([build_partial(all)])`,
so every executor produces the same index (tested via canonical JSON — raw pickle bytes differ because pickle memoizes by object identity).
`findex index --executor {serial,threads,processes} --workers N`. Full corpus, no positions, 3 runs each, median
(`scripts/parallel_bench.py`, plot: `docs/parallel_bench.png`):

| Executor | Workers | Wall (s) | CPU (s) | Peak RSS parent / worker (MB) | Merge (s) | Speedup |
|---|---|---|---|---|---|---|
| serial | 1 | 6.8 | 6.8 | 986 / – | 3.3 | 1.00× |
| threads | 2 | 8.1 | 7.9 | 974 / – | 4.0 | 0.84× |
| threads | 4 | 7.8 | 7.8 | 980 / – | 3.9 | 0.86× |
| threads | 8 | 15.7 | 14.9 | 1001 / – | 4.5 | 0.43× |
| threads | 10 | 8.4 | 8.4 | 1000 / – | 4.3 | 0.81× |
| processes | 2 | 4.6 | 6.9 | 1139 / 220 | 2.3 | 1.47× |
| processes | 4 | 3.9 | 7.7 | 1090 / 156 | 2.5 | 1.75× |
| processes | 8 | 3.6 | 9.5 | 1066 / 141 | 2.6 | 1.87× |
| processes | 10 | 3.6 | 9.8 | 1056 / 140 | 2.7 | 1.89× |

Free-threaded CPython 3.13t (GIL off, `docs/parallel_bench_freethreaded.md`), threads executor:

| Executor | Workers | Wall (s) | CPU (s) | Speedup vs its own serial |
|---|---|---|---|---|
| serial | 1 | 13.2 | 13.2 | 1.00× (but 1.9× slower than the regular build's 6.8 s) |
| threads | 2 / 4 / 8 / 10 | 10.2 / 8.6 / 8.6 / 8.5 | 12.1 / 12.9 / 25.5 / 30.0 | 1.29× / 1.54× / 1.52× / 1.55× |

I/O-bound counter-benchmark (50 requests to a local server that answers after 100 ms; `docs/io_bound.md`): threads 1/2/4/8/16/32 →
5.25 / 2.63 / 1.39 / 0.74 / 0.42 / 0.22 s, i.e. 1× → 24× — near-linear, because threads sleep in `recv`, which releases the GIL.

**Explanation.** Tokenizing and counting are pure-Python CPU work: under the GIL only one thread executes bytecode at a time, so
threads give no speedup (0.84× at 2 threads — switching overhead; the 8-thread row, 0.43×, is lock convoy/contention, and varies from run to run).
Processes bypass the GIL and reach 1.9×, but at a price visible in the table: CPU time rises from 6.8 s to 9.8 s (spawning interpreters,
re-importing, pickling partial indexes back) and every worker holds its own ~140–220 MB. The ceiling is **Amdahl's law**: `merge` is serial
and takes ~2.6 s of the 3.6 s wall time (≈ 72 %), so even infinite workers on the parallel ~1 s part cannot beat ~2.5×; we get 1.89× on 10 cores.
On the free-threaded build threads finally scale (1.55×) but the single-thread baseline is ~1.9× slower (per-object atomic refcounting/locks),
CPU time grows with workers (30 s at 10 threads) because of contention, and the merge phase is still serial. No race appeared because
each worker builds its own private `PartialIndex`; the only shared mutable state would have been a shared `Counter`, which is why it is not shared.
`--executor processes` is the CLI default, and the entry point is `spawn`-safe (`__main__.py` has the `if __name__ == "__main__"` guard).

## Lab 6 — async crawler

`crawler/fetch.py`: `asyncio.timeout`, ≤ 3 retries for 429/5xx/timeouts with exponential backoff + jitter, `Retry-After` honoured,
no retry for other 4xx, redirect loops detected. `robots.py`: `RobotsCache` (one download per host; 404 → allow, 5xx/unreachable → disallow;
`Crawl-delay` respected). `crawl.py`: `asyncio.Queue` frontier, `TaskGroup` workers, global + per-host semaphores, per-host minimum delay,
URL canonicalization + `seen` set, HTML parsing off the loop via `asyncio.to_thread`, async-generator output, clean cancellation.
`findex crawl <seed> --max-pages --concurrency --per-host --delay --out data/crawl.jsonl` streams one JSON line per page (consumed by
`findex index data/crawl.jsonl`), shows a live `rich` table, writes `crawl.log`. Tests use `httpx.MockTransport` (no network).

**Ethics / permission.** The benchmark crawled a **local mirror I run myself** (`scripts/mirror_site.py`, 600 generated pages, 80 ms latency) — no third
party was contacted. Set `--contact` to your own URL/e-mail before crawling anything real, and crawl only sites you have permission for.

200-page budget (199 pages fetched: one claimed slot went to the `robots.txt`-disallowed link), `scripts/crawl_bench.sh`:

| Concurrency | per-host | delay | Pages | Wall (s) | Pages/s | Errors | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0 | 199 | 17.6 | 11.3 | 0 | 57 |
| 5 | 5 | 0 | 199 | 3.8 | 52.4 | 0 | 49 |
| 20 | 20 | 0 | 199 | 1.4 | 142.1 | 0 | 50 |
| 20 | 2 | 0.1 s | 199 | 20.1 | 9.9 | 0 | 50 |

Why one thread with 20 tasks wins: every request is ~80 ms of waiting, so a single event loop overlaps 20 waits with no thread/GIL switching
and a flat ~50 MB (Lab 5's processes would cost ~150 MB each and pickling). The last row is what politeness costs: with 2 requests per host and
0.1 s between them the crawl falls back to ~10 pages/s no matter how many workers exist — I kept those limits as the *default* for real sites
(I only lifted them for my own mirror). Indexing is outside the loop (a separate `findex index` step), and HTML parsing is moved to a thread so the
loop never runs CPU-heavy code. Verified with `asyncio.run(..., debug=True)` and `slow_callback_duration = 0.1`
(`scripts/crawl_debug_check.py`, `docs/crawl_debug_check.txt`): 99 pages crawled, no slow-callback warnings, i.e. no blocking call in a coroutine. Failure handling, from `docs/crawl_faults.log` (mirror started with `--faults`):

```
retry 1/3 for /limited in 1.0s (HTTP 429)          <- Retry-After: 1 honoured
retry 1/3 for /flaky in 0.3s (HTTP 503)
retry 2/3 for /flaky in 1.0s (HTTP 503)
200 /flaky 1351ms attempts=3
200 /limited 1012ms attempts=2
ERR /loop 8ms attempts=1 (redirect loop)
SKIP robots.txt disallows /private/secret
retry 1/3 .. 3/3 for /slow (TimeoutError)  ->  ERR /slow 43070ms attempts=4
```

