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

## Lab 7 — search on the web

`findex serve index.pkl [--port] [--workers]` runs FastAPI: `GET /search?q&k&scorer&page` (validated, paginated), `GET /docs/{id}`,
`GET /stats`, `GET /health` (503 until the index is loaded), Swagger at `/docs`, HTML UI at `/` and `/doc/{id}` (Jinja2, highlighted snippets,
pagination, light/dark). Index loaded once in `lifespan`; configuration via `pydantic-settings` (`INDEX_PATH`, …); Pydantic models at the edge,
dataclasses inside; request-id middleware and a clean JSON 500. Tests use `TestClient` + `dependency_overrides` (200/422/404/503/500 paths).
Load tests with `scripts/loadtest.py` (own async load generator — `oha` is not installed here), local machine, 1 500–3 000 requests over 10 mixed queries:

| Target | Route | Concurrency | RPS | p50 (ms) | p95 | p99 |
|---|---|---|---|---|---|---|
| local, 1 worker, `async def` | /search-async | 50 | 64 / 63 | 783 / 794 | 876 / 868 | 1114 / 1081 |
| local, 1 worker, `def` | /search | 50 | 56 / 56 | 868 / 870 | 1200 / 1194 | 1361 / 1410 |
| local, 1 worker, `def` | /search | 20 | 54 | 356 | 554 | 634 |
| local, 4 workers, `def` | /search | 50 | 199 / 201 | 227 / 234 | 467 / 437 | 568 / 516 |
| deployed | – | – | not measured (not deployed) | | | | |

Reading the numbers with Labs 5–6 in mind: the work per request is CPU-bound Python (scoring + snippets), so with one process the GIL caps
throughput at ~55–65 RPS regardless of concurrency — latency just grows with the queue (p50 356 ms at 20 clients vs 868 ms at 50, same RPS).
`async def` is ~13 % faster than `def` here because `def` pays a thread-pool hop and GIL contention for no parallelism; I still keep **`def`**
for `/search`: it runs in the thread pool so `/health` and other requests are not blocked while one search runs (an inline `async def` would stall the
whole event loop), at a cost of ~13 % throughput. Four worker processes scale to ~200 RPS (3.5×, the same lesson as the processes row of Lab 5).
The profile in Lab 8 shows why 56 RPS is slow (≈ 15 ms/request): it is snippet generation, not ranking.

**Container.** The multi-stage `Dockerfile` (uv, non-root `uid 1000`, index built inside the image, `HEALTHCHECK`) was built and run locally:
643 MB image, `/health` → `ok`, `/search` answers, `id` inside = `app`. Through Docker (1 worker, 20 clients, 600 requests):
45 RPS, p50 419 ms, p95 731 ms, p99 842 ms — slightly below the bare-metal 54 RPS because of Docker Desktop's VM and port forwarding.

**Cloudflare (not live yet).** `wrangler.jsonc` + `cloudflare/index.ts` define a Cloudflare Containers deployment (`npm install && npx wrangler login && npx wrangler deploy`;
Containers need the Workers Paid plan). I tried `wrangler deploy --temporary` (preview account without login): the Worker uploaded but the Containers API
rejected the account, so nothing is live. A free alternative is a Cloudflare Tunnel to a locally running container
(`docker run -p 8001:8000 findex:1.0.0` + `cloudflared tunnel --url http://localhost:8001`), but that publishes this machine and was not started without your say-so.
Once a URL exists, paste it at the top of this README and run `scripts/loadtest.py <url> --concurrency 20` for the "deployed" row.

## Lab 8 — fast, then smart

**Profile** (`scripts/profile_hotspots.py`, `docs/profile_report.md`, cProfile; py-spy/Scalene were not run — py-spy needs `sudo` on macOS. Flame graphs are rendered
from the cProfile data with `flameprof`: `docs/flame_index.svg` (build) and `docs/flame_search.svg` (search with snippets, 300 requests, before the snippet fix is *not* in
the graph — it shows the optimized state). For a native-vs-Python split run `sudo py-spy record -o docs/flame.svg -- uv run findex search index.pkl "white whale"`).
Hypotheses written before measuring: (1) indexing is dominated by the tokenizer regex + `casefold` per token; (2) `search` is dominated by the per-posting
Python call to `Scorer.score`; (3) snippets are cheap.

| Command | Measured top hotspots | Hypothesis |
|---|---|---|
| `findex index` (6.4 s) | `tokenize` generator 4.0 s cum. (12 M iterations), `Posting` generator in `merge` 3.7 s cum., `build_partial` body 1.3 s | (1) confirmed: Python-time, ~63 % in the per-token loop |
| `findex search` "white whale captain" ×300, no snippets | `score_documents` 0.07 s total, `Scorer.score` 38 850 calls | (2) wrong: whole search ≈ 0.2 ms — scoring is not the problem for rare terms |
| `findex search` "white whale" ×300 with snippets | `iter_token_spans` 1.96 s cum. (3.5 M tokens regex-scanned) of 2.25 s | (3) wrong: snippets are 97 % of request time |

**Fix 1 — snippets:** find matches with one compiled regex over the passage instead of tokenizing all of it (equivalence with the tokenizing
reference is tested incl. Cyrillic, combining accents, `ß` → exact fallback path): 2.25 s → 0.39 s per 300 requests (**5.8×**).

**Fix 2 — vectorized scorer** (`findex vectorize index.pkl --out data/vectors`, `findex search data/vectors "..."`): postings in CSR layout
(`int32` doc_ids / tfs / offsets, `doc_len` `int32[N]`), BM25 and TF-IDF as array expressions, boolean evaluation via `np.intersect1d/union1d/setdiff1d`,
top-k via `np.partition` + `np.lexsort` (higher score first, lower doc_id on ties). Rankings are identical to the Python engine (parametrized test,
plus 0 differing rankings on the 10 evaluation queries). Phrase queries are not supported by this engine (no positions).

| Query | Docs matched | Python scorer (ms) | NumPy scorer (ms) | Speedup |
|---|---|---|---|---|
| common term `the` | 10 077 | 3.04 | 0.12 | 24.9× |
| rare term `ahab` | 88 | 0.04 | 0.02 | 1.7× |
| 3-term `white whale captain` | 37 | 0.24 | 0.08 | 3.1× |

The speedup scales with the postings-list length: for 10 k postings, NumPy replaces 10 k Python-level `score()` calls with one array expression;
for 88 postings the cost is dominated by Python call overhead on both sides, and on the 5-document test fixture NumPy is actually ~2.4× *slower*
(`tests/test_benchmarks.py`: 31 µs vs 13 µs). `pytest-benchmark` tests with generous ceilings run in CI.

**Memmap.** The vectorized index is 41 MB on disk vs 128 MB pickle and loads in 0.02 s vs 4.85 s (the arrays are `np.load(mmap_mode="r")`;
at 41 MB the memmap vs read-into-RAM difference is below the noise: 0.021 s vs 0.019 s).

**M3 — semantic and hybrid search** (`uv sync --extra semantic`; `findex embed index.pkl --out data/embeddings`, then `findex search index.pkl "..." --mode semantic|hybrid`;
API `GET /search?mode=...` and a mode selector in the UI when `EMBEDDINGS_PATH` is set). Model `all-MiniLM-L6-v2`; every passage is split into ~900-character windows
(55 523 windows × 384 dims, 85 MB, 2 min 18 s on this machine), vectors L2-normalized, `E @ q` → top 200 windows → best window per document. Hybrid = reciprocal rank fusion (k = 60)
of the BM25 ranking (a plain-words query is turned into `a OR b OR c`; queries with operators are used as written) and the semantic ranking.
precision@5 on the 10 original + 5 paraphrase queries (`scripts/eval_modes.py`, `docs/eval_modes.md`; paraphrases in `eval/paraphrase_queries.json`):

| group | keyword | semantic | hybrid |
|---|---|---|---|
| 10 original queries (names, explicit `OR`) | **0.96** | 0.92 | **0.98** |
| 5 paraphrases (no character names) | 0.00 | **0.44** | 0.36 |
| all 15 | 0.64 | 0.76 | **0.77** |

Interpretation: on distinctive keywords BM25 is already near-perfect and the neural model slightly hurts alone (`vampire OR dracula…` 0.6, `frankenstein OR monster…` 0.8) —
hybrid keeps BM25's precision and fixes its two misses (`raskolnikov…` 0.8 → 1.0). On paraphrases keyword search returns nothing useful (0.00): there is no shared vocabulary
with the book, and implicit AND over a whole sentence matches nothing. Semantic finds the right book for 2 of 5 (`sea captain…whale` 1.0, `detective…` 1.0) and fails on the
two classic romances/fantasies (`young woman rejects a proud rich man` and `girl follows a talking animal` score 0.0 — passages describing those scenes are
dominated by generic dialogue and the 6 000-character windows of other books are closer). Hybrid is slightly *below* semantic on paraphrases (0.36 vs 0.44) because the BM25 side
of an OR-query over common words adds noise that RRF only partly outvotes. Five paraphrase queries are a tiny, noisy sample; the direction (semantic ≫ keyword on paraphrases, hybrid best overall) is
the useful conclusion, the decimals are not. The semantic extras are optional (torch is not in the Docker image), so the deployed service is keyword-only.

**Still to do for the showcase (needs you):** deployed URL (see Lab 7), GIF and 2–3 min demo video, `v1.0.0` tag and release (see below).

---

## Git, tags, release (not done — needs your approval)

No commit, tag or push has been made. Suggested:

```bash
git add -A && git commit -m "findex: labs 1-8"
# or per lab: git tag lab-01 … lab-08, v0.4.0 (lab 4), v1.0.0
gh release create v1.0.0 dist/findex-1.0.0-py3-none-any.whl
```

## Reproducing the measurements

```bash
uv run python scripts/lab_measurements.py lab1 data/gutenberg     # eager vs lazy
uv run python scripts/lab_measurements.py memory data/gutenberg   # posting representations
uv run python scripts/lab_measurements.py merge index.pkl         # merge vs set
uv run python scripts/lab_measurements.py formats index.pkl       # pickle vs json
uv run python scripts/ranking_sanity.py index.pkl                 # three ranking checks
uv run python -m findex.evaluation index.pkl eval/queries.json    # precision@5
uv run python scripts/parallel_bench.py run data/gutenberg        # + report / io-bound
uv run python scripts/mirror_site.py & scripts/crawl_bench.sh     # crawler
uv run python scripts/loadtest.py http://127.0.0.1:8000 --route /search
uv run python scripts/profile_hotspots.py index.pkl data/gutenberg
```
