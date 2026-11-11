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

