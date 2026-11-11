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

