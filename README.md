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

