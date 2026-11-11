
### findex index (full corpus, no positions)
| function | calls | own time (s) | cumulative (s) |
|---|---|---|---|
| index.py:141 `<genexpr>` | 5049919 | 2.94 | 3.73 |
| tokenize.py:29 `tokenize` | 12063626 | 2.70 | 4.05 |
| index.py:97 `build_partial_from_documents` | 1 | 1.26 | 6.36 |
| ~:0 `<method 'group' of 're.Match' objects>` | 12053223 | 0.72 | 0.72 |
| ~:0 `<method 'casefold' of 'str' objects>` | 12053223 | 0.59 | 0.59 |
| ~:0 `<built-in method _collections._count_elements>` | 10403 | 0.40 | 0.40 |
| ~:0 `<method 'append' of 'list' objects>` | 5188656 | 0.31 | 0.31 |
| ~:0 `<method 'extend' of 'list' objects>` | 141262 | 0.24 | 3.97 |

### search 'white whale captain' x300 (no snippets)
| function | calls | own time (s) | cumulative (s) |
|---|---|---|---|
| search.py:38 `score_documents` | 300 | 0.03 | 0.07 |
| scoring.py:24 `score` | 38850 | 0.02 | 0.03 |
| index.py:90 `df` | 38850 | 0.01 | 0.01 |
| ~:0 `<built-in method builtins.len>` | 112000 | 0.00 | 0.00 |
| index.py:79 `num_docs` | 38852 | 0.00 | 0.00 |
| ~:0 `<method 'get' of 'dict' objects>` | 39908 | 0.00 | 0.00 |
| heapq.py:523 `nlargest` | 300 | 0.00 | 0.00 |
| index.py:87 `doc_length` | 38850 | 0.00 | 0.00 |

### search 'white whale' x300 with snippets
| function | calls | own time (s) | cumulative (s) |
|---|---|---|---|
| snippets.py:28 `find_term_spans` | 3000 | 0.27 | 0.35 |
| ~:0 `<method 'casefold' of 'str' objects>` | 4202 | 0.04 | 0.04 |
| ~:0 `<method 'lower' of 'str' objects>` | 3000 | 0.03 | 0.03 |
| scoring.py:24 `score` | 54600 | 0.02 | 0.05 |
| search.py:38 `score_documents` | 300 | 0.01 | 0.07 |
| ~:0 `<built-in method unicodedata.normalize>` | 4203 | 0.01 | 0.01 |
| snippets.py:42 `make_snippet` | 3000 | 0.01 | 0.39 |
| snippets.py:51 `<lambda>` | 55800 | 0.01 | 0.02 |

### Speedup (median of 200 runs)
| Query | Docs matched | Python scorer (ms) | NumPy scorer (ms) | Speedup |
|---|---|---|---|---|
| common term (`the`) | 10077 | 3.04 | 0.12 | 24.9x |
| rare term (`ahab`) | 88 | 0.04 | 0.02 | 1.7x |
| 3-term query (`white whale captain`) | 37 | 0.24 | 0.08 | 3.1x |

rankings differing from the Python engine on the 10 evaluation queries: 0
