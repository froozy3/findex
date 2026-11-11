| Version | Documents | Peak memory | Elapsed |
|---|---|---|---|
| lazy | 10403 | 35.1 MB | 12.07s |
| eager | 10403 | 792.2 MB | 11.21s |
| Postings representation | Peak memory, build (MB) | Pickle size (MB) | Pickle load (s) |
|---|---|---|---|
| `list[Posting]`, plain `@dataclass` | 510 | 95 | 1.06 |
| `list[Posting]`, `slots=True` | 313 | 110 | 0.90 |
| `array('I')` pairs, no objects | 99 | 46 | 0.07 |
