machine: {'machine': 'macOS-27.0.1-arm64-arm-64bit-Mach-O', 'cpus': 10, 'python': '3.13.12', 'gil_enabled': False}
| Executor | Workers | Wall (s, median) | CPU (s) | Peak RSS parent / worker (MB) | Merge (s) | Speedup |
|---|---|---|---|---|---|---|
| serial | 1 | 13.2 | 13.2 | 1015 / 0 | 3.0 | 1.00x |
| threads | 2 | 10.2 | 12.1 | 1054 / 0 | 3.0 | 1.29x |
| threads | 4 | 8.6 | 12.9 | 1071 / 0 | 2.7 | 1.54x |
| threads | 8 | 8.6 | 25.5 | 1090 / 0 | 2.7 | 1.52x |
| threads | 10 | 8.5 | 30.0 | 1098 / 0 | 2.7 | 1.55x |
plot: docs/parallel_bench_freethreaded.png
