machine: {'machine': 'macOS-27.0.1-arm64-arm-64bit-Mach-O', 'cpus': 10, 'python': '3.13.12'}
| Executor | Workers | Wall (s, median) | CPU (s) | Peak RSS parent / worker (MB) | Merge (s) | Speedup |
|---|---|---|---|---|---|---|
| serial | 1 | 6.8 | 6.8 | 986 / 0 | 3.3 | 1.00x |
| threads | 2 | 8.1 | 7.9 | 974 / 0 | 4.0 | 0.84x |
| threads | 4 | 7.8 | 7.8 | 980 / 0 | 3.9 | 0.86x |
| threads | 8 | 15.7 | 14.9 | 1001 / 0 | 4.5 | 0.43x |
| threads | 10 | 8.4 | 8.4 | 1000 / 0 | 4.3 | 0.81x |
| processes | 2 | 4.6 | 6.9 | 1139 / 220 | 2.3 | 1.47x |
| processes | 4 | 3.9 | 7.7 | 1090 / 156 | 2.5 | 1.75x |
| processes | 8 | 3.6 | 9.5 | 1066 / 141 | 2.6 | 1.87x |
| processes | 10 | 3.6 | 9.8 | 1056 / 140 | 2.7 | 1.89x |
plot: docs/parallel_bench.png
