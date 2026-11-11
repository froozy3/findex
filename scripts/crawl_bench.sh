#!/usr/bin/env bash
# Crawl a local mirror (scripts/mirror_site.py) at several concurrency levels:  scripts/crawl_bench.sh 8765
port=${1:-8765}
echo "| Concurrency | per-host | delay (s) | Pages | Wall (s) | Pages/s | Errors | Peak RSS (MB) |"
echo "|---|---|---|---|---|---|---|---|"
for config in "1 1 0" "5 5 0" "20 20 0" "20 2 0.1"; do
  read -r concurrency per_host delay <<< "$config"
  /usr/bin/time -l uv run findex crawl "http://127.0.0.1:$port/" --max-pages 200 --concurrency "$concurrency" \
    --per-host "$per_host" --delay "$delay" --out "/tmp/crawl_$concurrency.jsonl" --log-file "/tmp/crawl_$concurrency.log" \
    > /tmp/crawl_out.txt 2> /tmp/crawl_err.txt
  read -r pages _ errors _ _ wall < <(grep -o '[0-9]* pages ([0-9]* errors) in [0-9.]*s' /tmp/crawl_out.txt | tr -d '()s')
  rss=$(awk '/maximum resident/ {printf "%.0f", $1/1048576}' /tmp/crawl_err.txt)
  python3 - "$concurrency" "$per_host" "$delay" "$pages" "$wall" "$errors" "$rss" <<'PY'
import sys
c, h, d, pages, wall, errors, rss = sys.argv[1:]
print(f"| {c} | {h} | {d} | {pages} | {wall} | {int(pages) / float(wall):.1f} | {errors} | {rss} |")
PY
done
