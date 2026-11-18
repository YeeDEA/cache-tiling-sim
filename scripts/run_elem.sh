#!/usr/bin/env bash
# Effect of element width (fp64 -> packed int4) on cache behaviour, N=256, 32 KB 8-way 64 B.
set -euo pipefail
BIN=./cachesim
echo "| bytes/elem | workload | tile | pad | miss rate % | misses | AMAT |"
echo "|---|---|---|---|---|---|---|"
for pair in "8:64" "4:32" "2:16" "1:8" "0.5:4"; do
  E="${pair%%:*}"
  B="${pair##*:}"
  for cfg in "--workload naive --pad 8" "--workload tiled --tile 32 --pad 8" "--workload tiled --tile 64 --pad 8"; do
    $BIN --csv --n 256 --bits "$B" $cfg | awk -F, -v e="$E" '{printf "| %s | %s | %s | %s | %s | %s | %s |\n", e,$1,($1=="tiled"?$3:"-"),$4,$6,$5,$11}'
  done
done
