#!/usr/bin/env bash
# CSV for docs/heatmap.png: tiled matmul miss rate over tile x pad, for fp64 and packed int4.
set -euo pipefail
BIN=./cachesim
echo "bits,tile,pad,miss_rate"
for B in 64 4; do
  for T in 8 16 32 64 128; do
    for P in 0 1 2 4 8 16 32; do
      $BIN --csv --n 256 --workload tiled --tile "$T" --pad "$P" --bits "$B" |
        awk -F, -v b="$B" '{printf "%s,%s,%s,%s\n", b,$3,$4,$6}'
    done
  done
done