#!/usr/bin/env bash
# Is the int4 "pad 8" cliff an LRU artifact? Same accesses, four replacement policies.
# N=256, naive ijk, 32 KB / 8-way / 64 B. 3C split uses a fully-associative LRU shadow for every policy.
set -euo pipefail
BIN=./cachesim
echo "| data | pad | policy | misses | miss rate % | conflict | AMAT |"
echo "|---|---|---|---|---|---|---|"
for cfg in "64:0" "64:8" "4:0" "4:8" "4:128"; do
  B="${cfg%%:*}"; P="${cfg##*:}"
  name=$([ "$B" = 64 ] && echo fp64 || echo int4)
  for pol in lru fifo random plru; do
    $BIN --csv --n 256 --workload naive --bits "$B" --pad "$P" --policy "$pol" |
      awk -F, -v d="$name" -v p="$pol" '{printf "| %s | %s | %s | %s | %s | %s | %s |\n", d,$4,p,$5,$6,$9,$11}'
  done
done