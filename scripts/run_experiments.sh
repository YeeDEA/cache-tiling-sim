#!/usr/bin/env bash
# Reproduces the table in README.md (32 KB, 8-way, 64 B blocks, N=256 doubles).
set -euo pipefail
BIN=./cachesim
N=${N:-256}
echo "| workload | tile | pad | misses | miss rate % | compulsory | capacity | conflict | writebacks | AMAT |"
echo "|---|---|---|---|---|---|---|---|---|---|"
row() { $BIN --csv --n "$N" "$@" | awk -F, '{printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",$1,($1=="tiled"?$3:"-"),$4,$5,$6,$7,$8,$9,$10,$11}'; }
row --workload naive
row --workload naive --pad 8
row --workload ikj
for T in 8 16 32 64; do row --workload tiled --tile "$T"; done
row --workload tiled --tile 32 --pad 8
