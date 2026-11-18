# Design questions and answers

- Why does `naive, pad 0` have mostly conflict misses even though the working set fits?
- What changes in the 3C breakdown when you pad the rows, and why?
- Why is the best tile size not simply "the largest one that fits"?
- How does shared-memory + register tiling on a GPU relate to cache blocking on a CPU?
- Why does halving the element width change the best tile size?

## Answers

1. **`naive, pad 0` conflict misses:** At `N=256` with `8 B` doubles, each row is `2048 B = 32` cache blocks. Walking down column `j` of `B` steps by `32` blocks, mapping every row strictly onto 2 of the 64 cache sets (`set s` and `set s+32`), overflowing 8 ways (`14.81M` conflict vs `2.09M` capacity misses).
2. **Effect of padding (`pad=8`):** Shifting leading dimension from `256` (`32` blocks) to `264` (`33` blocks, odd) breaks the power-of-two modulus alignment so column accesses spread evenly across all 64 sets. Conflict misses drop from `14.81M -> 0` (`naive`) and `17.33M -> 7,056` (`tiled T=32`).
3. **Largest tile != best tile:** On the CPU, footprint decides: a `64x64` fp64 tile is `32 KB` per matrix, so A/B/C overflow the 32 KB cache (`803k` misses vs `152k` for `T=32`, both `pad 8`). Aliasing matters too: with `pad 0`, every `T >= 16` fails regardless of size ([heatmap](heatmap.png)). On the GPU the opposite happens here: the larger `128x128` register tile wins because it raises FLOPs per SMEM/L2 load.
4. **CPU blocking vs. GPU shared + register tiling:** CPU cache blocking improves reuse in hardware-managed L1 caches, bounded by register allocation and stride aliasing; GPU kernels explicitly stage tiles in programmer-managed Shared Memory (reducing staging-store bank conflicts via `+1` padding) and must add explicit 2D **register tiling** so SMEM bandwidth doesn't replace DRAM as the bottleneck.
5. **Halving element width (`--elem` / `--bits`):** Halving bits per element halves a `T x T` tile's byte footprint, shifting the capacity cliff upward (`64x64` thrashes at 8 B `fp64` `2.36%` misses, fits cleanly at $\le$ 4 B `0.162% -> 0.043%` at `int8`, and hits `0.023%` at packed `int4`), analogous to how `Q8_0` / `Q4_K_M` quantization moves `Qwen3-4B` (`5.9 -> 57.8 tok/s`) and `Llama-3.1-8B` inside the 8 GB RTX 5050 ceiling in `quantcliff`. Also notice that padding must scale relative to cache-line size (8 elements = 64 B at `fp64`, vs. only 4 B at `int4`).

