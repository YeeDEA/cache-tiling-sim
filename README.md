# cache-tiling-sim

A small set-associative **cache simulator** (C++14, no dependencies) that replays the memory
access patterns of matrix multiplication and classifies every miss as **compulsory / capacity /
conflict**. It exists to answer one question with numbers: *why does loop tiling work, and when
does it silently fail?*

A companion **CUDA SGEMM suite** (`cuda/`) measures the same memory-hierarchy progression on real
GPU hardware — from global-memory naive, to shared-memory tiling, to **2D register tiling**,
benchmarked directly against **cuBLAS**.

## Build and run

```bash
make            # builds ./cachesim (Linux/macOS g++)
make test       # built-in sanity tests (3C classification, LRU, write-back)
./cachesim --workload tiled --n 256 --tile 32 --pad 8
bash scripts/run_experiments.sh   # reproduces the CPU tables below
bash scripts/run_elem.sh
```

On Windows PowerShell (e.g., MinGW `g++`):

```powershell
g++ -O2 -static -std=c++14 -Wall -Wextra -o cachesim.exe src/cachesim.cpp
.\cachesim.exe --selftest
.\scripts\run_experiments.ps1
.\scripts\run_elem.ps1
```

Options: `--workload naive|ikj|tiled`, `--n`, `--tile`, `--pad` (row padding, in elements),
`--elem` (bytes per element: `8/4/2/1`), `--bits` (`64/32/16/8/4`, where `4` = packed `int4`),
`--size/--assoc/--block` (cache geometry), `--hit/--penalty` (AMAT inputs),
`--trace FILE` (replay `R 0x..` / `W 0x..` lines).

## Model

- Write-back, write-allocate, true-LRU, single level. Default: 32 KB, 8-way, 64 B blocks (64 sets).
- Miss classification: *compulsory* = first touch of a block; otherwise *conflict* if a
  fully-associative LRU cache of the same capacity would have hit, else *capacity*.
- Matrices are row-major, `N = 256`, 8 B elements unless `--elem` or `--bits` says otherwise.
- `naive` is ijk with the accumulator in a register; `ikj` is the loop-interchanged version;
  `tiled` is blocked ijk (C partial sums are loaded and stored once per k-tile).
- AMAT = hit time + miss rate x penalty, with 1 and 100 cycles.

## CPU Cache Results (N = 256, fp64, 32 KB / 8-way / 64 B)

| workload | tile | pad | misses | miss rate % | compulsory | capacity | conflict | writebacks | AMAT |
|---|---|---|---|---|---|---|---|---|---|
| naive | - | 0 | 16923648 | 50.3381 | 24576 | 2088960 | 14810112 | 65504 | 51.338 |
| naive | - | 8 | 2113536 | 6.2865 | 24576 | 2088960 | 0 | 8190 | 7.287 |
| ikj | - | 0 | 2113536 | 4.1938 | 24576 | 2088960 | 0 | 8160 | 5.194 |
| tiled | 8 | 0 | 486848 | 1.2897 | 24576 | 269824 | 192448 | 200605 | 2.290 |
| tiled | 16 | 0 | 10640384 | 29.8455 | 24576 | 138240 | 10477568 | 1048574 | 30.845 |
| tiled | 32 | 0 | 17498112 | 50.5682 | 24576 | 141568 | 17331968 | 524284 | 51.568 |
| tiled | 64 | 0 | 17174528 | 50.3966 | 24576 | 2138112 | 15011840 | 262136 | 51.397 |
| tiled | 32 | 8 | 152256 | 0.4400 | 24576 | 120624 | 7056 | 21058 | 1.440 |

### What the numbers show

1. **Tiling works, but only if the tile actually stays resident.** Best tiled configuration
   (tile 32, pad 8) has 111x fewer misses than the naive loop and AMAT drops from 51.3 to 1.44 cycles.
2. **Power-of-two strides are a trap.** With `N = 256` doubles a row is 2 KB = 32 blocks, so walking a
   column of B touches only 2 of the 64 sets. `naive, pad 0` is 87% conflict misses (14.8M of 16.9M),
   and tiles of 16 and 32 fail for the same reason even though a 32x32 double tile (8 KB) fits easily in
   a 32 KB cache. Capacity was never the problem.
3. **Padding the leading dimension fixes it.** Adding 8 elements per row removes the aliasing:
   conflict misses go from 14.8M to 0 for `naive`, and from 17.3M to 7k for `tiled, tile 32`.
4. **Tiling and loop interchange attack the same problem.** `ikj` (no tiling) already reaches
   2.1M misses because its inner loop has unit stride; tiling adds reuse on top of that.

### Element width (relevance to quantization)

Same experiment with `--bits` swept from `64` to `4` (`fp64 / fp32 / fp16 / int8 / packed int4`), `pad = 8` elements:

| bytes/elem | workload | tile | miss rate % | misses | AMAT |
|---|---|---|---|---|---|
| 8 (`fp64`) | naive | - | 6.2865 | 2113536 | 7.287 |
| 8 (`fp64`) | tiled | 32 | 0.4400 | 152256 | 1.440 |
| 8 (`fp64`) | tiled | 64 | 2.3574 | 803356 | 3.357 |
| 4 (`fp32`) | naive | - | 3.2415 | 1089792 | 4.242 |
| 4 (`fp32`) | tiled | 32 | 0.2282 | 78976 | 1.228 |
| 4 (`fp32`) | tiled | 64 | 0.1623 | 55296 | 1.162 |
| 2 (`fp16`) | naive | - | 1.6211 | 545024 | 2.621 |
| 2 (`fp16`) | tiled | 32 | 0.1385 | 47936 | 1.139 |
| 2 (`fp16`) | tiled | 64 | 0.0972 | 33125 | 1.097 |
| 1 (`int8`) | naive | - | 1.3692 | 460327 | 2.369 |
| 1 (`int8`) | tiled | 32 | 0.0585 | 20244 | 1.059 |
| 1 (`int8`) | tiled | 64 | 0.0425 | 14478 | 1.042 |
| 0.5 (`int4`) | naive | - | 8.4288 | 2833763 | 9.429 |
| 0.5 (`int4`) | tiled | 32 | 0.0442 | 15290 | 1.044 |
| 0.5 (`int4`) | tiled | 64 | 0.0227 | 7734 | 1.023 |

Narrower elements shrink the footprint: a `64x64` tile overflows L1 at 8 B (`2.357%` misses) but fits cleanly at `4 B`
and below (`0.162% -> 0.023%` at packed `int4`), so **the best tile size depends on the data type**. Notice also
that `pad = 8` elements is only `4 B` at `int4` (less than a 64 B cache block), so the 256 column blocks land unevenly across the sets (some sets get 9 blocks for 8 ways, which thrashes under LRU) and pushes `naive` to `8.429%` (`2.79M` conflict misses); padding by a full cache line (`--pad 128` = `64 B`) drops
`naive` `int4` misses down to `8,672` (`0.026%`, AMAT `1.026`), while `tiled 64` hits `7,734` (`0.023%`, AMAT `1.023`).

## GPU Companion (`cuda/`) — Measured on RTX 5050 Laptop GPU (`sm_120`)

`cuda/kernels.cuh` and `cuda/matmul.cu` extend the blocking experiment onto an actual GPU memory hierarchy
(Global DRAM -> L2 -> Shared Memory -> Registers), adding a **2D register-tiled SGEMM** kernel with transposed,
bank-conflict-padded `As` staging and benchmarking against **cuBLAS**.

Two ways to run:

```bash
cd cuda && make run        # nvcc + -lcublas (RTX 50-series needs CUDA Toolkit 12.8+)
cd cuda && make nvrtc      # zero-nvcc runner via PyTorch's NVRTC + Driver API (Windows, CUDA 13.0)
cd cuda && make nvrtc-test # unaligned boundary verification (N=129, 255 across all 5 kernels)
```

Measured with `run_nvrtc.py` (median of 10 runs, `NVIDIA GeForce RTX 5050 Laptop GPU, sm_120`, CUDA 13.0). Correctness is checked against PyTorch's FP32 cuBLAS result, so the cuBLAS FP32 row is the reference itself. Laptop boost clocks vary run to run, so treat the numbers as approximate:

### N = 2048 (A, B, C = 16 MB each)

| kernel | time (ms) | GFLOP/s | vs naive | % of cuBLAS (FP32) | max rel err |
|---|---|---|---|---|---|
| `naive (16x16)` | 31.641 | 543.0 | 1.00x | 14.3% | 0.00e+00 |
| `tiled T=16` (SMEM only) | 22.116 | 776.8 | 1.43x | 20.4% | 0.00e+00 |
| `tiled T=32` (SMEM only) | 27.692 | 620.4 | 1.14x | 16.3% | 0.00e+00 |
| `reg_tiled 64x64 (4x4/thread)` | **5.826** | **2948.9** | **5.43x** | **77.6%** | 0.00e+00 |
| `reg_tiled 128x128 (8x8/thread)` | 6.406 | 2682.0 | 4.94x | 70.6% | 0.00e+00 |
| `cuBLAS (FP32 SIMT)` | 4.521 | 3800.2 | 7.00x | 100.0% | (reference) |
| `cuBLAS (TF32 Tensor Core)` | 2.788 | 6162.5 | 11.35x | 162.2% | 5.25e-05 |

### N = 4096 (A, B, C = 64 MB each — exceeds L2 working set)

| kernel | time (ms) | GFLOP/s | vs naive | % of cuBLAS (FP32) | max rel err |
|---|---|---|---|---|---|
| `naive (16x16)` | 542.187 | 253.5 | 1.00x | 8.2% | 0.00e+00 |
| `tiled T=16` (SMEM only) | 386.706 | 355.4 | 1.40x | 11.5% | 0.00e+00 |
| `tiled T=32` (SMEM only) | 240.318 | 571.9 | 2.26x | 18.5% | 0.00e+00 |
| `reg_tiled 64x64 (4x4/thread)` | 104.245 | 1318.4 | 5.20x | 42.6% | 0.00e+00 |
| `reg_tiled 128x128 (8x8/thread)` | **60.088** | **2287.3** | **9.02x** | **73.9%** | 0.00e+00 |
| `cuBLAS (FP32 SIMT)` | 44.399 | 3095.6 | 12.21x | 100.0% | (reference) |
| `cuBLAS (TF32 Tensor Core)` | 20.669 | 6649.4 | 26.23x | 214.8% | 5.90e-05 |

### What the GPU numbers show

1. **Shared-memory tiling alone still hits a memory wall.** `tiled T=16` reaches only 1.4x over naive: cutting DRAM traffic helps, but each thread still issues two shared-memory reads (`As[ty][k]`, `Bs[k][tx]`) per FMA (0.25 FLOP/byte out of SMEM). Furthermore, at `N=2048`, `T=32` (1024 threads/block) likely limits active warps/SM compared to `T=16` (occupancy not profiled here) and runs slower (`620.4` vs `776.8 GFLOP/s`), until DRAM bandwidth pressure dominates at `N=4096`.
2. **Register tiling unlocks compute saturation.** Holding a 2D `TM x TN` accumulator tile in thread registers turns each `k`-step into an outer product (`regA[TM] x regB[TN]`). With `8+8` SMEM loads feeding `64` FMAs (`128` FLOPs in `128x128 (8x8)`), SMEM read traffic per FLOP drops 8x, reaching **77.6% of `cuBLAS FP32 SIMT` at `N=2048`** and **9.02x speedup over naive (`73.9%` of cuBLAS FP32 SIMT) at `N=4096`**.
3. **Why the best tile flips with N.** At `N=2048`, `128x128` launches only `16 x 16 = 256` thread blocks, likely hitting wave-quantization and occupancy limits compared to `64x64` (`1024` blocks, `2948.9` vs `2682.0 GFLOP/s`; not profiled). At `N=4096`, `128x128` also has `1024` blocks while halving global A/B reloads when L2 misses hit DRAM, nearly doubling `64x64` (`2287.3` vs `1318.4 GFLOP/s`).
4. **SIMT vs. Tensor Core precision.** Disabling TF32 isolates true IEEE-754 FP32 SIMT throughput (`0.00e+00` error); enabling TF32 engages Blackwell Tensor Cores for another ~2.1x (`6649 GFLOP/s`) at `5.90e-5` relative rounding error.

## Connection to companion repositories

See [docs/DEEP_DIVE.md](docs/DEEP_DIVE.md) for line-by-line code walkthroughs, set-aliasing derivations, SMEM bank-conflict analysis, and full study notes bridging these repositories:

- **[`YeeDEA/mac-array-dv`](https://github.com/YeeDEA/mac-array-dv)** (SystemVerilog + UVM 1.2 output-stationary `4x4 / 8x8` INT8 MAC array, `57/57` regression, `12` SVA): Continues the pipeline-and-cache motivation noted in `mac-array-dv`'s README (originating from a 5-stage pipelined MIPS processor). Architecturally, an output-stationary PE array pins partial sums in PE accumulator registers just as `matmul_reg_tiled_impl` pins `acc[TM][TN]` in thread registers — shifting the bottleneck to external tile feeding (`cachesim`). Methodologically, `cuda/run_nvrtc.py --selftest` (`N=129, 255`) applies the same anti-vacuous-pass rule learned when auditing `mac-array-dv`'s `xsim dist` stimulus and assertion `A8`.
- **[`YeeDEA/quantcliff`](https://github.com/YeeDEA/quantcliff)** & **[`YeeDEA/jetson-orin-llm`](https://github.com/YeeDEA/jetson-orin-llm)** (Quantized LLM behavior on **RTX 5050 8 GB** & `INT4/NF4` edge deployment): In `quantcliff`, shrinking `Qwen3-4B` from `bf16` (`7822 MiB`, `5.9 tok/s`) to `Q8_0` (`5414 MiB`, `57.8 tok/s`) removes 8 GB VRAM spillover for a ~`10x` speedup with flat accuracy (`78%`), mirrored on `Llama-3.1-8B` (`7818 MiB` `8.1 tok/s` -> `Q4_K_M` `5648 MiB` `46.9 tok/s`). The `--bits` sweep here illustrates the same capacity-cliff mechanism one level down, inside a 32 KB L1 (an analogy, not the same cliff): `64x64` thrashes at `fp64` (`2.357%` misses) and drops cleanly to `0.0425%` (`int8`) and `0.0227%` (packed `int4`).

## Limitations

CPU simulator models a single cache level (write-back, true-LRU) without hardware prefetching or TLB effects; absolute cycle counts depend on core microarchitecture, while 3C miss trends and stride aliasing hold generally.

## Questions this repo should let you answer

- Why does `naive, pad 0` have mostly conflict misses even though the working set fits?
- What changes in the 3C breakdown when you pad the rows, and why?
- Why is the best tile size not simply "the largest one that fits"?
- How does shared-memory + register tiling on a GPU relate to cache blocking on a CPU?
- Why does halving the element width change the best tile size?

<details>
<summary>Reference answers (toggle for review)</summary>

1. **`naive, pad 0` conflict misses:** At `N=256` with `8 B` doubles, each row is `2048 B = 32` cache blocks. Walking down column `j` of `B` steps by `32` blocks, mapping every row strictly onto 2 of the 64 cache sets (`set s` and `set s+32`), overflowing 8 ways (`14.81M` conflict vs `2.09M` capacity misses).
2. **Effect of padding (`pad=8`):** Shifting leading dimension from `256` (`32` blocks) to `264` (`33` blocks, odd) breaks the power-of-two modulus alignment so column accesses spread evenly across all 64 sets. Conflict misses drop from `14.81M -> 0` (`naive`) and `17.33M -> 7,056` (`tiled T=32`).
3. **Largest tile != best tile:** Both footprint constraints (a `64x64` fp64 tile = `32 KB` each for A/B/C exceeds the 32 KB cache, `803k` misses, `779k` of them capacity) and parallelism/occupancy matter (on RTX 5050 at `N=2048`, `128x128` generates only 256 thread blocks, trailing `64x64`'s 1024 blocks).
4. **CPU blocking vs. GPU shared + register tiling:** CPU cache blocking improves reuse in hardware-managed L1 caches, bounded by register allocation and stride aliasing; GPU kernels explicitly stage tiles in programmer-managed Shared Memory (reducing staging-store bank conflicts via `+1` padding) and must add explicit 2D **register tiling** so SMEM bandwidth doesn't replace DRAM as the bottleneck.
5. **Halving element width (`--elem` / `--bits`):** Halving bits per element halves a `T x T` tile's byte footprint, shifting the capacity cliff upward (`64x64` thrashes at 8 B `fp64` `2.36%` misses, fits cleanly at $\le$ 4 B `0.162% -> 0.043%` at `int8`, and hits `0.023%` at packed `int4`), analogous to how `Q8_0` / `Q4_K_M` quantization moves `Qwen3-4B` (`5.9 -> 57.8 tok/s`) and `Llama-3.1-8B` inside the 8 GB RTX 5050 ceiling in `quantcliff`. Also notice that padding must scale relative to cache-line size (8 elements = 64 B at `fp64`, vs. only 4 B at `int4`).

</details>
