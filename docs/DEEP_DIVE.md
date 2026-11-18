# Design Notes

Technical walkthrough of `cache-tiling-sim`, its memory-hierarchy derivations, and how it bridges the CPU pipeline, hardware accelerator RTL/DV, and quantized LLM deployment repositories under `github.com/YeeDEA`.

---

## 1. CPU Cache Simulator (`src/cachesim.cpp`)

### 1.1 Address Slicing & Packed Sub-Byte (`--bits`) Addressing
Default geometry is `32768 B` (`32 KB`), `8`-way set-associative, `64 B` cache blocks:

- `blocks = 32768 / 64 = 512`
- `sets = 512 / 8 = 64` -> `idx_bits_ = 6`, `off_bits_ = 6`
- `blk = addr >> 6`, `set = blk & 63`, `tag = blk >> 6`

In `run_matmul`, element index `idx = row * ld + col` (`ld = n + pad`) maps to byte address:

```cpp
base + (((idx * B_bits) >> 3))
```

- When `B_bits` is `64 / 32 / 16 / 8` (`fp64 / fp32 / fp16 / int8`), this matches `base + idx * elem` byte-for-byte.
- When `B_bits == 4` (packed `int4`), two adjacent elements share one byte and **128 elements fit inside a single `64 B` cache line**.

### 1.2 Why `FullyAssocLRU` Must Be Updated Before Checking Set-Associative Hits
In `Cache::access` (`src/cachesim.cpp`):

```cpp
const bool shadow_hit = shadow_.access(blk);  // keep shadow in sync on every access
```

`shadow_` implements Hill–Smith 3C classification against an idealized fully-associative LRU cache of identical capacity (`size / block = 512` blocks). Updating `shadow_` **before** returning on a set-associative hit keeps the recency order of the fully-associative baseline identical on every memory access:
- **Compulsory**: `!seen_.count(blk)` (first touch ever).
- **Conflict**: seen before, hits in `shadow_` (capacity was sufficient; set-associative index collision caused eviction).
- **Capacity**: seen before, misses in both real and `shadow_` caches.

### 1.3 Why Stride Aliasing Breaks `naive` & `tiled`, and Why `pad = 8` Flips at `INT4`
- At `N = 256, fp64 (8 B)`, one row is `256 * 8 = 2048 B = 32` blocks. Stepping down a column of `B` advances the set index by `32 mod 64` every row, touching **only 2 of 64 sets** (`s` and `s + 32`). That leaves at most `2 * 8 = 16` ways for `256` column elements, turning `naive pad=0` into `87.5%` conflict misses (`14,810,112` conflict vs. `2,088,960` capacity).
- Adding `--pad 8` shifts `ld` to `264` doubles (`33` blocks/row, coprime to `64`), cycling cleanly across all 64 sets (`conflict` drops `14.81M -> 0`).
- **Sub-byte subtlety (`--bits 4`, packed `INT4`)**: `--pad 8` adds only `8 * 0.5 = 4 B` (1/16th of a `64 B` block). The row pitch becomes `132 B` (2 blocks + 4 B). The `256` blocks of one `B` column then spread over `46` sets, but for some column offsets (e.g. `j = 37`) four sets receive `9` blocks, one more than the `8` ways, and under LRU a cyclic walk over `ways + 1` blocks misses on every access. At `pad=0` the same column fits exactly (`32` sets x `8` blocks), so `naive` jumps from `1.770%` to `8.429%` at `pad=8` (`2.79M` conflicts). Padding by one full cache block (`--pad 128` = `64 B`, making each row `3` blocks) drops `naive` `INT4` misses down to `8,672` (`0.026%`, `AMAT 1.026`), while `tiled 64` with `pad=8` reaches `7,734` (`0.0227%`, `AMAT 1.023`).

---

## 2. CUDA SGEMM Hierarchy (`cuda/kernels.cuh`, `cuda/matmul.cu`, `cuda/run_nvrtc.py`)

### 2.1 Arithmetic Intensity Across the Three Kernels
For `N = 4096` (`A`, `B`, `C` each `64 MB`, overflowing on-chip caches):

1. **`matmul_naive_impl`**: Each thread executes 1 FMA (`2` FLOPs) after reading `A[row,k]` (`4 B`) and `B[k,col]` (`4 B`) from global memory -> `2 / 8 = 0.25 FLOP/B` of load traffic per thread (real DRAM traffic is lower thanks to coalescing and L1/L2 reuse; ~`253 GFLOP/s`).
2. **`matmul_tiled_impl<T>`**: Cuts global DRAM loads by `T`, **but** the inner loop still loads `As[ty][k]` (`4 B`) and `Bs[k][tx]` (`4 B`) from Shared Memory for every single FMA -> still `0.25 FLOP/B` out of SMEM (~`355–572 GFLOP/s`). Also, `T = 32` uses `1024` threads/block, which likely limits active warps per SM at `N = 2048` (`620.4` vs `776.8 GFLOP/s` for `T = 16`; occupancy was not profiled).
3. **`matmul_reg_tiled_impl<BM, BN, BK, TM, TN>`**: Stages a `BK x (BM + 1)` transposed `As` tile and `BK x BN` `Bs` tile, then computes an outer product in registers (`acc[TM][TN] += regA[TM] * regB[TN]`):
   - In `128x128 (TM=8, TN=8)`, `8 + 8 = 16` floats (`64 B`) read from SMEM feed `64` FMAs (`128` FLOPs) -> **`2.0 FLOP/B` out of SMEM** (8x compute intensity vs `matmul_tiled`).
   - Transposing `A` into `As[BK][BM + 1]` gives `stride = BM + 1` (`65` or `129`, both odd and coprime to `32` SMEM banks), which reduces bank conflicts on the staging stores along `k`. The `As` reads are warp-broadcast (a warp spans at most 2 distinct `ty`), while the `Bs` reads at stride `TN` across lanes may still conflict; confirm with Nsight Compute's shared-memory bank-conflict counters.

### 2.2 `128x128 (8x8)` vs `64x64 (4x4)`
- Measured on RTX 5050 Laptop, 3 back-to-back runs (`docs/results/nvrtc_bench_run*.log`): `128x128` wins at both sizes,
  **`2,795–3,261` vs `2,319–2,508 GFLOP/s` at `N = 2048`** and **`3,570–3,622` vs `2,347–2,358` at `N = 4096`**
  (`10.05x` over naive, `72.9%` of FP32 SIMT cuBLAS in run 3).
- Why: per thread, 8x8 does 4x more FMAs per SMEM load than 4x4, and per block A/B tiles are reloaded half as often.
  The margin grows at `N = 4096` (~1.5x vs ~1.3x) because those reloads then miss L2.
- `128x128` launches only `256` blocks at `N = 2048`, so wave quantization could have hurt it; the measurements
  show it does not dominate here. An earlier unlogged run had `64x64` ahead at `N = 2048`; it did not reproduce.
  Occupancy, register count and spills are still unprofiled (Nsight Compute).

### 2.3 Non-Vacuous Verification (`run_nvrtc.py --selftest`)
Both `N = 2048` and `N = 4096` are exact multiples of `128`, so boundary guards (`gr < n`, `gk < n`, `gc < n`) in `cuda/kernels.cuh` are never exercised during the `2048 / 4096` benchmark. `python run_nvrtc.py --selftest` tests unaligned sizes `N = 129` and `N = 255` across all 5 custom kernels against FP32 cuBLAS (`torch.backends.cuda.matmul.allow_tf32 = False`, `rel_err < 1e-3`).

---

## 3. Bridges to Companion Repositories (`github.com/YeeDEA`)

| Question / Principle | In `cache-tiling-sim` | In Companion Repositories |
|---|---|---|
| **Register accumulation vs. memory feed** | `matmul_reg_tiled_impl` pins `acc[TM][TN]` in thread registers so outer-product reuse shifts pressure onto `As`/`Bs` staging. | **`YeeDEA/mac-array-dv`**: `mac_pe.sv` / `mac_array_4x4.sv` keeps partial sums pinned inside output-stationary PE accumulators (`K=64` sweep at `N=4, 8`), pushing the system bottleneck to feeding A/B tiles. |
| **Pipeline execution -> Memory wall** | Quantifies 3C misses, set-associative conflict aliasing, and `AMAT` across loop orderings. | **`YeeDEA/mac-array-dv` Motivation**: Grew directly out of a 5-stage pipelined MIPS CPU project analyzing hazard stalls, CPI, and cache hit/miss behavior. |
| **Do not trust unverified test conditions** | `run_nvrtc.py --selftest` tests `N=129, 255` because `2048/4096` pass boundary predicates vacuously. | **`YeeDEA/mac-array-dv`**: Caught `xsim` silently ignoring constrained-random `dist` weights while coverage read 100%, and reworked assertion `A8` after finding a vacuous pass. |
| **Quantization capacity cliffs** | `--bits 64 -> 4` moves a `64x64` tile out of L1 thrashing (`2.357%` at `fp64`) down to `0.043%` (`int8`) and `0.023%` (`int4`). | **`YeeDEA/quantcliff`** (RTX 5050 8 GB): `Qwen3-4B` fits comfortably at `Q8_0` (`5414 MiB`, `57.8 tok/s`) vs spilling 8 GB at `bf16` (`7822 MiB`, `5.9 tok/s` — `10x` gap); **`YeeDEA/jetson-orin-llm`** targets `NF4` / `INT4 AWQ` under the same 8 GB ceiling. |
