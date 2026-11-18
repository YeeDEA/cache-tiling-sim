#pragma once

// One thread computes one C element; every k-step reads A and B from global memory.
__device__ __forceinline__ void matmul_naive_impl(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float acc = 0.f;
        for (int k = 0; k < n; ++k) acc += A[row * n + k] * B[k * n + col];
        C[row * n + col] = acc;
    }
}

// A TxT tile of A and B is staged in shared memory once and reused T times,
// reducing global-memory loads per element by a factor of T.
template <int T>
__device__ __forceinline__ void matmul_tiled_impl(const float* A, const float* B, float* C, int n) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty;
    const int col = blockIdx.x * T + tx;
    float acc = 0.f;

    for (int t = 0; t < (n + T - 1) / T; ++t) {
        const int a_col = t * T + tx;
        const int b_row = t * T + ty;
        As[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.f;
        Bs[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = acc;
}

// 2D Register Tiling: each block computes a BM x BN tile in shared memory,
// while each thread accumulates a TM x TN sub-tile in registers to cut shared-memory read pressure.
// Threads per block must equal (BN / TN) x (BM / TM).
template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void matmul_reg_tiled_impl(const float* A, const float* B, float* C, int n) {
    // Transpose A into [BK][BM + 1] in shared memory; +1 padding avoids bank conflicts.
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[m][j] = 0.f;

    float regA[TM];
    float regB[TN];

    for (int k0 = 0; k0 < n; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
            int m = idx / BK;
            int k = idx % BK;
            int gr = block_row + m;
            int gk = k0 + k;
            As[k][m] = (gr < n && gk < n) ? A[gr * n + gk] : 0.f;
        }
        for (int idx = tid; idx < BK * BN; idx += num_threads) {
            int k = idx / BN;
            int j = idx % BN;
            int gk = k0 + k;
            int gc = block_col + j;
            Bs[k][j] = (gk < n && gc < n) ? B[gk * n + gc] : 0.f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int m = 0; m < TM; ++m) regA[m] = As[k][ty * TM + m];
#pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int m = 0; m < TM; ++m)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[m][j] = fmaf(regA[m], regB[j], acc[m][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; ++m) {
        int gr = block_row + ty * TM + m;
        if (gr < n) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int gc = block_col + tx * TN + j;
                if (gc < n) C[gr * n + gc] = acc[m][j];
            }
        }
    }
}
