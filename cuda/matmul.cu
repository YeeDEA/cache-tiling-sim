// matmul.cu - naive vs shared-memory tiled vs 2D register-tiled SGEMM (+ optional cuBLAS).
//
// Build: make        Run: ./matmul [N]      (default N = 2048)
// Or without nvcc on systems with PyTorch+CUDA: python run_nvrtc.py [N ...]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

#include "kernels.cuh"

#define CUDA_CHECK(x)                                                              \
    do {                                                                           \
        cudaError_t err__ = (x);                                                   \
        if (err__ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                       \
                         cudaGetErrorString(err__), __FILE__, __LINE__);           \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    matmul_naive_impl(A, B, C, n);
}

template <int T>
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    matmul_tiled_impl<T>(A, B, C, n);
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_reg_tiled(const float* A, const float* B, float* C, int n) {
    matmul_reg_tiled_impl<BM, BN, BK, TM, TN>(A, B, C, n);
}

template <typename LaunchFn>
static float time_op(LaunchFn fn, int reps) {
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    fn();  // warm-up
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < reps; ++i) fn();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms / reps;
}

static void launch_naive(const float* A, const float* B, float* C, int n) {
    dim3 blk(16, 16), grd((n + 15) / 16, (n + 15) / 16);
    matmul_naive<<<grd, blk>>>(A, B, C, n);
}
template <int T>
static void launch_tiled(const float* A, const float* B, float* C, int n) {
    dim3 blk(T, T), grd((n + T - 1) / T, (n + T - 1) / T);
    matmul_tiled<T><<<grd, blk>>>(A, B, C, n);
}
template <int BM, int BN, int BK, int TM, int TN>
static void launch_reg_tiled(const float* A, const float* B, float* C, int n) {
    dim3 blk(BN / TN, BM / TM), grd((n + BN - 1) / BN, (n + BM - 1) / BM);
    matmul_reg_tiled<BM, BN, BK, TM, TN><<<grd, blk>>>(A, B, C, n);
}

// Compare 64 random C elements with a double-precision CPU dot product.
static bool spot_check(const std::vector<float>& A, const std::vector<float>& B,
                       const std::vector<float>& C, int n, double tol = 1e-3) {
    std::srand(1);
    for (int t = 0; t < 64; ++t) {
        int i = std::rand() % n, j = std::rand() % n;
        double ref = 0.0;
        for (int k = 0; k < n; ++k) ref += (double)A[(size_t)i * n + k] * B[(size_t)k * n + j];
        double got = C[(size_t)i * n + j];
        if (std::fabs(got - ref) > tol * std::fmax(1.0, std::fabs(ref))) {
            std::fprintf(stderr, "mismatch at (%d,%d): got %f, ref %f\n", i, j, got, ref);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    const int n = (argc > 1) ? std::atoi(argv[1]) : 2048;
    if (n <= 0) { std::fprintf(stderr, "N must be positive\n"); return 2; }
    const size_t elems = (size_t)n * n, bytes = elems * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (sm_%d%d), N = %d\n", prop.name, prop.major, prop.minor, n);

    std::vector<float> hA(elems), hB(elems), hC(elems);
    for (size_t i = 0; i < elems; ++i) {
        hA[i] = (float)(std::rand() % 100) / 100.f;
        hB[i] = (float)(std::rand() % 100) / 100.f;
    }
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    struct Entry { const char* name; void (*fn)(const float*, const float*, float*, int); };
    const Entry kernels[] = {
        {"naive (16x16 threads)", launch_naive},
        {"tiled T=16", launch_tiled<16>},
        {"tiled T=32", launch_tiled<32>},
        {"reg_tiled 64x64 (4x4)", launch_reg_tiled<64, 64, 16, 4, 4>},
        {"reg_tiled 128x128 (8x8)", launch_reg_tiled<128, 128, 8, 8, 8>},
    };
    const double flops = 2.0 * (double)n * n * n;
    for (const Entry& k : kernels) {
        CUDA_CHECK(cudaMemset(dC, 0, bytes));
        float ms = time_op([&]() { k.fn(dA, dB, dC, n); }, 10);
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        bool ok = spot_check(hA, hB, hC, n);
        std::printf("%-26s %8.3f ms  %9.1f GFLOP/s  %s\n", k.name, ms, flops / (ms * 1e6),
                    ok ? "OK" : "FAILED");
    }

#ifdef USE_CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    // Row-major C = A * B equals Column-major C^T = B^T * A^T
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    CUDA_CHECK(cudaMemset(dC, 0, bytes));
    float ms_fp32 = time_op([&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n);
    }, 10);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    bool ok_fp32 = spot_check(hA, hB, hC, n);
    std::printf("%-26s %8.3f ms  %9.1f GFLOP/s  %s\n", "cuBLAS (FP32 SIMT)", ms_fp32, flops / (ms_fp32 * 1e6),
                ok_fp32 ? "OK" : "FAILED");

    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    CUDA_CHECK(cudaMemset(dC, 0, bytes));
    float ms_tf32 = time_op([&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n);
    }, 10);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    bool ok_tf32 = spot_check(hA, hB, hC, n);
    std::printf("%-26s %8.3f ms  %9.1f GFLOP/s  %s\n", "cuBLAS (TF32 TC)", ms_tf32, flops / (ms_tf32 * 1e6),
                ok_tf32 ? "OK" : "FAILED");
    cublasDestroy(handle);
#endif

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
