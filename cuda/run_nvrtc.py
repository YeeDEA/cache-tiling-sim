"""
Compile and benchmark the CUDA kernels in kernels.cuh via NVRTC + CUDA Driver API,
and compare against PyTorch's cuBLAS SGEMM (both FP32 SIMT and TF32 Tensor Core).

Useful on Windows/laptops that have a CUDA-enabled PyTorch installation without
a standalone nvcc / Visual Studio host-compiler setup.
"""
import ctypes
import glob
import os
import statistics
import sys

import torch


def _ck(res: int, what: str) -> None:
    """Raise on a non-zero NVRTC / driver-API status (unlike `assert`, survives `python -O`)."""
    if res != 0:
        raise RuntimeError(f"{what} failed with status {res}")


def _find_dll(torch_lib: str, prefix: str) -> str:
    matches = sorted(p for p in glob.glob(os.path.join(torch_lib, f"{prefix}*.dll")) if ".alt." not in os.path.basename(p))
    if not matches:
        matches = sorted(glob.glob(os.path.join(torch_lib, f"{prefix}*.dll")))
    if not matches:
        raise FileNotFoundError(f"Could not find {prefix}*.dll in {torch_lib}")
    return matches[0]


def load_cuda_driver():
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(torch_lib)
    nvrtc_path = _find_dll(torch_lib, "nvrtc64_")
    nvrtc = ctypes.WinDLL(nvrtc_path) if os.name == "nt" else ctypes.CDLL(nvrtc_path)
    nvcuda = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")

    nvcuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    nvcuda.cuModuleLoadData.restype = ctypes.c_int
    nvcuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    nvcuda.cuModuleGetFunction.restype = ctypes.c_int
    nvcuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
    ]
    nvcuda.cuLaunchKernel.restype = ctypes.c_int

    nvrtc.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
    nvrtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    return nvrtc, nvcuda


def compile_kernels(nvrtc, nvcuda, major: int, minor: int):
    kernels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.cuh")
    with open(kernels_path, "r", encoding="utf-8") as f:
        header_src = f.read().replace("#pragma once", "")

    wrapper_src = header_src + r"""
extern "C" {
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) { matmul_naive_impl(A, B, C, n); }
__global__ void matmul_tiled_16(const float* A, const float* B, float* C, int n) { matmul_tiled_impl<16>(A, B, C, n); }
__global__ void matmul_tiled_32(const float* A, const float* B, float* C, int n) { matmul_tiled_impl<32>(A, B, C, n); }
__global__ void matmul_reg_64_4x4(const float* A, const float* B, float* C, int n) { matmul_reg_tiled_impl<64, 64, 16, 4, 4>(A, B, C, n); }
__global__ void matmul_reg_128_8x8(const float* A, const float* B, float* C, int n) { matmul_reg_tiled_impl<128, 128, 8, 8, 8>(A, B, C, n); }
}
"""
    prog = ctypes.c_void_p()
    _ck(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), wrapper_src.encode("utf-8"), b"kernels.cu", 0, None, None), "nvrtcCreateProgram")
    arch_opt = f"--gpu-architecture=sm_{major}{minor}".encode("utf-8")
    opts = (ctypes.c_char_p * 1)(arch_opt)
    res = nvrtc.nvrtcCompileProgram(prog, 1, opts)
    if res != 0:
        log_sz = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
        buf = ctypes.create_string_buffer(log_sz.value)
        nvrtc.nvrtcGetProgramLog(prog, buf)
        raise RuntimeError(buf.value.decode("utf-8", errors="replace"))

    cubin_sz = ctypes.c_size_t()
    _ck(nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_sz)), "nvrtcGetCUBINSize")
    cubin = ctypes.create_string_buffer(cubin_sz.value)
    _ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")

    mod = ctypes.c_void_p()
    _ck(nvcuda.cuModuleLoadData(ctypes.byref(mod), cubin), "cuModuleLoadData")
    fns = {}
    for name in ("matmul_naive", "matmul_tiled_16", "matmul_tiled_32", "matmul_reg_64_4x4", "matmul_reg_128_8x8"):
        fn = ctypes.c_void_p()
        _ck(nvcuda.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode("utf-8")), "cuModuleGetFunction")
        fns[name] = fn
    return fns


def launch_kernel(nvcuda, fn, grid, block, dA: int, dB: int, dC: int, n: int) -> None:
    pA, pB, pC, pn = ctypes.c_uint64(dA), ctypes.c_uint64(dB), ctypes.c_uint64(dC), ctypes.c_int(n)
    args = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.pointer(pA), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(pB), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(pC), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(pn), ctypes.c_void_p),
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _ck(nvcuda.cuLaunchKernel(fn, grid[0], grid[1], 1, block[0], block[1], 1, 0, stream, args, None), "cuLaunchKernel")


KERNEL_SPECS = [
    ("naive (16x16)", "matmul_naive", lambda n: (((n + 15) // 16, (n + 15) // 16), (16, 16))),
    ("tiled T=16", "matmul_tiled_16", lambda n: (((n + 15) // 16, (n + 15) // 16), (16, 16))),
    ("tiled T=32", "matmul_tiled_32", lambda n: (((n + 31) // 32, (n + 31) // 32), (32, 32))),
    ("reg_tiled 64x64 (4x4)", "matmul_reg_64_4x4", lambda n: (((n + 63) // 64, (n + 63) // 64), (16, 16))),
    ("reg_tiled 128x128 (8x8)", "matmul_reg_128_8x8", lambda n: (((n + 127) // 128, (n + 127) // 128), (16, 16))),
]


def _init_runtime():
    """Initialize CUDA context and compile all kernels in kernels.cuh via NVRTC."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available.")
    torch.cuda.init()
    dev = torch.cuda.current_device()
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    major, minor = torch.cuda.get_device_capability(dev)
    nvrtc, nvcuda = load_cuda_driver()
    fns = compile_kernels(nvrtc, nvcuda, major, minor)
    return dev, major, minor, nvcuda, fns


def _make_inputs(n: int, seed: int):
    torch.manual_seed(seed)
    A = torch.randint(0, 100, (n, n), device="cuda", dtype=torch.int32).float() / 100.0
    B = torch.randint(0, 100, (n, n), device="cuda", dtype=torch.int32).float() / 100.0
    C = torch.empty((n, n), device="cuda", dtype=torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    return A, B, C, torch.matmul(A, B)


def run_suite(sizes):
    dev, major, minor, nvcuda, fns = _init_runtime()
    print(f"device: {torch.cuda.get_device_name(dev)} (sm_{major}{minor})")

    for n in sizes:
        print(f"\n=== N = {n} ===")
        A, B, C, ref = _make_inputs(n, seed=1)
        # independent FP64 reference: shows the true FP32 rounding error, which the
        # FP32-vs-cuBLAS column cannot (it is 0 when both use the same k order)
        ref64 = torch.matmul(A.double(), B.double())
        flops = 2.0 * n * n * n
        results = []

        def bench(label, fn_call, reps=10):
            C.zero_()
            for _ in range(2):
                fn_call()
            torch.cuda.synchronize()
            times = []
            for _ in range(reps):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn_call()
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            med = statistics.median(times)
            gflops = flops / (med * 1e6)
            max_rel = ((C - ref).abs() / torch.clamp(ref.abs(), min=1.0)).max().item()
            rel64 = ((C.double() - ref64).abs() / torch.clamp(ref64.abs(), min=1.0)).max().item()
            results.append((label, med, gflops, max_rel, rel64))
            return gflops

        g_naive = None
        for label, sym, launch_cfg in KERNEL_SPECS:
            grid, block = launch_cfg(n)
            g = bench(label, lambda s=sym, g_=grid, b_=block: launch_kernel(
                nvcuda, fns[s], g_, b_, A.data_ptr(), B.data_ptr(), C.data_ptr(), n
            ))
            if g_naive is None:
                g_naive = g

        torch.backends.cuda.matmul.allow_tf32 = False
        g_cublas_fp32 = bench("cuBLAS (FP32 SIMT)", lambda: torch.matmul(A, B, out=C))
        torch.backends.cuda.matmul.allow_tf32 = True
        bench("cuBLAS (TF32 TC)", lambda: torch.matmul(A, B, out=C))
        torch.backends.cuda.matmul.allow_tf32 = False

        print(f"{'kernel':<24} {'ms':>8} {'GFLOP/s':>10} {'vs naive':>9} {'%cuBLAS':>9} {'rel_err':>10} {'err_fp64':>10}")
        for label, med, gflops, rel, rel64 in results:
            print(f"{label:<24} {med:8.3f} {gflops:10.1f} {gflops / g_naive:8.2f}x {100.0 * gflops / g_cublas_fp32:8.1f}% {rel:10.2e} {rel64:10.2e}")


def run_selftest():
    """Verify every kernel on unaligned sizes not divisible by 16/32/64/128."""
    _, _, _, nvcuda, fns = _init_runtime()
    for n in (129, 255):
        A, B, C, ref = _make_inputs(n, seed=n)
        for label, sym, launch_cfg in KERNEL_SPECS:
            C.zero_()
            grid, block = launch_cfg(n)
            launch_kernel(nvcuda, fns[sym], grid, block, A.data_ptr(), B.data_ptr(), C.data_ptr(), n)
            torch.cuda.synchronize()
            rel = ((C - ref).abs() / torch.clamp(ref.abs(), min=1.0)).max().item()
            if not rel < 1e-3:  # `not <` also catches NaN
                raise AssertionError(f"{label} N={n} rel_err={rel}")
    print("cuda selftest passed (N=129, 255 unaligned boundaries)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        run_selftest()
    else:
        sizes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [2048, 4096]
        run_suite(sizes)
