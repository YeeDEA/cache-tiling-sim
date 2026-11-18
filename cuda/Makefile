# RTX 50-series (Blackwell, compute capability 12.0) needs CUDA Toolkit 12.8 or newer.
# -arch=native compiles for whatever GPU is installed.
NVCC   ?= nvcc
PYTHON ?= python

matmul: matmul.cu kernels.cuh
	$(NVCC) -O3 -arch=native -DUSE_CUBLAS -lcublas -o $@ $<

run: matmul
	./matmul 2048

nvrtc:
	$(PYTHON) run_nvrtc.py 2048 4096

nvrtc-test:
	$(PYTHON) run_nvrtc.py --selftest

clean:
	rm -f matmul matmul.exe

.PHONY: run nvrtc nvrtc-test clean
