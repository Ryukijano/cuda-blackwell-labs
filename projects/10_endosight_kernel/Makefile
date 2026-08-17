# Project 10: Custom Endosight CUDA Kernel — Makefile

PROJECT := 10_endosight_kernel

.PHONY: all build run clean

all: build

build:
	conda run -n 3d_recon python setup.py build_ext --inplace

run: build
	conda run -n 3d_recon python benchmark.py | tee ../../results/$(PROJECT).txt

clean:
	rm -rf build dist *.egg-info endosight_kernel*.so
	rm -f endosight_kernel.cpython*.so
