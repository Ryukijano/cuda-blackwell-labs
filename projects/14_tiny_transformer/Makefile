# Project 14: Tiny Transformer Training — Makefile

PROJECT := 14_tiny_transformer
# Use the 3d_recon env which has PyTorch + CUDA 13 support on this machine.
PYTHON  ?= /home/aimsgroupuol/miniconda3/envs/3d_recon/bin/python

.PHONY: all clean run

all:
	@echo "Python project — no compilation needed. Run 'make run' to train."

run:
	@mkdir -p ../../results
	$(PYTHON) train.py | tee ../../results/$(PROJECT).txt

clean:
	rm -f *.pyc
