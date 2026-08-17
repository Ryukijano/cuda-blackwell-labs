# Project 09: NVDEC Video Pipeline — Makefile

PROJECT := 09_nvdec_pipeline

.PHONY: all run test clean

all:
	@echo "Run with: conda run -n 3d_recon python nvdec_pipeline.py"

run: test

test:
	mkdir -p ../../results
	conda run -n 3d_recon python nvdec_pipeline.py | tee ../../results/$(PROJECT).txt

clean:
	rm -f test_video_1080p30.mp4 *.txt
