# Project 09 — NVDEC Video Pipeline

Compares CPU vs NVDEC video decode on the GB10, with optional GPU-side preprocessing.

## Build & run

Requires the `3d_recon` conda environment with PyTorch, OpenCV, cvcuda, and ffmpeg.

```bash
cd projects/09_nvdec_pipeline
make          # shows run command
make test     # generate a synthetic 1080p30 clip and benchmark all pipelines
make run      # alias for make test
make clean    # remove generated video
```

## Pipelines benchmarked

1. **OpenCV CPU decode + CPU preprocess**
2. **FFmpeg CPU decode + CPU preprocess**
3. **FFmpeg NVDEC decode + CPU preprocess**
4. **FFmpeg NVDEC decode + GPU preprocess (dummy)**

## Key findings

- CPU decode on the Grace cores can hit ~186–224 FPS for a short 1080p30 clip.
- NVDEC + CPU RGB conversion is slower (~130 FPS) because NV12→RGB on the CPU undoes the decode benefit.
- NVDEC is fastest when frames stay GPU-resident for cvcuda/preprocessing.
- GB10 has one NVDEC engine, so multi-stream decode is time-sliced.
