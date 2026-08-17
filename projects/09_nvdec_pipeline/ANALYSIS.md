# Project 09: NVDEC Video Pipeline — Analysis

## Results (5s, 150 frames, 1920×1080@30)

| Pipeline | FPS | Total (s) | Decode (ms) | Preprocess (ms) | Inference (ms) |
|----------|-----|-----------|-------------|-----------------|----------------|
| OpenCV CPU decode | 224.2 | 0.67 | 1.19 | 3.03 | 0.23 |
| FFmpeg CPU decode | 186.5 | 0.80 | 2.94 | 1.92 | 0.07 |
| NVDEC + CPU preprocess | 130.6 | 1.15 | 3.21 | 2.00 | 0.13 |
| NVDEC + GPU preprocess (dummy) | 230.8 | 0.65 | 2.73 | 0.12 | 0.07 |

## Observations

1. **CPU decode is very fast for this short synthetic clip.**
   - 186–224 FPS for CPU decode means the CPU is not the bottleneck for a 5s clip.
   - OpenCV may be using ffmpeg's optimized CPU decoder internally.

2. **NVDEC + CPU preprocess is slower (130 FPS).**
   - The decoded format from `h264_cuvid` is NV12.
   - Converting NV12 → RGB on the CPU is expensive and undoes the NVDEC benefit.
   - The CPU roundtrip (GPU→CPU for format conversion, then CPU→GPU for tensor)
     adds latency.

3. **NVDEC + GPU preprocess (dummy) is fastest (230 FPS).**
   - This is a *placeholder* GPU preprocess that avoids format conversion.
   - Real cvcuda-based NV12 → RGB → resize → normalize would be similar in
     avoiding the CPU roundtrip.
   - The key lesson: **NVDEC only wins if the frame stays on the GPU.**

## Why NVDEC Was Not Faster Here

- The clip is short and fully cached by the filesystem.
- CPU ffmpeg can decode 1080p30 at 186+ FPS on the Grace CPU cores.
- Moving NV12 from NVDEC to CPU and converting to RGB is a bottleneck.
- For longer real-world clinical videos with NVMe reads and random I/O, NVDEC
  should reduce CPU load and allow the CPU to do other work.

## GB10 Has 1 NVDEC Engine

- The GB10 SoC has a single NVDEC engine.
- Multiple concurrent decode streams will be time-sliced.
- For multi-stream workloads, CPU decode may actually scale better because
  there are many Grace CPU cores.

## Endosight Implications

- The Endosight pipeline should use NVDEC for the **initial video decode** and
  keep frames in GPU memory for preprocessing (resize, FOV crop, blur).
- The `browser_video.py` workflow uses `h264_nvenc` for output. NVDEC
  (`h264_cuvid`) is available but output should stay on GPU for cvcuda ops.
- PyNvVideoCodec (exp env) or `VideoProcessingFramework` would give zero-copy
  GPU decode without the `h264_cuvid` CPU roundtrip.

## Recommendations

1. **Use NVDEC when the downstream steps are GPU-resident.**
2. **Do not download NVDEC frames to CPU for RGB conversion.**
3. **Use cvcuda or a CUDA kernel to convert NV12 → RGB on the GPU.**
4. **For multi-stream ingest, benchmark CPU vs NVDEC with real clinical videos.**
5. **For short cached clips, CPU decode is often sufficient.**
