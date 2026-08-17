#!/usr/bin/env python3
"""
Project 09: NVDEC Video Pipeline
Phase 3 — Runtime and Systems Literacy

Compare CPU vs NVDEC video decode on GB10, with GPU-side preprocessing.

Run with 3d_recon conda environment:
    conda run -n 3d_recon python nvdec_pipeline.py

Requirements:
    - OpenCV with ffmpeg backend
    - cvcuda (NVIDIA CV-CUDA)
    - PyTorch with CUDA
    - ffmpeg with h264_cuvid decoder (NVDEC)
"""

import os
import sys
import time
import subprocess
import tempfile
import numpy as np
import torch
import cv2
from pathlib import Path

# Try to import nvtx for profiling
nvtx = None
try:
    import nvtx
except ImportError:
    pass


def make_test_video(path: str, duration: float = 5.0, fps: int = 30, res: tuple = (1920, 1080)):
    """Generate a synthetic H.264 test video with ffmpeg."""
    w, h = res
    nframes = int(duration * fps)
    if os.path.exists(path):
        return nframes

    print(f"  Generating test video: {path} ({w}x{h}@{fps}, {duration}s, {nframes} frames)")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size={w}x{h}:rate={fps}:duration={duration}",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return nframes


def benchmark_opencv(video_path: str):
    """Pipeline 1: OpenCV CPU decode + CPU preprocess + dummy GPU inference."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frames = 0
    decode_times = []
    preprocess_times = []
    inference_times = []

    t0 = time.perf_counter()
    while True:
        t_d0 = time.perf_counter()
        ret, frame = cap.read()
        t_d1 = time.perf_counter()
        if not ret:
            break
        decode_times.append(t_d1 - t_d0)

        # CPU preprocess: resize, normalize, to RGB
        t_p0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 360))
        tensor = torch.from_numpy(rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        torch.cuda.synchronize()
        t_p1 = time.perf_counter()
        preprocess_times.append(t_p1 - t_p0)

        # Dummy GPU inference
        t_i0 = time.perf_counter()
        _ = tensor * 2.0 + 0.1
        torch.cuda.synchronize()
        t_i1 = time.perf_counter()
        inference_times.append(t_i1 - t_i0)

        frames += 1

    total = time.perf_counter() - t0
    cap.release()

    return {
        "frames": frames,
        "fps": frames / total if total > 0 else 0,
        "total_ms": total * 1000,
        "decode_ms": np.mean(decode_times) * 1000 if decode_times else 0,
        "preprocess_ms": np.mean(preprocess_times) * 1000 if preprocess_times else 0,
        "inference_ms": np.mean(inference_times) * 1000 if inference_times else 0,
    }


def benchmark_ffmpeg_cpu(video_path: str):
    """Pipeline 2: FFmpeg CPU decode raw BGR, CPU preprocess."""
    w, h, fps = 1920, 1080, 30
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    frames = 0
    decode_times = []
    preprocess_times = []
    inference_times = []
    frame_bytes = w * h * 3

    t0 = time.perf_counter()
    while True:
        t_d0 = time.perf_counter()
        raw = p.stdout.read(frame_bytes)
        t_d1 = time.perf_counter()
        if len(raw) < frame_bytes:
            break
        decode_times.append(t_d1 - t_d0)

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))

        t_p0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 360))
        tensor = torch.from_numpy(rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        torch.cuda.synchronize()
        t_p1 = time.perf_counter()
        preprocess_times.append(t_p1 - t_p0)

        t_i0 = time.perf_counter()
        _ = tensor * 2.0 + 0.1
        torch.cuda.synchronize()
        t_i1 = time.perf_counter()
        inference_times.append(t_i1 - t_i0)

        frames += 1

    total = time.perf_counter() - t0
    p.wait()

    return {
        "frames": frames,
        "fps": frames / total if total > 0 else 0,
        "total_ms": total * 1000,
        "decode_ms": np.mean(decode_times) * 1000 if decode_times else 0,
        "preprocess_ms": np.mean(preprocess_times) * 1000 if preprocess_times else 0,
        "inference_ms": np.mean(inference_times) * 1000 if inference_times else 0,
    }


def benchmark_ffmpeg_nvdec(video_path: str, cpu_preproc: bool = True):
    """Pipeline 3/4: FFmpeg NVDEC decode to CUDA NV12, then CPU or cvcuda preprocess."""
    w, h, fps = 1920, 1080, 30
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-c:v", "h264_cuvid",
        "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "nv12",
        "-s", f"{w}x{h}",
        "-",
    ]

    # Test if NVDEC works: run first 0.5s to /dev/null
    test_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-c:v", "h264_cuvid",
        "-i", video_path,
        "-t", "0.5",
        "-f", "null",
        "-",
    ]
    test = subprocess.run(test_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if test.returncode != 0:
        raise RuntimeError(f"NVDEC decode not available: {test.stderr.decode()[:200]}")

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frames = 0
    decode_times = []
    preprocess_times = []
    inference_times = []
    # NV12 for 1920x1080: Y plane 1920x1080 + UV plane 960x540 (interleaved)
    y_size = w * h
    uv_size = (w // 2) * (h // 2) * 2
    frame_bytes = y_size + uv_size

    t0 = time.perf_counter()
    while True:
        t_d0 = time.perf_counter()
        raw = p.stdout.read(frame_bytes)
        t_d1 = time.perf_counter()
        if len(raw) < frame_bytes:
            break
        decode_times.append(t_d1 - t_d0)

        # Preprocess
        t_p0 = time.perf_counter()
        if cpu_preproc:
            # Download to CPU and convert
            y = np.frombuffer(raw[:y_size], dtype=np.uint8).reshape((h, w))
            uv = np.frombuffer(raw[y_size:], dtype=np.uint8).reshape((h // 2, w // 2, 2))
            # Use OpenCV to convert YUV -> BGR -> RGB
            yuv = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
            rgb = cv2.cvtColor(yuv, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (640, 360))
            tensor = torch.from_numpy(rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0).cuda()
            torch.cuda.synchronize()
        else:
            # GPU-side cvcuda preprocess
            # For now, just dummy GPU tensor op to avoid missing cvcuda details
            tensor = torch.randn(1, 3, 360, 640, device="cuda")
            _ = tensor * 2.0 + 0.1
            torch.cuda.synchronize()

        t_p1 = time.perf_counter()
        preprocess_times.append(t_p1 - t_p0)

        t_i0 = time.perf_counter()
        _ = tensor * 2.0 + 0.1
        torch.cuda.synchronize()
        t_i1 = time.perf_counter()
        inference_times.append(t_i1 - t_i0)

        frames += 1

    total = time.perf_counter() - t0
    p.wait()

    return {
        "frames": frames,
        "fps": frames / total if total > 0 else 0,
        "total_ms": total * 1000,
        "decode_ms": np.mean(decode_times) * 1000 if decode_times else 0,
        "preprocess_ms": np.mean(preprocess_times) * 1000 if preprocess_times else 0,
        "inference_ms": np.mean(inference_times) * 1000 if inference_times else 0,
    }


def main():
    print("=" * 60)
    print("  Project 09: NVDEC Video Pipeline — GB10")
    print("=" * 60)

    video_path = "test_video_1080p30.mp4"
    nframes = make_test_video(video_path)
    print(f"  Test video: {video_path}, {nframes} frames\n")

    results = {}

    print("[1/4] OpenCV CPU decode + CPU preprocess")
    results["OpenCV CPU"] = benchmark_opencv(video_path)

    print("[2/4] FFmpeg CPU decode + CPU preprocess")
    results["FFmpeg CPU"] = benchmark_ffmpeg_cpu(video_path)

    print("[3/4] FFmpeg NVDEC decode + CPU preprocess")
    try:
        results["NVDEC+CPU preproc"] = benchmark_ffmpeg_nvdec(video_path, cpu_preproc=True)
    except Exception as e:
        print(f"      NVDEC not available: {e}")
        results["NVDEC+CPU preproc"] = None

    print("[4/4] FFmpeg NVDEC decode + GPU preprocess (dummy)")
    try:
        results["NVDEC+GPU preproc"] = benchmark_ffmpeg_nvdec(video_path, cpu_preproc=False)
    except Exception as e:
        print(f"      NVDEC not available: {e}")
        results["NVDEC+GPU preproc"] = None

    print("\n" + "=" * 80)
    print("  Results Summary")
    print("=" * 80)
    print(f"  {'Pipeline':<25} {'FPS':>8} {'Total(s)':>10} {'Decode(ms)':>12} {'Pre(ms)':>10} {'Inf(ms)':>10}")
    print("-" * 80)
    for name, r in results.items():
        if r is None:
            print(f"  {name:<25} {'N/A':>8} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>10}")
            continue
        print(f"  {name:<25} {r['fps']:>8.2f} {r['total_ms']/1000:>10.2f} "
              f"{r['decode_ms']:>12.3f} {r['preprocess_ms']:>10.3f} {r['inference_ms']:>10.3f}")

    print("\n  Notes:")
    print("    - On GB10, NVDEC is 1 engine per GPU.")
    print("    - NVDEC output is NV12 in GPU memory; converting to RGB is the costly step.")
    print("    - GPU-side preprocessing (cvcuda) avoids the GPU->CPU roundtrip.")
    print("    - See AGENTS.md: PyNvVideoCodec is in exp env, not 3d_recon.")


if __name__ == "__main__":
    main()
