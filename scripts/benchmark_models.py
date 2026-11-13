#!/usr/bin/env python3
"""Benchmark face detection and recognition models on CPU.

Measures: latency, CPU usage, peak RAM, FPS.

Usage: python scripts/benchmark_models.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

MODELS_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface.onnx"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.faces.detector import FaceDetector as YuNetDetector
from backend.app.faces.recognizer import FaceRecognizer
from backend.app.faces.aligner import FaceAligner


def make_frame(w=640, h=480):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def get_process_mb():
    if psutil is None:
        return 0.0
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def bench_detector(detector, frame, runs=50):
    # Warm up
    for _ in range(5):
        detector.detect(frame)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        detector.detect(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "fps": 1000.0 / (sum(times) / len(times)),
    }


def bench_recognizer(recognizer, aligner, frame, runs=50):
    bbox = (frame.shape[1] // 4, frame.shape[0] // 4, frame.shape[1] // 2, frame.shape[0] // 2)
    aligned = aligner.align(frame, bbox, None)
    # Warm up
    for _ in range(5):
        recognizer.embed(aligned)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        recognizer.embed(aligned)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "fps": 1000.0 / (sum(times) / len(times)),
    }


def main():
    print("=" * 60)
    print("ForgetMeNot Face Pipeline Benchmark")
    print("=" * 60)

    ram_before = get_process_mb()
    print(f"\nRAM before: {ram_before:.1f} MB")

    # --- Detector ---
    print("\n--- YuNet Face Detector ---")
    det = YuNetDetector(
        model_path=YUNET_MODEL,
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
    )
    frame = make_frame(640, 480)
    d_res = bench_detector(det, frame)
    print(f"  Avg latency: {d_res['avg_ms']:.1f} ms")
    print(f"  Min/Max:     {d_res['min_ms']:.1f} / {d_res['max_ms']:.1f} ms")
    print(f"  Single-run FPS: {d_res['fps']:.1f}")

    # --- Recognizer ---
    print("\n--- SFace Face Recognizer (128-d) ---")
    rec = FaceRecognizer(
        model_path=SFACE_MODEL,
        model_name="sface",
        embedding_dim=128,
        cosine_threshold=0.363,
    )
    aligner = FaceAligner(output_size=(112, 112))
    r_res = bench_recognizer(rec, aligner, frame)
    print(f"  Avg latency: {r_res['avg_ms']:.1f} ms")
    print(f"  Min/Max:     {r_res['min_ms']:.1f} / {r_res['max_ms']:.1f} ms")
    print(f"  Single-run FPS: {r_res['fps']:.1f}")

    # --- Combined pipeline estimate ---
    print("\n--- Combined Pipeline Estimate (per frame, 1 face) ---")
    combined_ms = d_res["avg_ms"] + r_res["avg_ms"]
    print(f"  Detection + Recognition: {combined_ms:.1f} ms")
    print(f"  Theoretical max FPS (1 face): {1000.0 / combined_ms:.1f}")

    ram_after = get_process_mb()
    print(f"\nRAM after: {ram_after:.1f} MB (delta: {ram_after - ram_before:+.1f} MB)")

    # Recommendations
    print("\n--- Recommendations ---")
    if d_res["avg_ms"] > 100:
        print("  [WARN] Detection latency >100ms. Consider smaller input size or frame skipping.")
    else:
        print("  [OK] Detection latency <100ms.")
    if r_res["avg_ms"] > 100:
        print("  [WARN] Recognition latency >100ms. Consider caching embeddings or frame skipping.")
    else:
        print("  [OK] Recognition latency <100ms.")
    if ram_after - ram_before > 1000:
        print("  [WARN] Model memory usage >1GB. Monitor on 8GB system.")
    else:
        print("  [OK] Model memory usage within budget.")


if __name__ == "__main__":
    main()
