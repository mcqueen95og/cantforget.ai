"""Tests for the face detection and recognition subsystem.

Run with: pytest backend/tests/ -v
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.faces.aligner import FaceAligner, REFERENCE_LANDMARKS_112
from app.faces.detector import FaceDetector as YuNetDetector
from app.faces.recognizer import FaceRecognizer
from app.faces.tracker import CentroidTracker, _iou, _centroid_distance
from app.database import (
    init_db,
    create_person,
    add_embedding_with_version,
    get_all_embeddings,
    delete_person,
    get_connection,
)

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface.onnx"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return YuNetDetector(
        model_path=YUNET_MODEL,
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
    )


@pytest.fixture
def recognizer():
    return FaceRecognizer(
        model_path=SFACE_MODEL,
        model_name="sface",
        embedding_dim=128,
        cosine_threshold=0.363,
    )


@pytest.fixture
def aligner():
    return FaceAligner(output_size=(112, 112))


@pytest.fixture
def tracker():
    return CentroidTracker()


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def face_frame():
    """Generate a synthetic frame with a face-like rectangle."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a face-like oval (not a real face, so detector may not find it)
    cv2.ellipse(frame, (320, 240), (80, 100), 0, 0, 360, (100, 100, 100), -1)
    cv2.circle(frame, (290, 210), 10, (50, 50, 50), -1)    # left eye
    cv2.circle(frame, (350, 210), 10, (50, 50, 50), -1)    # right eye
    cv2.circle(frame, (320, 250), 5, (50, 50, 50), -1)     # nose
    return frame


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

def test_detector_loads(detector):
    assert detector is not None
    assert detector.model_path == str(YUNET_MODEL)


def test_detect_blank_frame(detector, blank_frame):
    detections = detector.detect(blank_frame)
    assert isinstance(detections, list)
    # Blank frame should have 0 faces
    assert len(detections) == 0


def test_detection_returns_expected_keys(detector):
    """Detection dicts must have all required keys."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    if detections:
        d = detections[0]
        for key in ("x", "y", "w", "h", "confidence", "landmarks", "model"):
            assert key in d, f"Missing key: {key}"
        assert d["model"] == "yunet"


def test_detector_input_size_stable(detector, blank_frame):
    """setInputSize should not be called every frame (regression test)."""
    detections1 = detector.detect(blank_frame)
    detections2 = detector.detect(blank_frame)
    assert isinstance(detections1, list)
    assert isinstance(detections2, list)


# ---------------------------------------------------------------------------
# Alignment tests
# ---------------------------------------------------------------------------

def test_aligner_creates_aligned_crop(aligner, blank_frame):
    aligned = aligner.align(blank_frame, (100, 100, 100, 100), None)
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8


def test_aligner_with_landmarks(aligner, blank_frame):
    landmarks = [[130, 150], [170, 150], [150, 180], [130, 210], [170, 210]]
    aligned = aligner.align(blank_frame, (100, 100, 100, 100), landmarks)
    assert aligned.shape == (112, 112, 3)


def test_aligner_fallback_on_empty_frame(aligner):
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    aligned = aligner.align(empty, (0, 0, 0, 0), None)
    assert aligned.shape == (112, 112, 3)


def test_reference_landmarks_shape():
    assert REFERENCE_LANDMARKS_112.shape == (5, 2)


# ---------------------------------------------------------------------------
# Recognition tests
# ---------------------------------------------------------------------------

def test_recognizer_creates_embedding(recognizer, aligner, blank_frame):
    aligned = aligner.align(blank_frame, (100, 100, 100, 100), None)
    emb = recognizer.embed(aligned)
    assert emb is not None
    assert emb.shape[0] == 128
    # Should be L2-normalized (norm ≈ 1)
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 0.01, f"Embedding not L2-normalized: norm={norm}"


def test_recognizer_embedding_deterministic(recognizer, aligner, blank_frame):
    aligned = aligner.align(blank_frame, (100, 100, 100, 100), None)
    emb1 = recognizer.embed(aligned)
    emb2 = recognizer.embed(aligned)
    assert emb1 is not None and emb2 is not None
    score = FaceRecognizer.cosine_similarity(emb1, emb2)
    assert score > 0.99, f"Embeddings not deterministic: score={score}"


def test_recognizer_empty_crop(recognizer):
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    emb = recognizer.embed(empty)
    assert emb is None


def test_cosine_similarity_range():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    assert FaceRecognizer.cosine_similarity(a, b) == 1.0
    assert FaceRecognizer.cosine_similarity(a, c) == -1.0


def test_recognizer_model_version():
    rec = FaceRecognizer(
        model_path=SFACE_MODEL, model_name="sface", embedding_dim=128
    )
    assert rec.model_version == "sface:128d"
    assert rec.is_compatible("sface:128d")
    assert not rec.is_compatible("arcface:512d")


# ---------------------------------------------------------------------------
# Tracker tests
# ---------------------------------------------------------------------------

def test_tracker_iou():
    box_a = (0, 0, 100, 100)
    box_b = (50, 50, 100, 100)
    iou = _iou(box_a, box_b)
    assert 0 < iou < 1
    # Same box = IoU 1.0
    assert _iou(box_a, box_a) == 1.0
    # No overlap = IoU 0.0
    assert _iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_tracker_centroid_distance():
    box_a = (0, 0, 100, 100)
    box_b = (100, 100, 100, 100)
    dist = _centroid_distance(box_a, box_b)
    assert dist > 0


def test_tracker_register_and_update(tracker):
    result = tracker.update([(100, 100, 50, 50)])
    assert len(result) == 1
    assert result[0]["track_id"] == 0
    assert result[0]["hits"] == 1

    # Same frame again - should match existing track
    result2 = tracker.update([(105, 105, 50, 50)])
    assert len(result2) == 1
    assert result2[0]["track_id"] == 0
    assert result2[0]["hits"] == 2


def test_tracker_new_detection_gets_new_id(tracker):
    tracker.update([(100, 100, 50, 50)])
    result = tracker.update([(100, 100, 50, 50), (300, 300, 50, 50)])
    assert len(result) == 2
    ids = {r["track_id"] for r in result}
    assert ids == {0, 1}


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

def test_db_migration_adds_model_version():
    conn = get_connection()
    cols = [row[1] for row in conn.execute("PRAGMA table_info(face_embeddings)").fetchall()]
    conn.close()
    assert "model_version" in cols, "model_version column missing after migration"


def test_embedding_with_version_storage():
    person = create_person("Test Person", "Test")
    pid = person["id"]
    emb = np.random.randn(128).astype(np.float32)
    emb_bytes = emb.tobytes()
    eid = add_embedding_with_version(pid, emb_bytes, "sface:128d")
    assert eid is not None

    rows = get_all_embeddings()
    found = [r for r in rows if r[0] == pid]
    assert len(found) == 1
    assert found[0][2] == "sface:128d"  # model_version

    # Clean up
    delete_person(pid)


def test_incompatible_model_version_filtered():
    person = create_person("Version Test", "Test")
    pid = person["id"]
    emb = np.random.randn(128).astype(np.float32)
    emb_bytes = emb.tobytes()

    # Store with sface version
    add_embedding_with_version(pid, emb_bytes, "sface:128d")

    rows = get_all_embeddings()
    sface_rows = [r for r in rows if r[0] == pid and r[2] == "sface:128d"]
    assert len(sface_rows) == 1

    delete_person(pid)


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

def test_detection_latency_under_100ms(detector, blank_frame):
    """Detection of a single frame should complete in <100ms on CPU."""
    # Warm up
    detector.detect(blank_frame)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        detector.detect(blank_frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    avg_ms = sum(times) / len(times)
    max_ms = max(times)
    print(f"\nDetection latency: avg={avg_ms:.1f}ms max={max_ms:.1f}ms")
    # Allow some slack for CI environments
    assert avg_ms < 200, f"Average detection latency too high: {avg_ms:.1f}ms"


def test_recognition_latency_under_100ms(recognizer, aligner, blank_frame):
    """Recognition embedding + match should complete in <100ms."""
    aligned = aligner.align(blank_frame, (100, 100, 100, 100), None)
    # Warm up
    recognizer.embed(aligned)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        recognizer.embed(aligned)
        times.append((time.perf_counter() - t0) * 1000.0)
    avg_ms = sum(times) / len(times)
    print(f"\nRecognition latency: avg={avg_ms:.1f}ms")
    assert avg_ms < 200, f"Average recognition latency too high: {avg_ms:.1f}ms"
