import numpy as np

from backend.app.faces.detector import FaceDetector
from backend.app.faces.tracker import CentroidTracker

MODEL = "data/models/face_detection_yunet.onnx"


def _row(x, y, w, h, score):
    # YuNet layout: box(4) + 5 landmarks(10) + score
    lm = [x + 10, y + 10, x + w - 10, y + 10, x + w / 2, y + h / 2, x + 12, y + h - 8, x + w - 12, y + h - 8]
    return np.array([x, y, w, h, *lm, score], dtype=np.float32)


class _FakeYN:
    def __init__(self, rows):
        self.rows = np.array(rows, dtype=np.float32)

    def setInputSize(self, size):
        self.size = size

    def detect(self, frame):
        return 1, self.rows


def test_real_model_blank_frame_returns_no_faces():
    det = FaceDetector(MODEL)
    assert det.detect(np.zeros((480, 640, 3), np.uint8)) == []


def test_detector_reads_score_at_index_14_and_landmarks_4_to_13():
    det = FaceDetector(MODEL)
    det._detector = _FakeYN([_row(100, 80, 120, 140, 0.93)])
    (d,) = det.detect(np.zeros((480, 640, 3), np.uint8))
    assert d["confidence"] == 0.93  # previously read the right-eye x by mistake
    assert d["landmarks"][0] == [110, 90]
    assert d["landmarks"][4] == [208, 212]
    assert d["row"].shape == (15,)


def test_tracker_reports_det_index_and_keeps_ids():
    t = CentroidTracker()
    out = t.update([(100, 100, 80, 80), (300, 100, 80, 80)])
    ids = {o["det_index"]: o["track_id"] for o in out}
    out2 = t.update([(304, 102, 80, 80), (102, 101, 80, 80)])  # detections swapped in order
    ids2 = {o["det_index"]: o["track_id"] for o in out2}
    assert ids2[0] == ids[1] and ids2[1] == ids[0]


def test_tracker_coasting_track_has_no_det_index():
    t = CentroidTracker()
    t.update([(100, 100, 80, 80)])
    (o,) = t.update([])
    assert o["det_index"] is None
