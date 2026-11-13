"""Simple centroid-based face tracker.

Assigns stable temporary track IDs to detections across consecutive frames
using nearest-centroid matching with IoU gating. No external dependencies.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("cantforget.faces.tracker")


class FaceTrack:
    """A single tracked face."""

    def __init__(self, track_id: int, bbox: tuple[int, int, int, int]) -> None:
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.hits = 1
        self.misses = 0
        self.age = 1
        # Index of the detection this track matched in the latest update()
        # call, or None if it coasted (no detection this frame).
        self.det_index: Optional[int] = None
        self._update_centroid(bbox)

    def _update_centroid(self, bbox: tuple[int, int, int, int]) -> None:
        x, y, w, h = bbox
        self.centroid = (x + w / 2.0, y + h / 2.0)

    def update(self, bbox: tuple[int, int, int, int], det_index: Optional[int] = None) -> None:
        self.bbox = bbox
        self.det_index = det_index
        self.hits += 1
        self.misses = 0
        self.age += 1
        self._update_centroid(bbox)

    def mark_missed(self) -> None:
        self.det_index = None
        self.misses += 1
        self.age += 1

    @property
    def center(self) -> tuple[float, float]:
        return self.centroid


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    if right <= left or bottom <= top:
        return 0.0

    inter = (right - left) * (bottom - top)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _centroid_distance(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ca = (ax + aw / 2.0, ay + ah / 2.0)
    cb = (bx + bw / 2.0, by + bh / 2.0)
    dx = ca[0] - cb[0]
    dy = ca[1] - cb[1]
    return (dx * dx + dy * dy) ** 0.5


class CentroidTracker:
    """Tracks faces across frames using nearest-centroid + IoU matching.

    Parameters
    ----------
    max_disappeared:
        Number of consecutive misses before a track is removed.
    max_distance:
        Maximum centroid distance (pixels) to consider a match.
    iou_threshold:
        Minimum IoU to accept a match (gates centroid-only matches).
    """

    def __init__(
        self,
        max_disappeared: int = 15,
        max_distance: float = 80.0,
        iou_threshold: float = 0.25,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold
        self._tracks: dict[int, FaceTrack] = {}
        self._next_id = 0
        self._disappeared: dict[int, int] = {}

    def update(
        self, detections: list[tuple[int, int, int, int]]
    ) -> list[dict]:
        """Match detections to existing tracks and return updated tracks.

        Each detection is (x, y, w, h). Returns a list of dicts:
        {track_id, x, y, w, h, hits, age, det_index}. ``det_index`` is the
        index into ``detections`` this track matched (None if it coasted),
        so callers can attach per-detection data without box-coordinate lookups.
        """
        # No detections: mark all existing tracks as missed.
        if not detections:
            for track_id in list(self._tracks.keys()):
                self._tracks[track_id].mark_missed()
                self._disappeared[track_id] = (
                    self._disappeared.get(track_id, 0) + 1
                )
            self._prune()
            return self._serialize()

        # No existing tracks: register all detections as new.
        if not self._tracks:
            for di, bbox in enumerate(detections):
                self._register(bbox, di)
            return self._serialize()

        # Compute matching between existing tracks and new detections.
        track_ids = list(self._tracks.keys())
        track_boxes = [self._tracks[tid].bbox for tid in track_ids]

        # Greedy nearest-neighbor with IoU gating.
        used_track_idx: set[int] = set()
        used_det_idx: set[int] = set()

        # Build candidate pairs sorted by distance.
        pairs: list[tuple[float, int, int]] = []
        for ti, tbox in enumerate(track_boxes):
            for di, dbox in enumerate(detections):
                iou = _iou(tbox, dbox)
                dist = _centroid_distance(tbox, dbox)
                # Accept only if close enough AND sufficient IoU.
                if dist <= self.max_distance and iou >= self.iou_threshold:
                    pairs.append((dist, ti, di))

        pairs.sort(key=lambda p: p[0])

        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()
        for dist, ti, di in pairs:
            if ti in matched_track_indices or di in matched_det_indices:
                continue
            matched_track_indices.add(ti)
            matched_det_indices.add(di)
            track_id = track_ids[ti]
            self._tracks[track_id].update(detections[di], di)
            self._disappeared[track_id] = 0

        # Unmatched detections -> new tracks.
        for di, bbox in enumerate(detections):
            if di not in matched_det_indices:
                self._register(bbox, di)

        # Unmatched existing tracks -> missed.
        for ti, track_id in enumerate(track_ids):
            if ti not in matched_track_indices:
                self._tracks[track_id].mark_missed()
                self._disappeared[track_id] = (
                    self._disappeared.get(track_id, 0) + 1
                )

        self._prune()
        return self._serialize()

    def _register(self, bbox: tuple[int, int, int, int], det_index: Optional[int] = None) -> None:
        track_id = self._next_id
        self._next_id += 1
        track = FaceTrack(track_id, bbox)
        track.det_index = det_index
        self._tracks[track_id] = track
        self._disappeared[track_id] = 0

    def _prune(self) -> None:
        gone = [
            tid
            for tid, count in self._disappeared.items()
            if count > self.max_disappeared
        ]
        for tid in gone:
            self._tracks.pop(tid, None)
            self._disappeared.pop(tid, None)
            logger.debug("Track %d pruned after %d misses", tid, self.max_disappeared)

    def _serialize(self) -> list[dict]:
        out: list[dict] = []
        for track in self._tracks.values():
            x, y, w, h = track.bbox
            out.append(
                {
                    "track_id": track.track_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "hits": track.hits,
                    "age": track.age,
                    "det_index": track.det_index,
                }
            )
        return out

    @property
    def active_count(self) -> int:
        return len(self._tracks)
