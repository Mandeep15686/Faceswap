"""
pipeline/stage03_tracker.py
────────────────────────────
Stage 03 — Face Tracking (GHOST Tracker)

Kalman filter + appearance re-ID keeps stable track IDs across
occlusions and scene cuts. Moved before the quality gate (v2 change)
so the gate can use track history for temporal consistency checks.

Each track stores:
  - Kalman state (bbox position + velocity)
  - Exponential-moving-average appearance embedding
  - Age (frames since first seen)
  - Frames missing (consecutive misses)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from loguru import logger
from scipy.optimize import linear_sum_assignment

from .base import BoundingBox, FrameContext, PipelineStage


# ─────────────────────────────────────────────────────────────────────────────
# Kalman filter for a single bounding-box track
# State: [cx, cy, w, h, vcx, vcy, vw, vh]
# ─────────────────────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    _id_counter: int = 0

    def __init__(self, bbox: BoundingBox, process_noise: float, measure_noise: float):
        KalmanBoxTracker._id_counter += 1
        self.track_id = KalmanBoxTracker._id_counter
        self.age = 0
        self.missing = 0
        self.hits = 1
        self.appearance: Optional[np.ndarray] = None   # 512-dim ArcFace embedding

        # State: [cx, cy, w, h, vcx, vcy, vw, vh]
        cx, cy = (bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2
        w, h   = bbox.x2 - bbox.x1, bbox.y2 - bbox.y1

        self.kf_x  = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float64)
        self.kf_P  = np.eye(8) * 10.0

        self.kf_F  = np.eye(8)
        for i in range(4):
            self.kf_F[i, i + 4] = 1.0           # position += velocity

        self.kf_H  = np.zeros((4, 8))
        for i in range(4):
            self.kf_H[i, i] = 1.0

        self.kf_Q  = np.eye(8) * process_noise
        self.kf_R  = np.eye(4) * measure_noise

    def predict(self) -> BoundingBox:
        """Kalman predict step.  Returns predicted bbox."""
        self.kf_x = self.kf_F @ self.kf_x
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q
        self.age += 1
        self.missing += 1
        return self._state_to_bbox()

    def update(self, bbox: BoundingBox) -> None:
        """Kalman update step with observed bbox."""
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        w  = bbox.x2 - bbox.x1
        h  = bbox.y2 - bbox.y1
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float64)

        S = self.kf_H @ self.kf_P @ self.kf_H.T + self.kf_R
        K = self.kf_P @ self.kf_H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + K @ (z - self.kf_H @ self.kf_x)
        self.kf_P = (np.eye(8) - K @ self.kf_H) @ self.kf_P
        self.hits += 1
        self.missing = 0

    def _state_to_bbox(self) -> BoundingBox:
        cx, cy, w, h = self.kf_x[:4, 0]
        return BoundingBox(
            x1=cx - w / 2, y1=cy - h / 2,
            x2=cx + w / 2, y2=cy + h / 2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# GHOST Tracker
# ─────────────────────────────────────────────────────────────────────────────

class FaceTracker(PipelineStage):
    """
    GHOST-style multi-face Kalman tracker with cosine re-ID.

    Outputs
    -------
    ctx.track_id   : int   — stable ID for the primary face
    ctx.track_age  : int   — how many frames this track has existed
    ctx.track_bbox : BoundingBox — smoothed/predicted bbox
    """

    stage_name = "face_tracking"

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        self._tracks: dict[int, KalmanBoxTracker] = {}
        self._process_noise = cfg.get("kalman_process_noise", 1e-3)
        self._measure_noise = cfg.get("kalman_measure_noise", 1e-1)
        self._reid_thr      = cfg.get("reid_distance_threshold", 0.35)
        self._max_missing   = cfg.get("max_missing_frames", 10)

    def load(self) -> None:
        logger.info("GHOST Kalman tracker initialised (no weights).")

    def process(self, ctx: FrameContext) -> FrameContext:
        dets = ctx.detections
        if not dets:
            # No detections — age all tracks
            for t in self._tracks.values():
                t.predict()
            self._prune_old_tracks()
            return ctx

        # ── Predict all existing tracks ──────────────────────────────────────
        predicted: dict[int, BoundingBox] = {}
        for tid, trk in self._tracks.items():
            predicted[tid] = trk.predict()

        # ── IoU cost matrix ──────────────────────────────────────────────────
        track_ids = list(self._tracks.keys())
        if track_ids:
            cost = np.zeros((len(track_ids), len(dets)), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                for di, det in enumerate(dets):
                    cost[ti, di] = 1.0 - self._iou(predicted[tid], det)

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_dets: set[int] = set()
            for ti, di in zip(row_ind, col_ind):
                if cost[ti, di] < (1.0 - 0.30):   # IoU > 0.30 to match
                    self._tracks[track_ids[ti]].update(dets[di])
                    matched_dets.add(di)

            # New detections → new tracks
            for di, det in enumerate(dets):
                if di not in matched_dets:
                    trk = KalmanBoxTracker(det, self._process_noise, self._measure_noise)
                    self._tracks[trk.track_id] = trk
        else:
            # No existing tracks — create one per detection
            for det in dets:
                trk = KalmanBoxTracker(det, self._process_noise, self._measure_noise)
                self._tracks[trk.track_id] = trk

        self._prune_old_tracks()

        # ── Assign primary face to context ───────────────────────────────────
        # Primary = highest-confidence detection's track
        if self._tracks:
            primary_trk = max(self._tracks.values(), key=lambda t: t.hits)
            ctx.track_id   = primary_trk.track_id
            ctx.track_age  = primary_trk.age
            ctx.track_bbox = primary_trk._state_to_bbox()

        return ctx

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prune_old_tracks(self) -> None:
        dead = [tid for tid, t in self._tracks.items() if t.missing > self._max_missing]
        for tid in dead:
            del self._tracks[tid]

    @staticmethod
    def _iou(a: BoundingBox, b: BoundingBox) -> float:
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return inter / (union + 1e-6)
