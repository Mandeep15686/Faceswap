"""
pipeline/quality_gate.py
─────────────────────────
Quality Gate — Reject frames that would produce bad swaps.

Runs after Stage 03 (tracking) so it can use track history.
Sets ctx.gate_passed = False for frames that should pass through
the original (un-swapped) pixels.

Rejection criteria
──────────────────
• Pose yaw or pitch > threshold (extreme angle)
• Motion blur score below threshold (blurry frame)
• Detection confidence below threshold
• Track age < minimum (unstable new track)
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from .base import FrameContext, PipelineStage


class QualityGate(PipelineStage):
    stage_name = "quality_gate"

    def load(self) -> None:
        self._max_yaw    = self.cfg.get("max_yaw_deg", 40.0)
        self._max_pitch  = self.cfg.get("max_pitch_deg", 40.0)
        self._min_blur   = self.cfg.get("min_blur_score", 0.15)
        self._min_age    = self.cfg.get("min_track_age", 3)
        self._min_conf   = self.cfg.get("min_confidence", 0.90)
        logger.info(
            f"Quality Gate: yaw<{self._max_yaw}° pitch<{self._max_pitch}° "
            f"blur>{self._min_blur} age>{self._min_age} conf>{self._min_conf}"
        )

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.gate_passed = True

        # ── No detection → reject ──────────────────────────────────────────
        if not ctx.detections:
            ctx.gate_passed = False
            ctx.gate_reason = "no_detection"
            return ctx

        primary_det = ctx.detections[0]

        # ── Low confidence → reject ───────────────────────────────────────
        if primary_det.confidence < self._min_conf:
            ctx.gate_passed = False
            ctx.gate_reason = f"low_confidence={primary_det.confidence:.2f}"
            return ctx

        # ── Track too new → reject ────────────────────────────────────────
        if ctx.track_age < self._min_age:
            ctx.gate_passed = False
            ctx.gate_reason = f"track_age={ctx.track_age}"
            return ctx

        # ── Motion blur ───────────────────────────────────────────────────
        bbox = ctx.track_bbox or primary_det
        face_crop = self._crop(ctx.image, bbox)
        blur = self._blur_score(face_crop)
        if blur < self._min_blur:
            ctx.gate_passed = False
            ctx.gate_reason = f"blur={blur:.3f}"
            return ctx

        # ── Pose check (requires geometry from Stage 04, skip if not yet set)
        if ctx.geometry is not None:
            yaw, pitch, _ = ctx.geometry.pose
            if abs(yaw) > self._max_yaw or abs(pitch) > self._max_pitch:
                ctx.gate_passed = False
                ctx.gate_reason = f"pose yaw={yaw:.1f} pitch={pitch:.1f}"
                return ctx

        return ctx

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _crop(image: np.ndarray, bbox) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
        x2, y2 = min(w, int(bbox.x2)), min(h, int(bbox.y2))
        return image[y1:y2, x1:x2]

    @staticmethod
    def _blur_score(crop: np.ndarray) -> float:
        """
        Normalised Laplacian variance in [0, 1].
        Low = blurry, High = sharp.
        """
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalise with empirical 1000 as "very sharp"
        return float(min(lap / 1000.0, 1.0))
