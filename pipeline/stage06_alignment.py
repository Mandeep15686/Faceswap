"""
pipeline/stage06_alignment.py
──────────────────────────────
Stage 06 — Face Alignment (Similarity Transform)

5-point anchor → similarity transform.
Crops face to canonical 256×256 aligned space.
Inverse transform stored per-frame for exact re-projection back to
original image coordinates in Stage 14.
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from .base import AlignedFace, FrameContext, PipelineStage

# ─────────────────────────────────────────────────────────────────────────────
# Canonical 5-point template at 256×256
# Order: left-eye, right-eye, nose, left-mouth, right-mouth
# ─────────────────────────────────────────────────────────────────────────────
_CANONICAL_256 = np.array([
    [85.82,  98.61],   # left eye
    [170.18, 98.61],   # right eye
    [128.0,  152.15],  # nose
    [92.37,  196.19],  # left mouth
    [163.63, 196.19],  # right mouth
], dtype=np.float32)

# 5-point subset indices from 68-landmark set
_ANCHOR_INDICES_68 = [36, 45, 30, 48, 54]   # configurable in YAML


class FaceAligner(PipelineStage):
    stage_name = "face_alignment"

    def load(self) -> None:
        self._crop_size = self.cfg.get("crop_size", 256)
        self._anchor_idx = self.cfg.get("anchor_indices", _ANCHOR_INDICES_68)
        # Scale canonical points to configured crop size
        scale = self._crop_size / 256.0
        self._canonical = _CANONICAL_256 * scale
        logger.info(f"Face aligner: crop={self._crop_size}px, anchors={self._anchor_idx}")

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.gate_passed:
            return ctx

        # Prefer smoothed landmarks; fall back to raw
        if ctx.smoothed_landmarks is not None:
            landmarks = ctx.smoothed_landmarks
        elif ctx.geometry is not None:
            landmarks = ctx.geometry.landmarks_2d
        else:
            # Last resort: use bbox centre + approximate points
            landmarks = self._approx_landmarks_from_bbox(ctx)
            if landmarks is None:
                return ctx

        # Extract 5 anchor points
        anchors = np.array(
            [landmarks[i] for i in self._anchor_idx], dtype=np.float32
        )

        # Estimate similarity transform (rotation, scale, translation; no shear)
        M, _ = cv2.estimateAffinePartial2D(
            anchors,
            self._canonical,
            method=cv2.LMEDS,
        )
        if M is None:
            return ctx

        # Warp face into canonical space
        crop = cv2.warpAffine(
            ctx.image, M,
            (self._crop_size, self._crop_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Inverse transform for re-projection
        M_inv = cv2.invertAffineTransform(M)

        ctx.aligned = AlignedFace(
            crop=crop,
            inverse_transform=M_inv,
            landmarks_5pt=anchors,
        )
        return ctx

    @staticmethod
    def _approx_landmarks_from_bbox(ctx: FrameContext):
        """Fallback: derive 5 rough landmark positions from bbox."""
        bbox = ctx.track_bbox or (ctx.detections[0] if ctx.detections else None)
        if bbox is None:
            return None
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        w  = bbox.x2 - bbox.x1
        h  = bbox.y2 - bbox.y1
        return np.array([
            [cx - w * 0.20, cy - h * 0.10],   # left eye
            [cx + w * 0.20, cy - h * 0.10],   # right eye
            [cx,            cy            ],   # nose
            [cx - w * 0.14, cy + h * 0.22],   # left mouth
            [cx + w * 0.14, cy + h * 0.22],   # right mouth
        ], dtype=np.float32)
