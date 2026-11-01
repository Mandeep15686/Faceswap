"""
pipeline/stage05_smoother.py
─────────────────────────────
Stage 05 — Landmark Smoothing (One-Euro Filter)

Adaptive cutoff frequency per-landmark per-axis:
  • Low velocity → low cutoff → heavy smoothing (removes jitter at rest)
  • High velocity → high cutoff → light smoothing (stays responsive)

Applied to all 68 landmarks independently (136 scalars total).
Also applied to the rotation vector (yaw/pitch/roll).

Reference: Casiez, Roussel, Vogel (2012) "1€ Filter"
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from loguru import logger

from .base import FrameContext, PipelineStage


# ─────────────────────────────────────────────────────────────────────────────
# One-Euro filter (single scalar)
# ─────────────────────────────────────────────────────────────────────────────

class _OneEuro:
    def __init__(self, freq: float, min_cutoff: float, beta: float, d_cutoff: float):
        self._freq = freq
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

    def __call__(self, x: float, freq: Optional[float] = None) -> float:
        if freq is not None:
            self._freq = freq
        if self._x_prev is None:
            self._x_prev = x
            return x

        # Derivative filter
        dx = (x - self._x_prev) * self._freq
        a_d = self._alpha(self._d_cutoff, self._freq)
        self._dx_prev = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Dynamic cutoff
        cutoff = self._min_cutoff + self._beta * abs(self._dx_prev)
        a = self._alpha(cutoff, self._freq)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


# ─────────────────────────────────────────────────────────────────────────────
# Stage
# ─────────────────────────────────────────────────────────────────────────────

class LandmarkSmoother(PipelineStage):
    stage_name = "landmark_smoothing"

    def load(self) -> None:
        min_cutoff = self.cfg.get("min_cutoff", 0.004)
        beta       = self.cfg.get("beta", 0.7)
        d_cutoff   = self.cfg.get("d_cutoff", 1.0)
        self._freq = 24.0   # Will be updated from decoder FPS

        # 68 landmarks × 2 axes = 136 filters
        self._lmk_filters = [
            [_OneEuro(self._freq, min_cutoff, beta, d_cutoff) for _ in range(2)]
            for _ in range(68)
        ]
        # 3 pose angles
        self._pose_filters = [
            _OneEuro(self._freq, min_cutoff, beta, d_cutoff) for _ in range(3)
        ]
        logger.info(f"One-Euro smoother: min_cutoff={min_cutoff}, beta={beta}")

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.gate_passed or ctx.geometry is None:
            return ctx

        lmks = ctx.geometry.landmarks_2d.copy()   # (68, 2)
        smoothed = np.empty_like(lmks)

        for i in range(68):
            for j in range(2):
                smoothed[i, j] = self._lmk_filters[i][j](lmks[i, j])

        ctx.smoothed_landmarks = smoothed

        # Smooth pose angles too
        p = ctx.geometry.pose
        ctx.geometry.pose = np.array(
            [self._pose_filters[k](p[k]) for k in range(3)], dtype=np.float32
        )

        return ctx
