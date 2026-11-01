"""
pipeline/base.py
────────────────
Shared data structures and abstract base class used by every pipeline stage.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


@dataclass
class FaceGeometry:
    """All geometric outputs from Stage 04 (3DDFA-V2)."""
    landmarks_2d: np.ndarray          # (68, 2) pixel coords
    landmarks_3d: np.ndarray          # (68, 3) 3D mesh points
    pose: np.ndarray                  # (3,) yaw, pitch, roll in degrees
    expression_coeffs: np.ndarray     # (64,) expression parameters
    shape_coeffs: np.ndarray          # (40,) identity shape parameters


@dataclass
class AlignedFace:
    """Crop + inverse transform produced by Stage 06."""
    crop: np.ndarray                  # (256, 256, 3) uint8 BGR
    inverse_transform: np.ndarray     # (2, 3) affine matrix for re-projection
    landmarks_5pt: np.ndarray         # (5, 2) anchor points in original coords


@dataclass
class FrameContext:
    """
    Everything the pipeline knows about a single video frame.
    Stages populate fields in-place as the frame flows through.
    """
    # ── Raw ──────────────────────────────────────────────────────────────────
    frame_idx: int
    timestamp_ms: float
    image: np.ndarray                             # Full-res BGR (H, W, 3) uint8

    # ── Stage 02: Detection ──────────────────────────────────────────────────
    detections: list[BoundingBox] = field(default_factory=list)

    # ── Stage 03: Tracking ───────────────────────────────────────────────────
    track_id: Optional[int] = None
    track_age: int = 0
    track_bbox: Optional[BoundingBox] = None

    # ── Quality gate ─────────────────────────────────────────────────────────
    gate_passed: bool = True
    gate_reason: Optional[str] = None            # Reason if rejected

    # ── Stage 04: Geometry ───────────────────────────────────────────────────
    geometry: Optional[FaceGeometry] = None

    # ── Stage 05: Smoothed geometry ──────────────────────────────────────────
    smoothed_landmarks: Optional[np.ndarray] = None   # (68, 2)

    # ── Stage 06: Alignment ──────────────────────────────────────────────────
    aligned: Optional[AlignedFace] = None

    # ── Stage 07: Identity ───────────────────────────────────────────────────
    target_embedding: Optional[np.ndarray] = None     # (512,) ArcFace vec

    # ── Stage 08: Swap ───────────────────────────────────────────────────────
    swapped_face: Optional[np.ndarray] = None         # (256, 256, 3) uint8
    prev_swapped_face: Optional[np.ndarray] = None    # Temporal feedback

    # ── Stage 09: Parsing ────────────────────────────────────────────────────
    parse_mask: Optional[np.ndarray] = None           # (256, 256) int, class idx

    # ── Stage 10: Occlusion ──────────────────────────────────────────────────
    occlusion_alpha: Optional[np.ndarray] = None      # (256, 256) float32 0–1

    # ── Stage 11: Enhancement ────────────────────────────────────────────────
    enhanced_face: Optional[np.ndarray] = None        # (256, 256, 3) uint8

    # ── Stage 12: Color-corrected ────────────────────────────────────────────
    color_corrected_face: Optional[np.ndarray] = None

    # ── Stage 13: Composited ────────────────────────────────────────────────
    composited_frame: Optional[np.ndarray] = None     # Full-res composited BGR

    # ── Timing ───────────────────────────────────────────────────────────────
    stage_times: dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract stage
# ─────────────────────────────────────────────────────────────────────────────

class PipelineStage(ABC):
    """
    Base class for every pipeline stage.

    Subclasses implement `process(ctx)` to mutate the FrameContext in-place.
    Timing is recorded automatically.
    """

    stage_name: str = "unnamed"

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._model: Optional[torch.nn.Module] = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called once before the main loop."""
        ...

    @abstractmethod
    def process(self, ctx: FrameContext) -> FrameContext:
        """
        Process one frame.  Mutate ctx and return it.
        For pass-through frames (gate_passed=False) implementations
        should return ctx unmodified unless they handle the pass-through case.
        """
        ...

    def __call__(self, ctx: FrameContext) -> FrameContext:
        t0 = time.perf_counter()
        result = self.process(ctx)
        ctx.stage_times[self.stage_name] = (time.perf_counter() - t0) * 1000  # ms
        return result

    def unload(self) -> None:
        """Optional: release GPU memory."""
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()
