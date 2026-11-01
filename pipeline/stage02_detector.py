"""
pipeline/stage02_detector.py
─────────────────────────────
Stage 02 — Face Detection (RetinaFace)

Multi-scale anchor-free detector. Returns bounding boxes + 5-point landmarks
per face. Confidence threshold raised to 0.90 vs 0.85 in v1 to reduce
false detections entering the pipeline.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from loguru import logger

from .base import BoundingBox, FrameContext, PipelineStage


class FaceDetector(PipelineStage):
    """
    RetinaFace face detector.

    Outputs
    -------
    ctx.detections : list[BoundingBox]
        All detected faces sorted by confidence (descending).
        Limited to cfg['max_faces'] if set.
    """

    stage_name = "face_detection"

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        self._detector = None

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load RetinaFace via insightface."""
        try:
            import insightface
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device.type == "cuda"
                    else ["CPUExecutionProvider"]
                ),
            )
            self._app.prepare(ctx_id=0 if self.device.type == "cuda" else -1, det_size=(640, 640))
            logger.info("RetinaFace loaded via insightface buffalo_l.")
        except ImportError:
            logger.error("insightface not installed. Run: pip install insightface")
            raise

        self._conf_thr = self.cfg.get("confidence_threshold", 0.90)
        self._nms_thr  = self.cfg.get("nms_threshold", 0.40)
        self._max_faces = self.cfg.get("max_faces", 1)

    # ── Process ───────────────────────────────────────────────────────────────

    def process(self, ctx: FrameContext) -> FrameContext:
        faces = self._app.get(ctx.image)

        boxes: list[BoundingBox] = []
        for face in faces:
            conf = float(face.det_score)
            if conf < self._conf_thr:
                continue
            x1, y1, x2, y2 = face.bbox.tolist()
            boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf))

        # Sort by confidence descending
        boxes.sort(key=lambda b: b.confidence, reverse=True)

        # Limit faces
        if self._max_faces:
            boxes = boxes[: self._max_faces]

        ctx.detections = boxes
        return ctx
