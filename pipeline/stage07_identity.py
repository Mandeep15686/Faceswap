"""
pipeline/stage07_identity.py
─────────────────────────────
Stage 07 — Identity Embedding (ArcFace iResNet-100)

512-dim angular margin embedding of the SOURCE face.
Encodes identity independently of pose/lighting.

The source embedding is computed ONCE at startup from the multi-angle
source face dataset and cached for every frame — zero per-frame cost.

At inference, the TARGET (in-video) face embedding is also computed
and stored in ctx.target_embedding so downstream stages can compare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from loguru import logger

from .base import FrameContext, PipelineStage


class IdentityEmbedder(PipelineStage):
    stage_name = "identity_embedding"

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__(cfg, device)
        self.source_embedding: Optional[np.ndarray] = None   # (512,) cached

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        try:
            import insightface
            from insightface.model_zoo import model_zoo

            self._recognizer = insightface.model_zoo.get_model(
                "arcface_r100_v1",
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device.type == "cuda"
                    else ["CPUExecutionProvider"]
                ),
            )
            self._recognizer.prepare(ctx_id=0 if self.device.type == "cuda" else -1)
            logger.info("ArcFace iResNet-100 loaded.")
        except Exception as exc:
            logger.error(f"ArcFace load failed: {exc}")
            raise

        self._emb_dim = self.cfg.get("embedding_dim", 512)
        self._aggregation = self.cfg.get("aggregation", "mean")

    # ── Build source embedding from multi-angle images ────────────────────────

    def build_source_embedding(self, image_paths: list[Path]) -> np.ndarray:
        """
        Compute a robust identity embedding from multiple face photos
        (different angles). Aggregates by mean or frontality-weighted mean.

        Parameters
        ----------
        image_paths : list of Paths to source face images

        Returns
        -------
        embedding : np.ndarray, shape (512,)
        """
        embeddings: list[np.ndarray] = []
        weights: list[float] = []

        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning(f"Could not load source image: {p}")
                continue

            emb = self._get_embedding(img)
            if emb is None:
                logger.warning(f"No face found in source image: {p}")
                continue

            embeddings.append(emb)

            # Frontality weight: prefer images with small estimated yaw
            yaw = self._estimate_yaw(img)
            weight = max(0.1, 1.0 - abs(yaw) / 90.0)
            weights.append(weight)

        if not embeddings:
            raise ValueError("No valid source face images found!")

        embeddings = np.stack(embeddings, axis=0)          # (N, 512)

        if self._aggregation == "weighted_by_frontality":
            w = np.array(weights, dtype=np.float32)
            w /= w.sum()
            agg = (embeddings * w[:, None]).sum(axis=0)
        else:
            agg = embeddings.mean(axis=0)

        # L2-normalise
        agg = agg / (np.linalg.norm(agg) + 1e-6)

        self.source_embedding = agg
        logger.info(
            f"Source embedding built from {len(embeddings)} images "
            f"(aggregation={self._aggregation})."
        )
        return agg

    # ── Per-frame processing ──────────────────────────────────────────────────

    def process(self, ctx: FrameContext) -> FrameContext:
        """
        Compute target-face embedding from the aligned crop.
        Source embedding is already cached.
        """
        if not ctx.gate_passed or ctx.aligned is None:
            return ctx

        emb = self._get_embedding(ctx.aligned.crop)
        ctx.target_embedding = emb
        return ctx

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_embedding(self, bgr_image: np.ndarray) -> Optional[np.ndarray]:
        """Return L2-normalised 512-d embedding, or None if no face found."""
        try:
            # insightface recognizer expects aligned face
            emb = self._recognizer.get_feat(bgr_image).flatten()
            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                return None
            return (emb / norm).astype(np.float32)
        except Exception as exc:
            logger.debug(f"Embedding error: {exc}")
            return None

    @staticmethod
    def _estimate_yaw(bgr_image: np.ndarray) -> float:
        """Rough yaw estimate from face box / image centre ratio."""
        try:
            import insightface
            app = insightface.app.FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(320, 320))
            faces = app.get(bgr_image)
            if not faces:
                return 0.0
            # Simple heuristic: horizontal position of nose tip
            h, w = bgr_image.shape[:2]
            bbox = faces[0].bbox
            cx = (bbox[0] + bbox[2]) / 2
            return float((cx / w - 0.5) * 90)
        except Exception:
            return 0.0
