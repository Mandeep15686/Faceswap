"""
pipeline/stage10_occlusion.py
──────────────────────────────
Stage 10 — Occlusion Detection (Confidence-weighted alpha)  [NEW in v2]

Detects external occlusions — hands, glasses, microphones — not captured
by BiSeNet's face-only semantic classes.

Approach
────────
1. Run MiDaS depth estimation on the full frame (small/fast variant).
2. Compare depth of BiSeNet face region vs surrounding foreground pixels.
3. Where foreground depth < face depth by > threshold → occlusion detected.
4. Dilate occlusion region and set blend_alpha → 0 for those pixels
   (keep original frame pixels, don't blend swapped face there).

Prevents the worst artifact class: swapped face visible through an occlusion.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from .base import FrameContext, PipelineStage


class OcclusionDetector(PipelineStage):
    stage_name = "occlusion_detection"

    def load(self) -> None:
        self._depth_threshold = self.cfg.get("delta_threshold", 0.20)
        self._dilation_k      = self.cfg.get("dilation_kernel", 7)

        try:
            self._load_midas()
        except Exception as exc:
            logger.warning(f"MiDaS load failed ({exc}). Occlusion detection disabled.")
            self._midas = None

    def _load_midas(self) -> None:
        weights_path = Path(__file__).parent.parent / "models" / "midas_small.pt"
        self._midas = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small",
            source="local" if weights_path.exists() else "github",
            pretrained=(not weights_path.exists()),
        )
        self._midas = self._midas.to(self.device).eval()

        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms",
            source="local" if weights_path.exists() else "github",
        )
        self._midas_transform = transforms.small_transform
        logger.info("MiDaS small depth model loaded.")

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.gate_passed or ctx.swapped_face is None:
            return ctx

        if ctx.parse_mask is None:
            ctx.occlusion_alpha = np.ones(ctx.swapped_face.shape[:2], dtype=np.float32)
            return ctx

        if self._midas is None:
            # No depth model → no occlusion detection; alpha = 1 everywhere
            ctx.occlusion_alpha = np.ones(ctx.swapped_face.shape[:2], dtype=np.float32)
            return ctx

        # ── Run MiDaS on aligned face crop ───────────────────────────────────
        depth = self._estimate_depth(ctx.swapped_face)           # (H, W) float
        depth_norm = (depth - depth.min()) / (depth.ptp() + 1e-6)

        # ── Face region from BiSeNet mask ─────────────────────────────────────
        face_classes = {1, 2, 3, 4, 5, 10, 11, 12, 13}          # skin + features
        face_mask = np.isin(ctx.parse_mask, list(face_classes))  # (H, W) bool

        if face_mask.sum() == 0:
            ctx.occlusion_alpha = np.ones_like(depth_norm)
            return ctx

        face_depth_mean = float(depth_norm[face_mask].mean())

        # ── Potential occlusion: non-face pixels that are CLOSER than face ──
        non_face = ~face_mask
        occlusion = non_face & (depth_norm < face_depth_mean - self._depth_threshold)

        # ── Dilate to cover object edges ──────────────────────────────────────
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._dilation_k, self._dilation_k)
        )
        occ_dilated = cv2.dilate(occlusion.astype(np.uint8), kernel)

        # Alpha = 1 (blend swapped) everywhere except occlusions (alpha = 0)
        alpha = 1.0 - occ_dilated.astype(np.float32)
        # Feather edges
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

        ctx.occlusion_alpha = alpha
        return ctx

    def _estimate_depth(self, face_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        inp = self._midas_transform(rgb).to(self.device)
        with torch.no_grad():
            depth = self._midas(inp)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=face_bgr.shape[:2],
                mode="bicubic", align_corners=False,
            ).squeeze().cpu().numpy()
        return depth.astype(np.float32)
