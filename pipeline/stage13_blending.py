"""
pipeline/stage13_blending.py
──────────────────────────────
Stage 13 — Seamless Blending (Laplacian + Poisson)

Two-pass compositing using the combined BiSeNet + occlusion mask:

Pass 1 — Laplacian Pyramid Blend
  High-frequency texture blend (sharp edges, pores, fine detail).

Pass 2 — Poisson Seamless Clone
  Low-frequency gradient matching (color continuity, shadow transitions).

Occlusion mask automatically zeros out blend contribution in
occluded regions, keeping original pixels there.
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from .base import FrameContext, PipelineStage


class SeamlessBlender(PipelineStage):
    stage_name = "seamless_blending"

    def load(self) -> None:
        self._lap_levels    = int(self.cfg.get("laplacian_levels", 6))
        self._poisson_solver = self.cfg.get("poisson_solver", "direct")
        self._feather_radius = int(self.cfg.get("feather_radius", 8))
        logger.info(
            f"Blender: Laplacian({self._lap_levels} levels) + "
            f"Poisson({self._poisson_solver}), feather={self._feather_radius}px"
        )

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.gate_passed or ctx.aligned is None:
            # Gate fail → pass through original
            ctx.composited_frame = ctx.image.copy()
            return ctx

        face = ctx.color_corrected_face
        if face is None:
            face = ctx.enhanced_face
        if face is None:
            face = ctx.swapped_face
        if face is None:
            ctx.composited_frame = ctx.image.copy()
            return ctx

        # ── Build composite mask ──────────────────────────────────────────────
        blend_mask = self._build_blend_mask(ctx, face.shape[:2])

        # ── Warp swapped face back to full-frame coordinates ──────────────────
        h_full, w_full = ctx.image.shape[:2]
        warped_face = cv2.warpAffine(
            face, ctx.aligned.inverse_transform,
            (w_full, h_full),
            flags=cv2.INTER_LINEAR,
        )
        warped_mask = cv2.warpAffine(
            blend_mask, ctx.aligned.inverse_transform,
            (w_full, h_full),
            flags=cv2.INTER_LINEAR,
        )

        # ── Pass 1: Laplacian pyramid blend ───────────────────────────────────
        blended = self._laplacian_blend(ctx.image, warped_face, warped_mask)

        # ── Pass 2: Poisson seamless clone ────────────────────────────────────
        blended = self._poisson_blend(ctx.image, blended, warped_mask)

        ctx.composited_frame = blended
        return ctx

    # ── Mask construction ─────────────────────────────────────────────────────

    def _build_blend_mask(self, ctx: FrameContext, shape: tuple[int, int]) -> np.ndarray:
        """
        Returns float32 mask (H, W, 1) in [0, 1].
        1 = blend swapped face, 0 = keep original.
        """
        h, w = shape

        # Start with BiSeNet face region
        if ctx.parse_mask is not None:
            face_classes = {1, 2, 3, 4, 5, 10, 11, 12, 13}
            mask = np.isin(ctx.parse_mask, list(face_classes)).astype(np.float32)
        else:
            # Ellipse fallback
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 2 - 4), 0, 0, 360, 1.0, -1)

        # Multiply by occlusion alpha (zeros out occluded regions)
        if ctx.occlusion_alpha is not None:
            occ = cv2.resize(ctx.occlusion_alpha, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = mask * occ

        # Feather edges
        if self._feather_radius > 0:
            k = self._feather_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        return mask[:, :, np.newaxis]   # (H, W, 1)

    # ── Laplacian pyramid blend ───────────────────────────────────────────────

    def _laplacian_blend(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        mask: np.ndarray,  # (H, W, 1) float32
    ) -> np.ndarray:
        levels = self._lap_levels
        orig_f  = original.astype(np.float32)
        swap_f  = swapped.astype(np.float32)
        mask_f  = np.repeat(mask, 3, axis=2)   # (H, W, 3)

        # Build Laplacian pyramids
        def _build_lap(img):
            pyr = []
            cur = img
            for _ in range(levels):
                down = cv2.pyrDown(cur)
                up   = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
                pyr.append(cur - up)
                cur  = down
            pyr.append(cur)
            return pyr

        def _build_gauss(m):
            pyr = [m]
            for _ in range(levels):
                pyr.append(cv2.pyrDown(pyr[-1]))
            return pyr

        lap_o = _build_lap(orig_f)
        lap_s = _build_lap(swap_f)
        gau_m = _build_gauss(mask_f)

        # Blend each level
        blended_pyr = []
        for lo, ls, gm in zip(lap_o, lap_s, gau_m):
            gm3 = gm if gm.ndim == 3 else gm[:, :, np.newaxis]
            blended_pyr.append(lo * (1 - gm3) + ls * gm3)

        # Collapse
        result = blended_pyr[-1]
        for layer in reversed(blended_pyr[:-1]):
            result = cv2.pyrUp(result, dstsize=(layer.shape[1], layer.shape[0]))
            result = result + layer

        return np.clip(result, 0, 255).astype(np.uint8)

    # ── Poisson seamless clone ────────────────────────────────────────────────

    def _poisson_blend(
        self,
        original: np.ndarray,
        laplacian_blended: np.ndarray,
        mask: np.ndarray,   # (H, W, 1) float32
    ) -> np.ndarray:
        """
        Apply Poisson seamless cloning in the face region using OpenCV.
        Falls back to laplacian_blended if no valid mask region.
        """
        try:
            mask_uint8 = (mask[:, :, 0] * 255).astype(np.uint8)

            # Find bounding box of non-zero mask region
            ys, xs = np.where(mask_uint8 > 30)
            if len(ys) == 0:
                return laplacian_blended

            cy = int((ys.min() + ys.max()) / 2)
            cx = int((xs.min() + xs.max()) / 2)

            result = cv2.seamlessClone(
                laplacian_blended, original, mask_uint8,
                (cx, cy), cv2.NORMAL_CLONE
            )
            return result

        except Exception as exc:
            logger.debug(f"Poisson blend error: {exc}")
            return laplacian_blended
