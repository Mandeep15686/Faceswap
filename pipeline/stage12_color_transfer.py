"""
pipeline/stage12_color_transfer.py
────────────────────────────────────
Stage 12 — Region-Aware Color Transfer  [MOVED POST-ENHANCE, NEW REGION-AWARE in v2]

Two key fixes vs v1
────────────────────
1. ORDER FIXED  — Color transfer now runs after CodeFormer enhancement, so the
   enhancer cannot re-introduce color bias after LAB correction.

2. REGION-AWARE — LAB histogram matching applied independently per BiSeNet
   semantic region (skin, hair, brows) instead of uniformly across the face.
   Eliminates uniform color-shift artifacts where face illumination differs
   from body.

Algorithm
──────────
For each configured semantic region:
  • Extract pixels from ENHANCED swapped face (source distribution)
  • Extract pixels from the TARGET face crop (reference distribution)
  • Match LAB histogram per-channel using CDF lookup table
  • Write corrected pixels back only to that region
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from loguru import logger

from .base import FrameContext, PipelineStage

# BiSeNet class indices for the regions we correct
_REGION_MAP: dict[str, list[int]] = {
    "skin":   [1],
    "l_brow": [2],
    "r_brow": [3],
    "l_eye":  [4],
    "r_eye":  [5],
    "hair":   [17],
    "nose":   [10],
    "u_lip":  [12],
    "l_lip":  [13],
}


class ColorTransfer(PipelineStage):
    stage_name = "color_transfer"

    def load(self) -> None:
        self._method        = self.cfg.get("method", "lab_histogram")
        self._region_aware  = bool(self.cfg.get("region_aware", True))
        self._regions       = self.cfg.get("regions", ["skin", "hair", "l_brow", "r_brow"])
        self._clip_pct      = float(self.cfg.get("clip_percentile", 1.0))
        logger.info(
            f"Color transfer: method={self._method}, "
            f"region_aware={self._region_aware}, regions={self._regions}"
        )

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.gate_passed or ctx.enhanced_face is None:
            return ctx

        source = ctx.enhanced_face        # Face to correct (swapped, enhanced)
        reference = ctx.aligned.crop      # Original target face (color reference)

        if self._region_aware and ctx.parse_mask is not None:
            ctx.color_corrected_face = self._region_aware_transfer(
                source, reference, ctx.parse_mask
            )
        elif self._method == "lab_histogram":
            ctx.color_corrected_face = self._lab_histogram_transfer(source, reference)
        else:
            # Reinhard global transfer
            ctx.color_corrected_face = self._reinhard_transfer(source, reference)

        return ctx

    # ── Region-aware LAB histogram matching ───────────────────────────────────

    def _region_aware_transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        result = source.astype(np.float32)

        src_lab = cv2.cvtColor(source,    cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

        for region_name in self._regions:
            indices = _REGION_MAP.get(region_name)
            if indices is None:
                continue

            region_mask = np.isin(mask, indices)
            if region_mask.sum() < 10:
                continue   # Too few pixels in region → skip

            for ch in range(3):  # L, A, B channels
                src_vals  = src_lab[region_mask, ch]
                ref_vals  = ref_lab[region_mask, ch]

                corrected = self._histogram_match_1d(src_vals, ref_vals)
                corrected = np.clip(corrected, 0, 255)

                # Write corrected channel back to region pixels
                lab_channel = src_lab[:, :, ch].copy()
                lab_channel[region_mask] = corrected
                src_lab[:, :, ch] = lab_channel

        result_bgr = cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result_bgr

    # ── Global LAB histogram matching ─────────────────────────────────────────

    def _lab_histogram_transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        src_lab = cv2.cvtColor(source,    cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

        for ch in range(3):
            src_vals = src_lab[:, :, ch].flatten()
            ref_vals = ref_lab[:, :, ch].flatten()
            matched  = self._histogram_match_1d(src_vals, ref_vals)
            src_lab[:, :, ch] = matched.reshape(src_lab.shape[:2])

        return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ── Reinhard method (fast global alternative) ─────────────────────────────

    @staticmethod
    def _reinhard_transfer(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        src_lab = cv2.cvtColor(source,    cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

        for ch in range(3):
            src_m, src_s = src_lab[:,:,ch].mean(), src_lab[:,:,ch].std() + 1e-6
            ref_m, ref_s = ref_lab[:,:,ch].mean(), ref_lab[:,:,ch].std() + 1e-6
            src_lab[:,:,ch] = (src_lab[:,:,ch] - src_m) * (ref_s / src_s) + ref_m

        return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ── Histogram matching (CDF method) ───────────────────────────────────────

    def _histogram_match_1d(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        n_bins: int = 256,
    ) -> np.ndarray:
        """Match the histogram of `source` values to `reference` values."""
        src_vals  = np.clip(source,    0, 255).astype(np.uint8)
        ref_vals  = np.clip(reference, 0, 255).astype(np.uint8)

        # CDFs
        src_hist, _ = np.histogram(src_vals, bins=n_bins, range=(0, 255))
        ref_hist, _ = np.histogram(ref_vals, bins=n_bins, range=(0, 255))

        src_cdf = src_hist.cumsum().astype(np.float64)
        ref_cdf = ref_hist.cumsum().astype(np.float64)

        src_cdf = src_cdf / (src_cdf[-1] + 1e-6)
        ref_cdf = ref_cdf / (ref_cdf[-1] + 1e-6)

        # Build LUT: for each source bin, find closest reference bin CDF
        lut = np.interp(src_cdf, ref_cdf, np.arange(n_bins)).astype(np.uint8)
        return lut[src_vals].astype(np.float32)
