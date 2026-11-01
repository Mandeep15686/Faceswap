"""
pipeline/stage11_enhancement.py
─────────────────────────────────
Stage 11 — Enhancement (CodeFormer)  [ORDER FIXED in v2]

Blind face restoration for structure fidelity.
Operates on the raw swapped face BEFORE color correction.

Change from v1
──────────────
Enhancement now runs BEFORE color transfer.
v1's ordering allowed CodeFormer to re-introduce color bias after
LAB correction was already applied. Fixed by moving this stage earlier.

GFPGAN removed
──────────────
CodeFormer alone is sufficient at 256px crop size.
Running both added 8ms with marginal gain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from loguru import logger

from .base import FrameContext, PipelineStage


class FaceEnhancer(PipelineStage):
    stage_name = "enhancement_codeformer"

    def load(self) -> None:
        self._fidelity_weight = float(self.cfg.get("fidelity_weight", 0.70))
        self._upscale         = int(self.cfg.get("upscale", 1))

        try:
            self._load_codeformer()
        except Exception as exc:
            logger.warning(
                f"CodeFormer load failed ({exc}). "
                "Enhancement will be bypassed. "
                "Install: pip install basicsr facexlib"
            )
            self._model = None

    def _load_codeformer(self) -> None:
        """Load CodeFormer via basicsr/facexlib."""
        # Primary: facexlib CodeFormer
        try:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            from basicsr.utils import img2tensor, tensor2img
            from basicsr.utils.registry import ARCH_REGISTRY

            weights_path = Path(__file__).parent.parent / "models" / "codeformer.pth"

            if not weights_path.exists():
                raise FileNotFoundError(
                    f"CodeFormer weights not found at {weights_path}. "
                    "Run: python scripts/download_weights.py"
                )

            net = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]
            ).to(self.device)

            checkpoint = torch.load(str(weights_path), map_location=self.device)
            net.load_state_dict(checkpoint["params_ema"])
            net.eval()
            self._model = net
            self._img2tensor = img2tensor
            self._tensor2img = tensor2img

            logger.info(
                f"CodeFormer loaded. fidelity={self._fidelity_weight}, "
                f"upscale={self._upscale}x"
            )

        except ImportError as exc:
            raise ImportError(f"basicsr/facexlib not available: {exc}") from exc

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.gate_passed or ctx.swapped_face is None:
            return ctx

        if self._model is None:
            # Bypass: pass swapped face directly
            ctx.enhanced_face = ctx.swapped_face
            return ctx

        ctx.enhanced_face = self._run_codeformer(ctx.swapped_face)
        return ctx

    def _run_codeformer(self, face_bgr: np.ndarray) -> np.ndarray:
        """Run CodeFormer restoration on a single aligned face crop."""
        try:
            # Resize to 256 if needed (CodeFormer standard input)
            h, w = face_bgr.shape[:2]
            if h != 256 or w != 256:
                face_bgr = cv2.resize(face_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)

            # To tensor, normalise to [-1, 1]
            face_t = self._img2tensor(face_bgr / 255.0, bgr2rgb=True, float32=True)
            face_t = face_t.unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                face_t = face_t.half()

            w_tensor = torch.tensor([self._fidelity_weight], device=self.device)

            with torch.no_grad():
                output = self._model(face_t, w=w_tensor, adain=True)[0]

            # Back to BGR uint8
            restored = self._tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            restored = restored.astype(np.uint8)

            # Optional super-resolution
            if self._upscale > 1:
                restored = cv2.resize(
                    restored,
                    (restored.shape[1] * self._upscale, restored.shape[0] * self._upscale),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                # Scale back to original crop size
                restored = cv2.resize(restored, (w, h), interpolation=cv2.INTER_AREA)

            return restored

        except Exception as exc:
            logger.debug(f"CodeFormer inference error: {exc}")
            return face_bgr
