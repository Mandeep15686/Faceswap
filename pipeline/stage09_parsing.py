"""
pipeline/stage09_parsing.py
────────────────────────────
Stage 09 — Face Parsing (BiSeNet)

19-class semantic segmentation of the face crop:
  0:background  1:skin       2:l_brow    3:r_brow    4:l_eye
  5:r_eye       6:eye_glass  7:l_ear     8:r_ear     9:earring
  10:nose       11:mouth     12:u_lip    13:l_lip    14:neck
  15:neck_l     16:cloth     17:hair     18:hat

Output mask used by:
  • Stage 10: occlusion detection (background depth comparisons)
  • Stage 12: region-aware color transfer
  • Stage 13: seamless blending
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from .base import FrameContext, PipelineStage


class FaceParser(PipelineStage):
    stage_name = "face_parsing"

    def load(self) -> None:
        self._num_classes = self.cfg.get("num_classes", 19)
        self._input_size  = 512   # BiSeNet internal resolution

        try:
            self._load_bisenet()
        except Exception as exc:
            logger.warning(f"BiSeNet load failed ({exc}). Mask will be an ellipse fallback.")
            self._model_type = "fallback"

    def _load_bisenet(self) -> None:
        """Load BiSeNet from torchvision hub or local weights."""
        weights_path = Path(__file__).parent.parent / "models" / "bisenet_face_parsing.pt"

        # Define BiSeNet architecture inline (simplified)
        from torchvision.models import resnet18
        import torch.nn as nn

        class BiSeNet(nn.Module):
            def __init__(self, n_classes: int):
                super().__init__()
                backbone = resnet18(pretrained=False)
                self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
                self.maxpool = backbone.maxpool
                self.layer1 = backbone.layer1
                self.layer2 = backbone.layer2
                self.layer3 = backbone.layer3
                self.layer4 = backbone.layer4
                self.head = nn.Conv2d(512, n_classes, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.layer0(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.head(x)
                return x

        model = BiSeNet(self._num_classes)

        if weights_path.exists():
            state = torch.load(str(weights_path), map_location=self.device)
            model.load_state_dict(state, strict=False)
            logger.info(f"BiSeNet weights loaded from {weights_path}.")
        else:
            logger.warning(
                "BiSeNet weights not found. "
                "Download: python scripts/download_weights.py"
            )

        model = model.to(self.device).eval()
        if self.device.type == "cuda":
            model = model.half()
        self._model = model
        self._model_type = "bisenet"

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.gate_passed or ctx.swapped_face is None:
            return ctx

        if self._model_type == "bisenet":
            ctx.parse_mask = self._run_bisenet(ctx.swapped_face)
        else:
            ctx.parse_mask = self._ellipse_mask(ctx.swapped_face)
        return ctx

    def _run_bisenet(self, face_bgr: np.ndarray) -> np.ndarray:
        h, w = face_bgr.shape[:2]
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
            / 255.0
        )
        # Normalise (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        tensor = F.interpolate(tensor, size=(self._input_size, self._input_size), mode="bilinear")
        tensor = tensor.to(self.device)
        if self.device.type == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            logits = self._model(tensor)                  # (1, 19, H, W)
            mask = logits.argmax(dim=1).squeeze(0)        # (H, W)
            mask = F.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0),
                size=(h, w), mode="nearest"
            ).squeeze().long()
        return mask.cpu().numpy().astype(np.int32)

    @staticmethod
    def _ellipse_mask(face_bgr: np.ndarray) -> np.ndarray:
        """Fallback: simple ellipse mask — everything inside = class 1 (skin)."""
        h, w = face_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.int32)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - 4, h // 2 - 4), 0, 0, 360, 1, -1)
        return mask
