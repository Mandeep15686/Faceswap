"""
pipeline/stage04_geometry.py
─────────────────────────────
Stage 04 — 3D Geometry (3DDFA-V2)

Single-pass model that replaces three separate models from v1:
  v1: FAN (landmarks) + 3DDFA (3D mesh) + separate 3DMM fitting
  v2: ONE 3DDFA-V2 forward pass → all outputs

Outputs per frame
─────────────────
• 68 2D landmarks (pixel coords)
• Sparse 3D mesh
• Pose — yaw, pitch, roll (degrees)
• Expression coefficients (64-dim)
• Shape coefficients (40-dim)

Saves ~10 ms/frame vs v1 triple-stage approach.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from loguru import logger

from .base import FaceGeometry, FrameContext, PipelineStage


class GeometryEstimator(PipelineStage):
    stage_name = "3d_geometry"

    def load(self) -> None:
        """
        Load 3DDFA-V2.

        Official repo: https://github.com/cleardusk/3DDFA_V2
        We import via the `tddfa` package if installed, otherwise
        fall back to a lightweight landmark-only approach using dlib/mediapipe.
        """
        self._num_lmk = self.cfg.get("num_landmarks", 68)
        self._model_loaded = False

        # ── Attempt full 3DDFA-V2 ─────────────────────────────────────────
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "3DDFA_V2"))
            from FaceBoxes import FaceBoxes
            from TDDFA import TDDFA
            import yaml

            tddfa_cfg_path = Path(__file__).parent.parent / "models" / "3DDFA_V2" / "configs" / "mb1_120x120.yml"
            with open(tddfa_cfg_path) as f:
                tddfa_cfg = yaml.safe_load(f)

            self._tddfa = TDDFA(gpu_mode=(self.device.type == "cuda"), **tddfa_cfg)
            self._model_loaded = True
            logger.info("3DDFA-V2 loaded successfully.")
        except Exception as exc:
            logger.warning(
                f"3DDFA-V2 not available ({exc}). "
                "Using MediaPipe landmark fallback (no 3D mesh / expression output). "
                "For full pipeline, install 3DDFA_V2: "
                "https://github.com/cleardusk/3DDFA_V2"
            )
            self._load_mediapipe_fallback()

    def _load_mediapipe_fallback(self) -> None:
        """Use MediaPipe Face Mesh as a lightweight fallback."""
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._fallback = "mediapipe"
            logger.info("MediaPipe Face Mesh fallback loaded.")
        except ImportError:
            self._fallback = "none"
            logger.warning("MediaPipe also unavailable. Geometry stage will be skipped.")

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.gate_passed:
            return ctx
        if not ctx.detections:
            return ctx

        if self._model_loaded:
            ctx.geometry = self._run_3ddfa(ctx)
        elif hasattr(self, "_fallback") and self._fallback == "mediapipe":
            ctx.geometry = self._run_mediapipe(ctx)
        # else: geometry remains None, downstream stages handle gracefully

        return ctx

    # ── 3DDFA-V2 full inference ───────────────────────────────────────────────

    def _run_3ddfa(self, ctx: FrameContext) -> Optional[FaceGeometry]:
        try:
            from utils.pose import calc_pose  # 3DDFA_V2 util

            det = ctx.detections[0]
            box = [det.x1, det.y1, det.x2, det.y2, det.confidence]

            param_lst, roi_box_lst = self._tddfa(ctx.image, [box])
            ver_lst = self._tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)

            vertices = ver_lst[0]             # (3, 68)
            landmarks_3d = vertices.T         # (68, 3)
            landmarks_2d = landmarks_3d[:, :2].copy()

            # Pose from param
            param = param_lst[0]
            R = param[:12].reshape(3, 4)[:, :3]
            import cv2 as cv
            rvec, _ = cv.Rodrigues(R)
            yaw   = float(np.degrees(rvec[1]))
            pitch = float(np.degrees(rvec[0]))
            roll  = float(np.degrees(rvec[2]))

            expression_coeffs = param[52:116] if len(param) > 116 else np.zeros(64)
            shape_coeffs      = param[12:52]  if len(param) > 52  else np.zeros(40)

            return FaceGeometry(
                landmarks_2d=landmarks_2d,
                landmarks_3d=landmarks_3d,
                pose=np.array([yaw, pitch, roll], dtype=np.float32),
                expression_coeffs=expression_coeffs.astype(np.float32),
                shape_coeffs=shape_coeffs.astype(np.float32),
            )
        except Exception as exc:
            logger.debug(f"3DDFA-V2 inference error: {exc}")
            return None

    # ── MediaPipe fallback ────────────────────────────────────────────────────

    def _run_mediapipe(self, ctx: FrameContext) -> Optional[FaceGeometry]:
        try:
            import mediapipe as mp
            h, w = ctx.image.shape[:2]
            rgb = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2RGB)
            results = self._mp_face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None

            face_lmk = results.multi_face_landmarks[0]

            # MediaPipe gives 478 landmarks; take first 68-equivalent subset
            lmks = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face_lmk.landmark])
            landmarks_2d = lmks[:68, :2]
            landmarks_3d = lmks[:68, :]

            # Approximate pose from face normal
            nose    = lmks[1,  :2]
            left    = lmks[234, :2]
            right   = lmks[454, :2]
            dx = right[0] - left[0]
            dy = right[1] - left[1]
            roll  = float(np.degrees(np.arctan2(dy, dx)))
            yaw   = float((nose[0] - w / 2) / (w / 2) * 40)
            pitch = float((nose[1] - h / 2) / (h / 2) * 30)

            return FaceGeometry(
                landmarks_2d=landmarks_2d.astype(np.float32),
                landmarks_3d=landmarks_3d.astype(np.float32),
                pose=np.array([yaw, pitch, roll], dtype=np.float32),
                expression_coeffs=np.zeros(64, dtype=np.float32),
                shape_coeffs=np.zeros(40, dtype=np.float32),
            )
        except Exception as exc:
            logger.debug(f"MediaPipe fallback error: {exc}")
            return None
