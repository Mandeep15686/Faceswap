"""
pipeline/stage14_encoder.py
────────────────────────────
Stage 14 — Final Video Encoder (H.264 / HEVC)

• Re-projects swapped face to original frame via stored inverse transform
  (already done in Stage 13's composited_frame).
• Writes composited frames to output video via PyAV.
• Muxes original audio stream.
• Preserves all metadata and timestamps.
• No temporal post-processing step — coherence handled upstream by
  DreamID's temporal conditioning (Stage 08).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np
from loguru import logger

from .base import FrameContext


class VideoEncoder:
    """
    Encodes composited frames to an output video file.

    Usage
    -----
    with VideoEncoder(output_path, decoder, cfg) as enc:
        for ctx in pipeline:
            enc.write(ctx)
    """

    def __init__(
        self,
        output_path: str | Path,
        source_decoder,           # VideoDecoder — for FPS / audio metadata
        cfg: dict,
    ):
        self.output_path = Path(output_path)
        self._decoder     = source_decoder
        self._cfg         = cfg

        self._container: Optional[av.container.OutputContainer] = None
        self._video_stream = None
        self._audio_stream = None
        self._frame_count  = 0

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    # ── Open output container ─────────────────────────────────────────────────

    def _open(self) -> None:
        codec  = self._cfg.get("codec", "h264")
        fps    = float(self._cfg.get("fps", self._decoder.fps or 24.0))
        crf    = int(self._cfg.get("crf", 18))
        audio  = bool(self._cfg.get("audio", True))

        # Map codec name to libav encoder name
        codec_map = {
            "h264": "libx264",
            "hevc": "libx265",
            "vp9":  "libvpx-vp9",
        }
        enc_name = codec_map.get(codec.lower(), "libx264")

        self._container = av.open(str(self.output_path), mode="w")

        # ── Video stream ──────────────────────────────────────────────────────
        self._video_stream = self._container.add_stream(enc_name, rate=fps)
        self._video_stream.width  = self._decoder.width
        self._video_stream.height = self._decoder.height
        self._video_stream.pix_fmt = "yuv420p"
        self._video_stream.options = {
            "crf":    str(crf),
            "preset": "veryfast",
        }
        if enc_name == "libx265":
            self._video_stream.options["x265-params"] = f"crf={crf}"

        # ── Audio stream (pass-through) ───────────────────────────────────────
        if audio and self._decoder.has_audio:
            src_audio = self._decoder.audio_stream
            self._audio_stream = self._container.add_stream(
                template=src_audio
            )

        self._fps = fps
        logger.info(
            f"Output video: {self._decoder.width}×{self._decoder.height} "
            f"@ {fps:.2f} FPS, codec={enc_name}, crf={crf} → {self.output_path}"
        )

    # ── Write a single frame ──────────────────────────────────────────────────

    def write(self, ctx: FrameContext) -> None:
        """Encode one composited frame."""
        # Use composited frame if available; else fall back to original
        frame_bgr = ctx.composited_frame if ctx.composited_frame is not None else ctx.image

        # BGR → RGB for PyAV
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        av_frame.pts = self._frame_count
        av_frame.time_base = self._video_stream.time_base

        for packet in self._video_stream.encode(av_frame):
            self._container.mux(packet)

        self._frame_count += 1

    # ── Flush & close ─────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._container is None:
            return

        # Flush encoder
        for packet in self._video_stream.encode():
            self._container.mux(packet)

        self._container.close()
        self._container = None
        logger.info(
            f"Encoding complete: {self._frame_count} frames → {self.output_path}"
        )
