"""
pipeline/stage01_decoder.py
────────────────────────────
Stage 01 — Video Frame Decoder

Uses PyAV for hardware-accelerated decode (NVDEC on NVIDIA GPUs).
Yields FrameContext objects with the raw BGR frame populated.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Optional

import av
import cv2
import numpy as np
from loguru import logger

from .base import FrameContext


class VideoDecoder:
    """
    Hardware-accelerated video decoder.

    Supports:
    - H.264 / HEVC input via NVDEC (NVIDIA) or libav software fallback
    - Up to 4K input resolution
    - Audio stream pass-through metadata for Stage 14 muxing

    Usage
    -----
    decoder = VideoDecoder(video_path, hw_decode=True)
    for ctx in decoder:
        pipeline.process(ctx)
    decoder.close()
    """

    def __init__(
        self,
        video_path: str | Path,
        hw_decode: bool = True,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ):
        self.video_path = Path(video_path)
        self.hw_decode = hw_decode
        self.start_frame = start_frame
        self.end_frame = end_frame

        self._container: Optional[av.container.InputContainer] = None
        self._video_stream = None
        self._audio_stream = None

        self._fps: float = 0.0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0

        self._open()

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def has_audio(self) -> bool:
        return self._audio_stream is not None

    @property
    def audio_stream(self):
        return self._audio_stream

    # ── Internal open ─────────────────────────────────────────────────────────

    def _open(self) -> None:
        options: dict = {}

        if self.hw_decode:
            # Try NVIDIA hardware decode; fall back to software
            try:
                options = {"hwaccel": "cuda", "hwaccel_output_format": "cuda"}
                self._container = av.open(str(self.video_path), options=options)
                logger.info("Opened video with NVDEC hardware decode.")
            except Exception as exc:
                logger.warning(f"NVDEC unavailable ({exc}); using software decode.")
                options = {}
                self._container = av.open(str(self.video_path))
        else:
            self._container = av.open(str(self.video_path))

        # Find streams
        for stream in self._container.streams:
            if stream.type == "video" and self._video_stream is None:
                self._video_stream = stream
            elif stream.type == "audio" and self._audio_stream is None:
                self._audio_stream = stream

        if self._video_stream is None:
            raise RuntimeError(f"No video stream found in {self.video_path}")

        # Metadata
        vs = self._video_stream
        self._fps = float(vs.average_rate) if vs.average_rate else 24.0
        self._total_frames = vs.frames if vs.frames else 0
        self._width = vs.width
        self._height = vs.height

        logger.info(
            f"Video: {self._width}×{self._height} @ {self._fps:.2f} FPS "
            f"({self._total_frames} frames)"
        )

    # ── Iteration ─────────────────────────────────────────────────────────────

    def __iter__(self) -> Generator[FrameContext, None, None]:
        frame_idx = 0

        for packet in self._container.demux(self._video_stream):
            for frame in packet.decode():
                if frame_idx < self.start_frame:
                    frame_idx += 1
                    continue
                if self.end_frame is not None and frame_idx >= self.end_frame:
                    return

                # Convert to BGR numpy array
                bgr = self._frame_to_bgr(frame)

                ts_ms = (frame.pts * self._video_stream.time_base * 1000
                         if frame.pts is not None else frame_idx * (1000.0 / self._fps))

                yield FrameContext(
                    frame_idx=frame_idx,
                    timestamp_ms=float(ts_ms),
                    image=bgr,
                )

                frame_idx += 1

    def _frame_to_bgr(self, frame: av.VideoFrame) -> np.ndarray:
        """Convert PyAV frame → uint8 BGR numpy array."""
        # If NVDEC returned a CUDA frame, transfer to CPU first
        if frame.format.name in ("cuda", "d3d11"):
            frame = frame.to_rgb()
        rgb = frame.to_ndarray(format="rgb24")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._container:
            self._container.close()
            self._container = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
