"""
utils/utils.py
───────────────
General-purpose utilities shared across pipeline stages.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────

def read_bgr(path: str | Path) -> np.ndarray:
    """Read an image as uint8 BGR. Raises if unreadable."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def write_bgr(path: str | Path, image: np.ndarray) -> None:
    """Save BGR uint8 image. Creates parent dirs if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def resize_keep_aspect(image: np.ndarray, max_size: int) -> np.ndarray:
    """Resize so the longer edge = max_size, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_bbox(
    image: np.ndarray,
    bbox,                              # BoundingBox or (x1,y1,x2,y2)
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """Draw a bounding box and optional label on a copy of image."""
    out = image.copy()
    if hasattr(bbox, "x1"):
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    else:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(out, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def draw_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,             # (N, 2) pixel coords
    color: tuple[int, int, int] = (0, 200, 255),
    radius: int = 2,
) -> np.ndarray:
    """Draw landmark dots on a copy of image."""
    out = image.copy()
    for (x, y) in landmarks:
        cv2.circle(out, (int(x), int(y)), radius, color, -1)
    return out


def visualise_parse_mask(
    mask: np.ndarray,
    alpha: float = 0.5,
    base_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Render a BiSeNet segmentation mask as a colourful overlay.
    Returns (H, W, 3) uint8.
    """
    # 19-class colour palette
    PALETTE = np.array([
        [0,   0,   0  ],  # 0  background
        [255, 200, 180],  # 1  skin
        [200, 120, 80 ],  # 2  l_brow
        [200, 120, 80 ],  # 3  r_brow
        [80,  200, 255],  # 4  l_eye
        [80,  200, 255],  # 5  r_eye
        [128, 128, 255],  # 6  eye_glass
        [180, 180, 180],  # 7  l_ear
        [180, 180, 180],  # 8  r_ear
        [255, 215,   0],  # 9  earring
        [255, 150, 100],  # 10 nose
        [200, 80,  80 ],  # 11 mouth
        [220, 100, 120],  # 12 u_lip
        [220, 100, 120],  # 13 l_lip
        [100, 100, 100],  # 14 neck
        [100, 100, 100],  # 15 neck_l
        [ 60,  60,  60],  # 16 cloth
        [ 50, 200, 100],  # 17 hair
        [200, 200,   0],  # 18 hat
    ], dtype=np.uint8)

    coloured = PALETTE[np.clip(mask, 0, 18)]

    if base_image is not None:
        return cv2.addWeighted(base_image, 1 - alpha, coloured, alpha, 0)
    return coloured


# ─────────────────────────────────────────────────────────────────────────────
# Timing decorator
# ─────────────────────────────────────────────────────────────────────────────

def timeit(func: Callable) -> Callable:
    """Decorator that prints function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        print(f"[timeit] {func.__name__}: {ms:.1f} ms")
        return result
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip and convert to uint8."""
    return np.clip(image, 0, 255).astype(np.uint8)
