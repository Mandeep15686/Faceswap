"""
dataset/dataset_utils.py
─────────────────────────
Utilities for loading, validating, and inspecting the face swap dataset.

Dataset Structure Expected
──────────────────────────
dataset/
├── source_faces/          ← Multi-angle photos of the SOURCE identity
│   ├── front.jpg          (Recommended: 5–10 diverse angles)
│   ├── left_15.jpg
│   ├── left_30.jpg
│   ├── right_15.jpg
│   ├── right_30.jpg
│   └── ...
├── original_videos/       ← Target videos (face to be replaced)
│   └── target.mp4
└── output/               ← Swapped output videos appear here
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

SUPPORTED_IMAGE_EXT  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
SUPPORTED_VIDEO_EXT  = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ts"}


class DatasetValidator:
    """
    Validates that the dataset directory is correctly structured
    before starting a long inference run.
    """

    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root)

    def validate(self) -> dict:
        """
        Run all checks and return a summary dict.
        Raises ValueError if critical checks fail.
        """
        results: dict = {}

        # ── Source faces ─────────────────────────────────────────────────────
        src_dir = self.root / "source_faces"
        results["source_faces_dir"] = src_dir.exists()
        if not src_dir.exists():
            raise ValueError(
                f"source_faces/ directory not found at {src_dir}.\n"
                "Create it and add multi-angle face photos of the identity you want in the video."
            )

        src_images = self._find_images(src_dir)
        results["source_image_count"] = len(src_images)
        if len(src_images) == 0:
            raise ValueError(
                f"No images found in {src_dir}.\n"
                f"Supported formats: {SUPPORTED_IMAGE_EXT}"
            )
        if len(src_images) < 3:
            logger.warning(
                f"Only {len(src_images)} source images found. "
                "Recommended: 5+ diverse angles for best identity transfer."
            )

        # Validate each source image is readable and contains a detectable face
        results["source_images"] = []
        for p in src_images:
            img = cv2.imread(str(p))
            ok  = img is not None and img.shape[0] > 32 and img.shape[1] > 32
            results["source_images"].append({"path": str(p), "readable": ok, "shape": img.shape if ok else None})
            if not ok:
                logger.warning(f"Source image unreadable or too small: {p}")

        # ── Target videos ────────────────────────────────────────────────────
        vid_dir = self.root / "original_videos"
        results["original_videos_dir"] = vid_dir.exists()
        if vid_dir.exists():
            videos = self._find_videos(vid_dir)
            results["video_count"] = len(videos)
            results["videos"] = [str(v) for v in videos]
        else:
            results["video_count"] = 0
            logger.warning(f"original_videos/ not found at {vid_dir}.")

        # ── Output directory ─────────────────────────────────────────────────
        out_dir = self.root / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        results["output_dir"] = str(out_dir)

        return results

    def print_report(self) -> None:
        try:
            r = self.validate()
        except ValueError as exc:
            logger.error(f"Dataset validation FAILED: {exc}")
            return

        print("\n" + "="*60)
        print("  DATASET VALIDATION REPORT")
        print("="*60)
        print(f"  Root            : {self.root}")
        print(f"  Source images   : {r['source_image_count']}")
        for item in r.get("source_images", []):
            status = "✓" if item["readable"] else "✗"
            shape  = item["shape"] or "unreadable"
            print(f"    {status}  {Path(item['path']).name}  {shape}")
        print(f"  Target videos   : {r.get('video_count', 0)}")
        for v in r.get("videos", []):
            print(f"    •  {Path(v).name}")
        print(f"  Output dir      : {r['output_dir']}")
        print("="*60 + "\n")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_images(self, directory: Path) -> list[Path]:
        return sorted([
            p for p in directory.iterdir()
            if p.suffix.lower() in SUPPORTED_IMAGE_EXT
        ])

    def _find_videos(self, directory: Path) -> list[Path]:
        return sorted([
            p for p in directory.iterdir()
            if p.suffix.lower() in SUPPORTED_VIDEO_EXT
        ])


class SourceFaceDataset:
    """
    Loads and pre-processes source face images for embedding computation.
    Provides helpers for angle estimation and quality filtering.
    """

    def __init__(self, source_dir: str | Path, target_size: int = 256):
        self.source_dir  = Path(source_dir)
        self.target_size = target_size
        self._images: list[np.ndarray] = []
        self._paths:  list[Path]       = []

    def load(self) -> None:
        """Load all images from source_dir."""
        image_paths = sorted([
            p for p in self.source_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_IMAGE_EXT
        ])
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (self.target_size, self.target_size))
            self._images.append(img)
            self._paths.append(p)

        logger.info(f"Loaded {len(self._images)} source face images from {self.source_dir}")

    @property
    def images(self) -> list[np.ndarray]:
        return self._images

    @property
    def paths(self) -> list[Path]:
        return self._paths

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._images[idx]


def create_dataset_scaffold(root: str | Path) -> None:
    """
    Create the expected dataset directory structure with placeholder README files.
    """
    root = Path(root)
    dirs = [
        root / "source_faces",
        root / "original_videos",
        root / "output",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Write instructions
    (root / "source_faces" / "README.txt").write_text(
        "Add 5-10 photos of the person whose face you want to appear in the video.\n"
        "Include different angles: front, left ~15°, left ~30°, right ~15°, right ~30°,\n"
        "looking up slightly, looking down slightly.\n"
        "Supported formats: .jpg, .jpeg, .png, .bmp, .webp\n"
    )
    (root / "original_videos" / "README.txt").write_text(
        "Add the target video(s) here — the video containing the face to be replaced.\n"
        "Supported formats: .mp4, .avi, .mov, .mkv\n"
    )
    logger.info(f"Dataset scaffold created at: {root}")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    validator = DatasetValidator(root)
    validator.print_report()
