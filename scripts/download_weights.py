"""
scripts/download_weights.py
────────────────────────────
Downloads all required model weights for the FaceSwap Pipeline v2.

Models downloaded
─────────────────
• RetinaFace       — via insightface (auto-download on first use)
• ArcFace          — via insightface (auto-download on first use)
• InsightFace inswapper_128.onnx  — fallback swap model
• 3DDFA-V2         — manual instructions (no redistributable URL)
• BiSeNet          — face parsing weights
• CodeFormer       — blind face restoration
• MiDaS Small      — depth estimation for occlusion
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Weight registry
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "models"

WEIGHTS = {
    "codeformer": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "dest": MODELS_DIR / "codeformer.pth",
        "sha256": None,   # Set to actual hash for production
    },
    "bisenet_face_parsing": {
        "url": "https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth",
        "dest": MODELS_DIR / "bisenet_face_parsing.pt",
        "sha256": None,
    },
    "midas_small": {
        "url": "https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21_small_256.pt",
        "dest": MODELS_DIR / "midas_small.pt",
        "sha256": None,
    },
}


def download_file(url: str, dest: Path, sha256: str | None = None) -> None:
    if dest.exists():
        logger.info(f"  Already exists: {dest.name} — skipping.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading {dest.name} ...")

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = int(count * block_size * 100 / total_size)
            print(f"\r  Progress: {pct:3d}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
        print()   # newline after progress
    except Exception as exc:
        logger.error(f"  Download failed: {exc}")
        if dest.exists():
            dest.unlink()
        return

    # Checksum verification
    if sha256:
        actual = hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual != sha256:
            logger.error(f"  SHA256 mismatch for {dest.name}! Expected {sha256}, got {actual}")
            dest.unlink()
            return

    logger.info(f"  ✓  {dest.name}")


def download_insightface_models() -> None:
    """Trigger insightface auto-download."""
    logger.info("Pre-downloading InsightFace models ...")
    try:
        import insightface
        app = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("  ✓  InsightFace buffalo_l (RetinaFace + ArcFace)")
    except ImportError:
        logger.warning("  insightface not installed: pip install insightface")
    except Exception as exc:
        logger.warning(f"  InsightFace pre-download failed: {exc}")


def print_manual_instructions() -> None:
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MANUAL SETUP REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 3DDFA-V2 (Stage 04 — 3D Geometry)
 ──────────────────────────────────
 1. git clone https://github.com/cleardusk/3DDFA_V2
 2. Follow its README to install and download weights
 3. Place/symlink the repo at:
    faceswap_pipeline_v2/models/3DDFA_V2/

 DreamID (Stage 08 — Face Swap, primary model)
 ──────────────────────────────────────────────
 DreamID is a research model. Weights are available from:
    https://huggingface.co/nandechao/DreamID  (community impl)
 Download and place at:
    faceswap_pipeline_v2/models/dreamid_weights/
 (sub-directories: vae/, unet/, scheduler/, id_projection.pt)

 Without DreamID weights, the pipeline automatically falls back
 to InsightFace inswapper_128 (good quality, no temporal coherence).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    logger.info("FaceSwap Pipeline v2 — Model Weight Downloader")
    logger.info("="*55)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, meta in WEIGHTS.items():
        logger.info(f"[{name}]")
        download_file(meta["url"], meta["dest"], meta.get("sha256"))

    download_insightface_models()
    print_manual_instructions()

    logger.info("Weight download complete.")
