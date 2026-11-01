"""
main.py
────────
FaceSwap Pipeline v2 — Main Entry Point

Wires all 14 stages together, handles dataset loading,
source embedding pre-computation, and the main frame loop.

Usage
─────
python main.py \
  --source_dir  dataset/source_faces/ \
  --target_video dataset/original_videos/target.mp4 \
  --output       dataset/output/swapped.mp4 \
  [--config      configs/default.yaml] \
  [--preview]    [--start_frame N] [--end_frame N]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
import cv2
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from pipeline import (
    ColorTransfer,
    FaceAligner,
    FaceDetector,
    FaceEnhancer,
    FaceParser,
    FaceTracker,
    GeometryEstimator,
    IdentityEmbedder,
    LandmarkSmoother,
    OcclusionDetector,
    QualityGate,
    SeamlessBlender,
    VideoDecoder,
    VideoEncoder,
)
from pipeline.stage08_dreamid import DreamIDSwapper


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML config, merging with defaults."""
    defaults_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(defaults_path) as f:
        cfg = yaml.safe_load(f)

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        # Deep merge
        cfg = _deep_merge(cfg, overrides)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(cfg: dict, device: torch.device) -> dict:
    """Instantiate, load, and return all processing stages."""
    stages = {
        "detector":      FaceDetector(cfg["detection"], device),
        "tracker":       FaceTracker(cfg["tracking"], device),
        "gate":          QualityGate(cfg["quality_gate"], device),
        "geometry":      GeometryEstimator(cfg["geometry"], device),
        "smoother":      LandmarkSmoother(cfg["smoothing"], device),
        "aligner":       FaceAligner(cfg["alignment"], device),
        "identity":      IdentityEmbedder(cfg["identity"], device),
        "swapper":       DreamIDSwapper(cfg["swap"], device),
        "parser":        FaceParser(cfg["parsing"], device),
        "occlusion":     OcclusionDetector(cfg["occlusion"], device),
        "enhancer":      FaceEnhancer(cfg["enhancement"], device),
        "color":         ColorTransfer(cfg["color_transfer"], device),
        "blender":       SeamlessBlender(cfg["blending"], device),
    }

    logger.info("Loading pipeline models ...")
    for name, stage in stages.items():
        logger.info(f"  Loading {name} ...")
        stage.load()

    return stages


# ─────────────────────────────────────────────────────────────────────────────
# Source dataset loader
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_source_images(source_dir: str) -> list[Path]:
    """Return sorted list of image paths from source_faces directory."""
    d = Path(source_dir)
    if not d.exists():
        raise FileNotFoundError(f"Source directory not found: {d}")
    paths = sorted([p for p in d.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    if not paths:
        raise ValueError(
            f"No face images found in {d}. "
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )
    logger.info(f"Found {len(paths)} source face images: {[p.name for p in paths]}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    source_dir: str,
    target_video: str,
    output: str,
    config_path: str,
    preview: bool,
    start_frame: int,
    end_frame: Optional[int],
) -> None:
    # ── Setup ─────────────────────────────────────────────────────────────────
    cfg = load_config(config_path)
    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if cfg.get("dtype") == "float16" and device.type == "cuda":
        torch.set_default_dtype(torch.float16)

    # ── Build stages ──────────────────────────────────────────────────────────
    stages = build_pipeline(cfg, device)
    identity_stage: IdentityEmbedder = stages["identity"]
    swapper_stage:  DreamIDSwapper   = stages["swapper"]

    # ── Pre-compute source embedding ──────────────────────────────────────────
    source_paths = load_source_images(source_dir)
    logger.info("Pre-computing source identity embedding ...")
    source_embedding = identity_stage.build_source_embedding(source_paths)
    swapper_stage._source_embedding = source_embedding
    logger.info(f"Source embedding ready (dim={len(source_embedding)}).")

    # ── Ensure output directory exists ────────────────────────────────────────
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # ── Main decode → process → encode loop ──────────────────────────────────
    with VideoDecoder(
        target_video,
        hw_decode=(device.type == "cuda"),
        start_frame=start_frame,
        end_frame=end_frame,
    ) as decoder:

        with VideoEncoder(output, decoder, cfg["output"]) as encoder:

            total = (end_frame or decoder.total_frames) - start_frame
            pbar = tqdm(total=total, desc="Processing frames", unit="fr")

            gate_fails = 0
            stage_order = [
                "detector", "tracker", "gate",
                "geometry", "smoother", "aligner",
                "identity", "swapper",
                "parser", "occlusion", "enhancer", "color", "blender",
            ]

            for ctx in decoder:
                # Inject source embedding reference into context
                ctx._pipeline_source_embedding = source_embedding

                # Run all stages in order
                for name in stage_order:
                    ctx = stages[name](ctx)

                # Write frame
                encoder.write(ctx)

                # Preview window
                if preview:
                    _show_preview(ctx)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Preview quit by user.")
                        break

                if not ctx.gate_passed:
                    gate_fails += 1

                pbar.update(1)
                pbar.set_postfix({
                    "gate_fails": gate_fails,
                    "ms": f"{sum(ctx.stage_times.values()):.0f}",
                })

            pbar.close()

    if preview:
        cv2.destroyAllWindows()

    logger.info(f"Done. Output: {output} | Gate failures: {gate_fails}/{total}")


# ─────────────────────────────────────────────────────────────────────────────
# Preview helper
# ─────────────────────────────────────────────────────────────────────────────

def _show_preview(ctx) -> None:
    """Display side-by-side original vs composited in a window."""
    orig  = ctx.image
    final = ctx.composited_frame if ctx.composited_frame is not None else orig

    h = max(orig.shape[0], final.shape[0])
    orig_r  = cv2.resize(orig,  (int(orig.shape[1]  * h / orig.shape[0]),  h))
    final_r = cv2.resize(final, (int(final.shape[1] * h / final.shape[0]), h))

    combined = cv2.hconcat([orig_r, final_r])

    # Overlay timing
    total_ms = sum(ctx.stage_times.values())
    cv2.putText(
        combined, f"Frame {ctx.frame_idx} | {total_ms:.0f}ms",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.imshow("FaceSwap Pipeline v2  |  Original (L)  vs  Swapped (R)  |  q to quit", combined)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--source_dir",   required=True,  help="Directory of multi-angle source face images.")
@click.option("--target_video", required=True,  help="Input video file to perform face swap on.")
@click.option("--output",       required=True,  help="Output video file path.")
@click.option("--config",       default=None,   help="Path to YAML config file (optional).")
@click.option("--preview",      is_flag=True,   default=False, help="Show live preview window.")
@click.option("--start_frame",  default=0,      type=int, help="Start from this frame index.")
@click.option("--end_frame",    default=None,   type=int, help="Stop at this frame index.")
@click.option("--verbose",      is_flag=True,   default=False, help="Enable DEBUG logging.")
def cli(
    source_dir: str,
    target_video: str,
    output: str,
    config: Optional[str],
    preview: bool,
    start_frame: int,
    end_frame: Optional[int],
    verbose: bool,
) -> None:
    """FaceSwap Pipeline v2 — DreamID Architecture."""
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    run_pipeline(
        source_dir=source_dir,
        target_video=target_video,
        output=output,
        config_path=config,
        preview=preview,
        start_frame=start_frame,
        end_frame=end_frame,
    )


if __name__ == "__main__":
    cli()
