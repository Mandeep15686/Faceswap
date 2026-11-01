"""
scripts/benchmark.py
─────────────────────
Profile per-stage latency over N frames and print a timing breakdown table.

Usage
─────
python scripts/benchmark.py \
  --source_dir   dataset/source_faces/ \
  --target_video dataset/original_videos/target.mp4 \
  --frames       100
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import build_pipeline, load_config, load_source_images
from pipeline import VideoDecoder


@click.command()
@click.option("--source_dir",   required=True)
@click.option("--target_video", required=True)
@click.option("--frames",       default=100, type=int, help="Number of frames to benchmark.")
@click.option("--config",       default=None)
def cli(source_dir: str, target_video: str, frames: int, config: str | None) -> None:
    """Benchmark pipeline stage latencies."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    cfg    = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stages = build_pipeline(cfg, device)

    # Pre-compute source embedding
    from pipeline.stage07_identity import IdentityEmbedder
    from pipeline.stage08_dreamid  import DreamIDSwapper

    identity: IdentityEmbedder = stages["identity"]
    swapper:  DreamIDSwapper   = stages["swapper"]
    src_paths = load_source_images(source_dir)
    src_emb = identity.build_source_embedding(src_paths)
    swapper._source_embedding = src_emb

    # Collect timings
    accumulated: dict[str, list[float]] = {}
    stage_order = [
        "detector", "tracker", "gate",
        "geometry", "smoother", "aligner",
        "identity", "swapper",
        "parser", "occlusion", "enhancer", "color", "blender",
    ]

    count = 0
    with VideoDecoder(target_video, end_frame=frames) as decoder:
        for ctx in decoder:
            ctx._pipeline_source_embedding = src_emb
            for name in stage_order:
                ctx = stages[name](ctx)
            for k, v in ctx.stage_times.items():
                accumulated.setdefault(k, []).append(v)
            count += 1

    # ── Print results ──────────────────────────────────────────────────────────
    console = Console()
    table = Table(title=f"Pipeline Latency — {count} frames — device={device}")
    table.add_column("Stage",   style="cyan",  no_wrap=True)
    table.add_column("Mean ms", style="green", justify="right")
    table.add_column("P95 ms",  style="yellow", justify="right")
    table.add_column("Max ms",  style="red",   justify="right")
    table.add_column("% total", justify="right")

    total_mean = sum(np.mean(v) for v in accumulated.values())

    for name in stage_order:
        if name not in accumulated:
            continue
        vals = np.array(accumulated[name])
        mean = np.mean(vals)
        p95  = np.percentile(vals, 95)
        mx   = np.max(vals)
        pct  = 100 * mean / total_mean if total_mean else 0
        table.add_row(name, f"{mean:.1f}", f"{p95:.1f}", f"{mx:.1f}", f"{pct:.0f}%")

    table.add_section()
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_mean:.1f}[/bold]", "", "", "")

    console.print(table)


if __name__ == "__main__":
    cli()
