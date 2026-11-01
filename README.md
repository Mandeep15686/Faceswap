# FaceSwap Pipeline v2 — DreamID Architecture

A production-grade, real-time face swap pipeline implementing 14 sequential stages with temporal coherence, occlusion handling, and region-aware color correction.

---

## Architecture Overview

```
INPUT  → [01] Video Decoder
         [02] Face Detection    (RetinaFace)
         [03] Face Tracking     (GHOST / Kalman)
            ↓ [QUALITY GATE]
         [04] 3D Geometry       (3DDFA-V2)
         [05] Landmark Smooth   (One-Euro Filter)
         [06] Face Alignment    (Similarity Transform)
         [07] Identity Embed    (ArcFace iResNet-100)
         [08] Face Swap         (DreamID distilled 4-step DDIM) ←──┐
         [09] Face Parsing      (BiSeNet)                           │ prev frame
         [10] Occlusion Detect  (Depth-delta alpha)                 │ feedback
         [11] Enhancement       (CodeFormer)                        │
         [12] Color Transfer    (LAB histogram, region-aware)       │
         [13] Seamless Blend    (Laplacian + Poisson)               │
         [14] Video Encoder     (H.264 / HEVC)  ───────────────────┘
OUTPUT
```

**Target:** 24 FPS · GPU: A100 / RTX 4090 · ~42ms per frame

---

## Dataset Structure

```
dataset/
├── source_faces/          ← Multi-angle photos of the identity you want IN the video
│   ├── front.jpg
│   ├── left_15.jpg
│   ├── left_30.jpg
│   ├── right_15.jpg
│   ├── right_30.jpg
│   ├── up_15.jpg
│   └── down_15.jpg
├── original_videos/       ← Target videos (the face to be REPLACED)
│   └── target.mp4
└── output/               ← Swapped videos appear here
```

---

## Installation

```bash
# 1. Clone / unzip this project
cd faceswap_pipeline_v2

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights
python scripts/download_weights.py
```

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 8 GB    | 24 GB (A100/4090) |
| RAM       | 16 GB   | 32 GB |
| CUDA      | 11.8+   | 12.1+ |

---

## Quick Start

```bash
# Basic face swap
python main.py \
  --source_dir  dataset/source_faces/ \
  --target_video dataset/original_videos/target.mp4 \
  --output       dataset/output/swapped.mp4

# With custom config
python main.py \
  --source_dir   dataset/source_faces/ \
  --target_video dataset/original_videos/target.mp4 \
  --output       dataset/output/swapped.mp4 \
  --config       configs/high_quality.yaml

# Preview (no encode, show window)
python main.py \
  --source_dir   dataset/source_faces/ \
  --target_video dataset/original_videos/target.mp4 \
  --preview
```

---

## Configuration

Edit `configs/default.yaml` or create a custom profile:

```yaml
detection:
  confidence_threshold: 0.9
  nms_threshold: 0.4

quality_gate:
  max_yaw_deg:   40
  max_pitch_deg: 40
  min_blur_score: 0.15
  min_track_age:  3

swap:
  ddim_steps: 4          # 4 = realtime, 50 = max quality
  temporal_weight: 0.3   # prev-frame conditioning strength

enhancement:
  codeformer_fidelity: 0.7  # 0=quality, 1=fidelity

output:
  codec: h264            # h264 | hevc
  fps: 24
  crf: 18
```

---

## Pipeline Stage Details

| # | Stage | Model | Latency |
|---|-------|-------|---------|
| 01 | Video Decoder | NVDEC (hardware) | — |
| 02 | Face Detection | RetinaFace | ~2ms |
| 03 | Face Tracking | GHOST + Kalman | ~1ms |
| — | Quality Gate | Rule-based | <1ms |
| 04 | 3D Geometry | 3DDFA-V2 | ~4ms |
| 05 | Landmark Smooth | One-Euro Filter | <1ms |
| 06 | Face Alignment | Similarity Transform | <1ms |
| 07 | Identity Embed | ArcFace iResNet-100 | ~3ms |
| 08 | Face Swap | DreamID (4-step DDIM) | ~18ms |
| 09 | Face Parsing | BiSeNet | ~2ms |
| 10 | Occlusion Detect | Depth-delta alpha | ~1ms |
| 11 | Enhancement | CodeFormer | ~10ms |
| 12 | Color Transfer | LAB histogram | ~1ms |
| 13 | Seamless Blend | Laplacian + Poisson | ~2ms |
| 14 | Video Encoder | H.264 / HEVC | — |
| **Total** | | | **~42ms** |

---

## Troubleshooting

**OOM on large videos:** Reduce `batch_size` in config or use `--half` for FP16 inference.

**Flickering output:** Increase `temporal_weight` in config (default 0.3 → try 0.5).

**Poor identity transfer:** Add more diverse angle photos to `source_faces/`. Minimum 5 angles recommended.

**Low FPS:** Enable TensorRT export via `python scripts/export_trt.py`.
