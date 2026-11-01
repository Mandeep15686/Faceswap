"""
pipeline/__init__.py
─────────────────────
Public API for all 14 pipeline stages.
"""

from .base import (
    AlignedFace,
    BoundingBox,
    FaceGeometry,
    FrameContext,
    PipelineStage,
)
from .stage01_decoder      import VideoDecoder
from .stage02_detector     import FaceDetector
from .stage03_tracker      import FaceTracker
from .quality_gate         import QualityGate
from .stage04_geometry     import GeometryEstimator
from .stage05_smoother     import LandmarkSmoother
from .stage06_alignment    import FaceAligner
from .stage07_identity     import IdentityEmbedder
from .stage08_dreamid      import DreamIDSwapper
from .stage09_parsing      import FaceParser
from .stage10_occlusion    import OcclusionDetector
from .stage11_enhancement  import FaceEnhancer
from .stage12_color_transfer import ColorTransfer
from .stage13_blending     import SeamlessBlender
from .stage14_encoder      import VideoEncoder

__all__ = [
    # Data structures
    "FrameContext",
    "BoundingBox",
    "FaceGeometry",
    "AlignedFace",
    "PipelineStage",
    # Stages
    "VideoDecoder",
    "FaceDetector",
    "FaceTracker",
    "QualityGate",
    "GeometryEstimator",
    "LandmarkSmoother",
    "FaceAligner",
    "IdentityEmbedder",
    "DreamIDSwapper",
    "FaceParser",
    "OcclusionDetector",
    "FaceEnhancer",
    "ColorTransfer",
    "SeamlessBlender",
    "VideoEncoder",
]
