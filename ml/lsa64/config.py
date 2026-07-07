from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.features import FEATURE_VECTOR_LENGTH, FEATURE_VERSION

DEFAULT_MAX_FRAMES = 120
DEFAULT_FRAME_STEP = 2
DEFAULT_MIN_SEQUENCE_LEN = 8
DATASET_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MODELS_DIR = Path(__file__).resolve().parents[2] / "backend" / "models"


@dataclass(frozen=True)
class LSA64TrainingConfig:
    dataset_dir: Path
    annotations: Path | None
    output: Path
    epochs: int = 40
    hidden_size: int = 128
    batch_size: int = 8
    frame_step: int = DEFAULT_FRAME_STEP
    max_frames: int = DEFAULT_MAX_FRAMES
    min_seq_len: int = DEFAULT_MIN_SEQUENCE_LEN
    keep_empty_frames: bool = False
    seed: int = 42
    feature_version: str = FEATURE_VERSION
    feature_vector_length: int = FEATURE_VECTOR_LENGTH

