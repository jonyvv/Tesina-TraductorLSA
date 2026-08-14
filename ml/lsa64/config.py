from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.features import FEATURE_VECTOR_LENGTH, FEATURE_VERSION

DEFAULT_MAX_FRAMES = 120
DEFAULT_FRAME_STEP = 2
DEFAULT_MIN_SEQUENCE_LEN = 8
DATASET_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MODELS_DIR = Path(__file__).resolve().parents[2] / "backend" / "models"
# Reportes de experimentos (LOSO, matrices de confusion). Van aparte de
# MODELS_DIR: backend/models/ es lo que carga el backend en produccion, no el
# lugar para artefactos de investigacion.
REPORTS_DIR = Path(__file__).resolve().parents[2] / "ml" / "reports"

# El cache de features vive en .lsa64_cache/ (ya ignorado por git): son ~100 MB
# de landmarks derivados del dataset, no tiene sentido versionarlos.
DEFAULT_CACHE_NAME = "features_lsa64.npz"


def default_cache_path(work_dir: Path | None = None) -> Path:
    base = work_dir or (Path.cwd() / ".lsa64_cache")
    return base / DEFAULT_CACHE_NAME


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
    # Cache de landmarks: si existe y es compatible, el entrenamiento no vuelve
    # a correr MediaPipe. Ver ml/lsa64/cache.py.
    cache_path: Path | None = None
    refresh_cache: bool = False
    workers: int | None = None
    labels_map: Path | None = None
    patience: int = 8
    learning_rate: float = 1e-3

