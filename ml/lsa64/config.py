from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.features import (
    FEATURE_VECTOR_LENGTH,
    FEATURE_VERSION,
    feature_vector_length_de,
    feature_version_de,
)

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
    # --- Regularizacion (experimento contra el sobreajuste: loss train 0,064
    # contra val 0,79). Los defaults reproducen EXACTAMENTE el baseline de
    # 79,8 %: dropout 0,2 es el que el modelo ya tenia fijo en el codigo y
    # todo lo demas esta apagado. Una corrida sin flags nuevos sigue dando el
    # mismo numero, que es lo que hace que la comparacion valga.
    dropout: float = 0.2
    weight_decay: float = 0.0
    aug_noise: float = 0.0
    aug_frame_drop: float = 0.0
    aug_time_scale: float = 0.0
    # v2: agrega la posicion de la muneca al vector (138 -> 144). Ver el bloque
    # "Variante v2" de common/features.py.
    incluir_posicion: bool = False

    def __post_init__(self) -> None:
        # feature_version y feature_vector_length son DERIVADOS: no se pueden
        # contradecir con incluir_posicion. Si quedaran sueltos, alcanzaria con
        # olvidarse un flag para entrenar sobre un cache v2 declarando v1 y que
        # el chequeo de compatibilidad no avise nada.
        object.__setattr__(self, "feature_version", feature_version_de(self.incluir_posicion))
        object.__setattr__(
            self, "feature_vector_length", feature_vector_length_de(self.incluir_posicion)
        )

