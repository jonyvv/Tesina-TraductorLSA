from __future__ import annotations

from pathlib import Path

from .base import ModeloAdaptador
from .sklearn_adapter import SklearnJoblibAdapter
from .torch_adapter import TorchCheckpointAdapter


def crear_adaptador_modelo(ruta_modelo: str | Path) -> ModeloAdaptador:
    path = Path(ruta_modelo)
    if path.suffix.lower() == ".joblib":
        return SklearnJoblibAdapter(path)
    if path.suffix.lower() == ".pt":
        return TorchCheckpointAdapter(path)
    raise ValueError(f"Formato de modelo no soportado: {path.suffix}")

