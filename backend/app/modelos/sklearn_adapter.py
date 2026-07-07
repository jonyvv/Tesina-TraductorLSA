from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .base import ModeloAdaptador


class SklearnJoblibAdapter(ModeloAdaptador):
    def __init__(self, ruta_modelo: Path):
        self.ruta_modelo = ruta_modelo
        self.model = None
        self._clases: list[str] = []

    def cargar(self) -> dict:
        data = joblib.load(self.ruta_modelo)
        self.model = data["model"]
        label_encoder = data["label_encoder"]
        self._clases = list(label_encoder.classes_)
        return data

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba([features])[0], dtype=np.float32)

    def predict_proba_secuencia(self, secuencia: np.ndarray) -> np.ndarray:
        raise NotImplementedError("El modelo scikit-learn no trabaja con secuencias.")

    @property
    def clases(self) -> list[str]:
        return self._clases
