from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ModeloAdaptador(ABC):
    """Interfaz común a todos los formatos de modelo que sabe servir el backend.

    Contrato que TODO adaptador tiene que cumplir: `clases[i]` es la etiqueta de
    la columna `i` que devuelve `predict_proba`. No es automático (en
    scikit-learn las columnas siguen a `model.classes_`, no al LabelEncoder), y
    romperlo no lanza ningún error: simplemente devuelve la seña equivocada.
    """

    @abstractmethod
    def cargar(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba_secuencia(self, secuencia: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def requiere_secuencia(self) -> bool:
        return False

    @property
    def ventana_inferencia(self) -> int:
        return 1

    @property
    def label_encoder(self):
        """Solo los modelos de scikit-learn traen uno; el resto devuelve None."""
        return None

    @property
    @abstractmethod
    def clases(self) -> list[str]:
        raise NotImplementedError
