from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ModeloAdaptador(ABC):
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
    @abstractmethod
    def clases(self) -> list[str]:
        raise NotImplementedError
