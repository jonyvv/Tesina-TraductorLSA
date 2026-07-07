# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.features import FEATURE_VECTOR_LENGTH, FEATURE_VERSION  # noqa: E402

from .modelos.factory import crear_adaptador_modelo  # noqa: E402
from .prediccion import Prediccion  # noqa: E402


class ModeloLSE:
    def __init__(self, ruta_modelo: str, umbral_confianza: float = 0.6):
        self.ruta_modelo = ruta_modelo
        self.umbral_confianza = umbral_confianza
        self.adaptador = None
        self.clases: list[str] = []
        self.feature_version_entrenamiento: str | None = None

    @property
    def model(self):
        return None if self.adaptador is None else self.adaptador.model

    @property
    def requiere_secuencia(self) -> bool:
        return False if self.adaptador is None else self.adaptador.requiere_secuencia

    @property
    def ventana_inferencia(self) -> int:
        return 1 if self.adaptador is None else self.adaptador.ventana_inferencia

    def cargar(self) -> None:
        path = Path(self.ruta_modelo)
        if not path.exists():
            raise FileNotFoundError(f"No existe el modelo en '{self.ruta_modelo}'.")

        self.adaptador = crear_adaptador_modelo(path)
        data = self.adaptador.cargar()
        self.clases = list(self.adaptador.clases)
        self.feature_version_entrenamiento = data.get("feature_version", "desconocida")

        if self.feature_version_entrenamiento != FEATURE_VERSION:
            raise RuntimeError(
                f"Desalineación de esquema de features: el modelo fue entrenado con "
                f"'{self.feature_version_entrenamiento}' pero el backend usa "
                f"'{FEATURE_VERSION}'."
            )

        expected_len = data.get("feature_vector_length")
        if expected_len is not None and expected_len != FEATURE_VECTOR_LENGTH:
            raise RuntimeError(
                f"El modelo espera vectores de longitud {expected_len}, pero el "
                f"backend genera vectores de longitud {FEATURE_VECTOR_LENGTH}."
            )

    def predecir(self, features: np.ndarray) -> Prediccion:
        if self.adaptador is None:
            raise RuntimeError("Llamá a cargar() antes de predecir()")

        probas = self.adaptador.predict_proba(features)
        idx = int(np.argmax(probas))
        confianza = float(probas[idx])
        etiqueta = self.clases[idx]
        return Prediccion(etiqueta=str(etiqueta), confianza=confianza, umbral=self.umbral_confianza)

    def predecir_secuencia(self, secuencia: np.ndarray) -> Prediccion:
        if self.adaptador is None:
            raise RuntimeError("Llamá a cargar() antes de predecir()")

        probas = self.adaptador.predict_proba_secuencia(secuencia)
        idx = int(np.argmax(probas))
        confianza = float(probas[idx])
        etiqueta = self.clases[idx]
        return Prediccion(etiqueta=str(etiqueta), confianza=confianza, umbral=self.umbral_confianza)

    def top_n(self, features: np.ndarray, n: int = 3) -> list[Prediccion]:
        if self.adaptador is None:
            raise RuntimeError("Llamá a cargar() antes de predecir()")

        probas = self.adaptador.predict_proba(features)
        top_idx = np.argsort(probas)[::-1][:n]
        return [
            Prediccion(etiqueta=str(self.clases[i]), confianza=float(probas[i]), umbral=self.umbral_confianza)
            for i in top_idx
        ]
