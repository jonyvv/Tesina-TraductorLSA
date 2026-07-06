# -*- coding: utf-8 -*-
"""
backend/app/modelo_lse.py

Carga el modelo entrenado y expone predicción, tal como aparece en el diagrama
de clases (`ModeloLSE: ruta_modelo, clases, cargar(), predecir(), top_n()`).

Diseño deliberado: esta clase NO sabe nada de MediaPipe ni de OpenCV. Solo
recibe un vector de features (numpy) y devuelve una `Prediccion`. Esto permite
cambiar de Random Forest a un MLP o a un LSTM sin tocar el resto del backend
(WebSocketHandler / TraductorService quedan intactos) — ver docs/ARQUITECTURA.md,
sección "Cómo migrar de RandomForest a un modelo de Deep Learning".
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.features import FEATURE_VECTOR_LENGTH, FEATURE_VERSION  # noqa: E402

from .prediccion import Prediccion  # noqa: E402


class ModeloLSE:
    def __init__(self, ruta_modelo: str, umbral_confianza: float = 0.6):
        self.ruta_modelo = ruta_modelo
        self.umbral_confianza = umbral_confianza
        self.model = None
        self.label_encoder = None
        self.clases: list[str] = []
        self.feature_version_entrenamiento: str | None = None

    def cargar(self) -> None:
        """Carga el modelo desde disco (.joblib, exportado por ml/train.py).

        Valida que el modelo se haya entrenado con la MISMA versión del esquema
        de features que corre actualmente en el backend. Si no coincide, falla
        explícitamente en el arranque en vez de servir predicciones basura
        silenciosamente (este chequeo no existía en el prototipo original y es
        justamente el tipo de bug que causa "el modelo predice cualquier cosa"
        sin ningún error visible).
        """
        path = Path(self.ruta_modelo)
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el modelo en '{self.ruta_modelo}'. "
                f"Corré primero ml/train.py para generarlo."
            )

        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.clases = list(self.label_encoder.classes_)
        self.feature_version_entrenamiento = data.get("feature_version", "desconocida")

        if self.feature_version_entrenamiento != FEATURE_VERSION:
            raise RuntimeError(
                f"Desalineación de esquema de features: el modelo fue entrenado con "
                f"'{self.feature_version_entrenamiento}' pero el backend usa "
                f"'{FEATURE_VERSION}'. Re-entrená el modelo con ml/train.py."
            )

        expected_len = data.get("feature_vector_length")
        if expected_len is not None and expected_len != FEATURE_VECTOR_LENGTH:
            raise RuntimeError(
                f"El modelo espera vectores de longitud {expected_len}, pero el "
                f"backend genera vectores de longitud {FEATURE_VECTOR_LENGTH}."
            )

    def predecir(self, features: np.ndarray) -> Prediccion:
        if self.model is None:
            raise RuntimeError("Llamá a cargar() antes de predecir()")

        probas = self.model.predict_proba([features])[0]
        idx = int(np.argmax(probas))
        confianza = float(probas[idx])
        etiqueta = self.label_encoder.inverse_transform([idx])[0]

        return Prediccion(etiqueta=str(etiqueta), confianza=confianza, umbral=self.umbral_confianza)

    def top_n(self, features: np.ndarray, n: int = 3) -> list[Prediccion]:
        """Devuelve las n predicciones más probables, útil para debugging y para
        una futura UI que muestre alternativas (ej. abecedario con letras
        visualmente parecidas)."""
        if self.model is None:
            raise RuntimeError("Llamá a cargar() antes de predecir()")

        probas = self.model.predict_proba([features])[0]
        top_idx = np.argsort(probas)[::-1][:n]
        etiquetas = self.label_encoder.inverse_transform(top_idx)
        return [
            Prediccion(etiqueta=str(etq), confianza=float(probas[i]), umbral=self.umbral_confianza)
            for etq, i in zip(etiquetas, top_idx)
        ]
