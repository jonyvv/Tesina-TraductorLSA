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

        # IMPORTANTE: validar TODO antes de tocar `self`. Si se asigna
        # `self.model` primero y después una validación falla, el modelo queda
        # cargado igual: /health reporta `modelo_cargado: true`, `predecir()`
        # funciona, y el backend termina sirviendo justo el modelo que se acaba
        # de rechazar. O sea, los chequeos no protegían nada. Se arma todo en
        # variables locales y recién al final se compromete el estado.
        model = data["model"]
        label_encoder = data["label_encoder"]
        clases = list(label_encoder.classes_)
        version_entrenamiento = data.get("feature_version", "desconocida")

        if version_entrenamiento != FEATURE_VERSION:
            raise RuntimeError(
                f"Desalineación de esquema de features: el modelo fue entrenado con "
                f"'{version_entrenamiento}' pero el backend usa "
                f"'{FEATURE_VERSION}'. Re-entrená el modelo con ml/train.py."
            )

        expected_len = data.get("feature_vector_length")
        if expected_len is not None and expected_len != FEATURE_VECTOR_LENGTH:
            raise RuntimeError(
                f"El modelo espera vectores de longitud {expected_len}, pero el "
                f"backend genera vectores de longitud {FEATURE_VECTOR_LENGTH}."
            )

        # Chequeo de sanidad: que el modelo haya visto TODAS las clases del
        # LabelEncoder durante el entrenamiento. Si el split de train/test dejó
        # alguna clase entera fuera del set de entrenamiento (pasa cuando cada
        # etiqueta se capturó en una sola sesión y el split agrupa por sesión),
        # el modelo queda entrenado con menos clases de las que dice tener y
        # predice siempre lo mismo, con confianza 1.0 y sin ningún error visible.
        # Preferimos fallar acá, en el arranque, que servir eso en la demo.
        clases_vistas = getattr(model, "classes_", None)
        if clases_vistas is not None and len(clases_vistas) < len(clases):
            vistas = {int(c) for c in clases_vistas}
            faltantes = [c for i, c in enumerate(clases) if i not in vistas]
            raise RuntimeError(
                f"El modelo fue entrenado con solo {len(clases_vistas)} de las "
                f"{len(clases)} clases del dataset. Nunca va a poder predecir: "
                f"{faltantes}.\n"
                f"Causa típica: cada etiqueta fue capturada en una sola sesión, así "
                f"que el split agrupado por sujeto+sesión separó por CLASE en vez de "
                f"por sesión. Capturá cada seña en al menos 2 sesiones distintas y "
                f"volvé a correr ml/train.py."
            )

        # Las columnas de `predict_proba` tienen que corresponderse 1 a 1 con
        # `model.classes_`; `_etiqueta_de_indice` depende de esa correspondencia.
        # No siempre se cumple: MLPClassifier entrenado con una sola clase
        # devuelve DOS columnas y un `classes_` de un elemento, y ahí el argmax
        # puede indexar fuera de rango en pleno WebSocket. Se comprueba acá, con
        # un vector de prueba, para que falle en el arranque y no en la demo.
        try:
            n_columnas = len(model.predict_proba([np.zeros(FEATURE_VECTOR_LENGTH,
                                                           dtype=np.float32)])[0])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"El modelo no pudo predecir sobre un vector de prueba de "
                f"longitud {FEATURE_VECTOR_LENGTH}: {exc}"
            ) from exc

        if n_columnas != len(clases_vistas if clases_vistas is not None else clases):
            raise RuntimeError(
                f"El modelo devuelve {n_columnas} probabilidades pero declara "
                f"{len(clases_vistas)} clase(s). Traducir la predicción a una "
                f"etiqueta sería adivinar.\n"
                f"Causa típica: se entrenó un MLPClassifier con una sola clase. "
                f"Entrená con al menos dos señas distintas (ver ml/train.py)."
            )

        # Todo validado: recién ahora el modelo pasa a estar disponible.
        self.model = model
        self.label_encoder = label_encoder
        self.clases = clases
        self.feature_version_entrenamiento = version_entrenamiento

    def _etiqueta_de_indice(self, idx_columna: int) -> str:
        """Traduce un índice de COLUMNA de `predict_proba` a la etiqueta real.

        Ojo: las columnas de `predict_proba` corresponden a `model.classes_`, NO a
        `0..n-1`. Coinciden solo si el modelo vio todas las clases al entrenar; si
        vio un subconjunto, indexar el LabelEncoder directamente con el argmax
        devuelve la etiqueta EQUIVOCADA (sin error, solo mal). Por eso siempre
        pasamos por `model.classes_` primero.
        """
        clase_codificada = self.model.classes_[idx_columna]
        return str(self.label_encoder.inverse_transform([clase_codificada])[0])

    def predecir(self, features: np.ndarray) -> Prediccion:
        if self.model is None:
            raise RuntimeError("Llamá a cargar() antes de predecir()")

        probas = self.model.predict_proba([features])[0]
        idx = int(np.argmax(probas))

        return Prediccion(
            etiqueta=self._etiqueta_de_indice(idx),
            confianza=float(probas[idx]),
            umbral=self.umbral_confianza,
        )

    def top_n(self, features: np.ndarray, n: int = 3) -> list[Prediccion]:
        """Devuelve las n predicciones más probables, útil para debugging y para
        una futura UI que muestre alternativas (ej. abecedario con letras
        visualmente parecidas)."""
        if self.model is None:
            raise RuntimeError("Llamá a cargar() antes de predecir()")

        probas = self.model.predict_proba([features])[0]
        top_idx = np.argsort(probas)[::-1][:n]
        return [
            Prediccion(
                etiqueta=self._etiqueta_de_indice(int(i)),
                confianza=float(probas[i]),
                umbral=self.umbral_confianza,
            )
            for i in top_idx
        ]
