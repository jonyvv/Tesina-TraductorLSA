# -*- coding: utf-8 -*-
"""
Adaptador para los modelos de scikit-learn serializados con joblib
(Random Forest / MLP de ml/train.py).

Acá viven las validaciones que antes estaban en `ModeloLSE.cargar()`: son
específicas de scikit-learn, así que su lugar es el adaptador. Todas son
regresiones de bugs reales — ver tests/test_modelo_lse.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))

from common.features import FEATURE_VECTOR_LENGTH  # noqa: E402

from .base import ModeloAdaptador  # noqa: E402


class SklearnJoblibAdapter(ModeloAdaptador):
    def __init__(self, ruta_modelo: Path):
        self.ruta_modelo = ruta_modelo
        self.model = None
        self._label_encoder = None
        self._clases: list[str] = []

    @classmethod
    def desde_objetos(cls, model, label_encoder) -> "SklearnJoblibAdapter":
        """Arma el adaptador sobre objetos ya en memoria, sin pasar por disco.

        Existe para los tests, que necesitan construir modelos deliberadamente
        rotos (ej. entrenados con un subconjunto de las clases) sin que la
        validación de `cargar()` los rechace antes de poder probar el mapeo.
        """
        adaptador = cls(ruta_modelo=Path("(en memoria)"))
        adaptador._comprometer(model, label_encoder)
        return adaptador

    def cargar(self) -> dict:
        data = joblib.load(self.ruta_modelo)
        model = data["model"]
        label_encoder = data["label_encoder"]

        self._validar(model, label_encoder)
        self._comprometer(model, label_encoder)
        return data

    # --- Validación ---------------------------------------------------------

    @staticmethod
    def _validar(model, label_encoder) -> None:
        """Rechaza modelos que se servirían mal en silencio. Corre ANTES de
        tocar el estado del adaptador, así un modelo rechazado no queda cargado."""
        clases_dataset = list(label_encoder.classes_)
        clases_vistas = getattr(model, "classes_", None)

        # Que el modelo haya visto TODAS las clases del LabelEncoder durante el
        # entrenamiento. Si el split de train/test dejó alguna clase entera
        # fuera del set de entrenamiento (pasa cuando cada etiqueta se capturó
        # en una sola sesión y el split agrupa por sesión), el modelo queda
        # entrenado con menos clases de las que dice tener y predice siempre lo
        # mismo, con confianza 1.0 y sin ningún error visible. Preferimos fallar
        # acá, en el arranque, que servir eso en la demo.
        if clases_vistas is not None and len(clases_vistas) < len(clases_dataset):
            vistas = {int(c) for c in clases_vistas}
            faltantes = [c for i, c in enumerate(clases_dataset) if i not in vistas]
            raise RuntimeError(
                f"El modelo fue entrenado con solo {len(clases_vistas)} de las "
                f"{len(clases_dataset)} clases del dataset. Nunca va a poder predecir: "
                f"{faltantes}.\n"
                f"Causa típica: cada etiqueta fue capturada en una sola sesión, así "
                f"que el split agrupado por sujeto+sesión separó por CLASE en vez de "
                f"por sesión. Capturá cada seña en al menos 2 sesiones distintas y "
                f"volvé a correr ml/train.py."
            )

        # Las columnas de `predict_proba` tienen que corresponderse 1 a 1 con
        # `model.classes_`; el mapeo de índice a etiqueta depende de eso. No
        # siempre se cumple: un MLPClassifier entrenado con una sola clase
        # devuelve DOS columnas y un `classes_` de un elemento, y ahí el argmax
        # puede indexar fuera de rango en pleno WebSocket. Se comprueba acá, con
        # un vector de prueba, para que falle en el arranque y no en la demo.
        try:
            n_columnas = len(
                model.predict_proba(
                    [np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32)]
                )[0]
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"El modelo no pudo predecir sobre un vector de prueba de "
                f"longitud {FEATURE_VECTOR_LENGTH}: {exc}"
            ) from exc

        n_declaradas = len(clases_vistas if clases_vistas is not None else clases_dataset)
        if n_columnas != n_declaradas:
            raise RuntimeError(
                f"El modelo devuelve {n_columnas} probabilidades pero declara "
                f"{n_declaradas} clase(s). Traducir la predicción a una "
                f"etiqueta sería adivinar.\n"
                f"Causa típica: se entrenó un MLPClassifier con una sola clase. "
                f"Entrená con al menos dos señas distintas (ver ml/train.py)."
            )

    def _comprometer(self, model, label_encoder) -> None:
        # `clases` tiene que quedar en el ORDEN DE LAS COLUMNAS de
        # `predict_proba`, que es `model.classes_` y NO `0..n-1`. Coinciden solo
        # si el modelo vio todas las clases al entrenar; si vio un subconjunto,
        # indexar el LabelEncoder con el argmax devuelve la etiqueta EQUIVOCADA
        # (sin error, solo mal). Alineándolas acá, el resto del backend puede
        # hacer `clases[argmax]` sin saber nada de esto.
        clases_vistas = getattr(model, "classes_", None)
        if clases_vistas is not None:
            self._clases = [
                str(label_encoder.inverse_transform([c])[0]) for c in clases_vistas
            ]
        else:
            self._clases = [str(c) for c in label_encoder.classes_]

        # Un Random Forest con `n_jobs=-1` reparte los árboles entre todos los
        # hilos, pero acá se predice de a UNA muestra por frame: el overhead de
        # joblib supera al cómputo. Medido sobre el modelo de 200 árboles del
        # proyecto: 55 ms con n_jobs=-1 contra 19,6 ms con n_jobs=1, o sea más
        # que todo MediaPipe junto. El modelo entrenado no cambia; solo cambia
        # cómo se evalúa.
        if getattr(model, "n_jobs", None) not in (None, 1):
            model.n_jobs = 1

        self.model = model
        self._label_encoder = label_encoder

    # --- Inferencia ---------------------------------------------------------

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba([features])[0], dtype=np.float32)

    def predict_proba_secuencia(self, secuencia: np.ndarray) -> np.ndarray:
        raise NotImplementedError("El modelo scikit-learn no trabaja con secuencias.")

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def clases(self) -> list[str]:
        return self._clases
