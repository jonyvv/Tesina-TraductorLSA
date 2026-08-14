# -*- coding: utf-8 -*-
"""
backend/app/modelo_lse.py

Carga el modelo entrenado y expone predicción, tal como aparece en el diagrama
de clases (`ModeloLSE: ruta_modelo, clases, cargar(), predecir(), top_n()`).

Diseño deliberado: esta clase NO sabe nada de MediaPipe ni de OpenCV, y desde
la integración con LSA64 tampoco sabe en qué formato está serializado el modelo:
eso lo resuelve `modelos/factory.py` según la extensión del archivo (`.joblib`
-> scikit-learn, `.pt` -> checkpoint de PyTorch). Cambiar de Random Forest a la
BiLSTM no toca `TraductorService`, `WebSocketHandler` ni el frontend — es
exactamente la migración descripta en docs/ARQUITECTURA.md §7.

`ModeloLSE` queda entonces con dos responsabilidades: elegir el adaptador y
**validar** que el modelo cargado sea servible. La validación es la parte que no
se puede delegar: es la que evita que el backend sirva silenciosamente un modelo
roto (ver los tests de regresión en tests/test_modelo_lse.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.features import FEATURE_VECTOR_LENGTH, FEATURE_VERSION  # noqa: E402

from .modelos.base import ModeloAdaptador  # noqa: E402
from .modelos.factory import crear_adaptador_modelo  # noqa: E402
from .prediccion import Prediccion  # noqa: E402


class ModeloLSE:
    def __init__(self, ruta_modelo: str, umbral_confianza: float = 0.6):
        self.ruta_modelo = ruta_modelo
        self.umbral_confianza = umbral_confianza
        self.adaptador: ModeloAdaptador | None = None
        self.feature_version_entrenamiento: str | None = None

    # --- Estado derivado del adaptador -------------------------------------
    # Se exponen como propiedades y no como atributos copiados para que haya una
    # sola fuente de verdad. Si `clases` fuera una copia, un adaptador que
    # reordena las clases (ver SklearnJoblibAdapter) dejaría a `predecir()`
    # traduciendo índices contra una lista desactualizada — que es justo el bug
    # de mapeo que documenta tests/test_modelo_lse.py.

    @property
    def model(self):
        return None if self.adaptador is None else self.adaptador.model

    @property
    def label_encoder(self):
        return None if self.adaptador is None else self.adaptador.label_encoder

    @property
    def clases(self) -> list[str]:
        # `str()` explícito: los LabelEncoder de scikit-learn devuelven `np.str_`,
        # que en /health y en los logs se imprime como `np.str_('clase_01')`.
        return [] if self.adaptador is None else [str(c) for c in self.adaptador.clases]

    @property
    def requiere_secuencia(self) -> bool:
        return False if self.adaptador is None else self.adaptador.requiere_secuencia

    @property
    def ventana_inferencia(self) -> int:
        return 1 if self.adaptador is None else self.adaptador.ventana_inferencia

    # --- Carga --------------------------------------------------------------

    def cargar(self) -> None:
        """Carga el modelo desde disco (`.joblib` de ml/train.py o `.pt` de
        ml/train_lsa64.py).

        Valida que el modelo se haya entrenado con la MISMA versión del esquema
        de features que corre actualmente en el backend. Si no coincide, falla
        explícitamente en el arranque en vez de servir predicciones basura
        silenciosamente (este chequeo no existía en el prototipo original y es
        justamente el tipo de bug que causa "el modelo predice cualquier cosa"
        sin ningún error visible).

        IMPORTANTE: se valida TODO antes de tocar `self`. Si se asignara el
        adaptador primero y después una validación fallara, el modelo quedaría
        cargado igual: /health reportaría `modelo_cargado: true`, `predecir()`
        funcionaría, y el backend terminaría sirviendo justo el modelo que se
        acaba de rechazar. O sea, los chequeos no protegerían nada. Se arma todo
        en variables locales y recién al final se compromete el estado.
        """
        path = Path(self.ruta_modelo)
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el modelo en '{self.ruta_modelo}'. "
                f"Corré primero ml/train.py (o ml/train_lsa64.py) para generarlo."
            )

        adaptador = crear_adaptador_modelo(path)
        # Cada adaptador valida acá lo que es propio de su formato (ver
        # SklearnJoblibAdapter: clases faltantes y columnas de predict_proba).
        data = adaptador.cargar()

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

        # Todo validado: recién ahora el modelo pasa a estar disponible.
        self.adaptador = adaptador
        self.feature_version_entrenamiento = version_entrenamiento

    # --- Inferencia ---------------------------------------------------------

    def _a_prediccion(self, probas: np.ndarray) -> Prediccion:
        """Traduce el vector de probabilidades a una `Prediccion`.

        El índice del argmax es un índice de COLUMNA, y cada adaptador garantiza
        que `clases[i]` sea la etiqueta de la columna `i`. No es gratis: en
        scikit-learn las columnas corresponden a `model.classes_`, no a
        `0..n-1` (ver SklearnJoblibAdapter).
        """
        idx = int(np.argmax(probas))
        return Prediccion(
            etiqueta=str(self.adaptador.clases[idx]),
            confianza=float(probas[idx]),
            umbral=self.umbral_confianza,
        )

    def predecir(self, features: np.ndarray) -> Prediccion:
        if self.adaptador is None:
            raise RuntimeError("Llamá a cargar() antes de predecir()")
        return self._a_prediccion(self.adaptador.predict_proba(features))

    def predecir_secuencia(self, secuencia: np.ndarray) -> Prediccion:
        """Inferencia sobre una ventana de frames `(T, F)`, para los modelos
        dinámicos (BiLSTM de LSA64). Ver `SesionTraduccion` en
        traductor_service.py: la ventana la acumula la sesión, no el modelo."""
        if self.adaptador is None:
            raise RuntimeError("Llamá a cargar() antes de predecir()")
        return self._a_prediccion(self.adaptador.predict_proba_secuencia(secuencia))

    def top_n(self, features: np.ndarray, n: int = 3) -> list[Prediccion]:
        """Devuelve las n predicciones más probables, útil para debugging y para
        una futura UI que muestre alternativas (ej. abecedario con letras
        visualmente parecidas)."""
        if self.adaptador is None:
            raise RuntimeError("Llamá a cargar() antes de predecir()")

        probas = self.adaptador.predict_proba(features)
        top_idx = np.argsort(probas)[::-1][:n]
        return [
            Prediccion(
                etiqueta=str(self.adaptador.clases[int(i)]),
                confianza=float(probas[i]),
                umbral=self.umbral_confianza,
            )
            for i in top_idx
        ]
