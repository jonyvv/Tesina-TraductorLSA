# -*- coding: utf-8 -*-
"""
Tests de backend/app/modelo_lse.py.

El grueso de estos tests son REGRESIONES de un bug real que se encontró en el
proyecto: un modelo entrenado con un subconjunto de las clases del dataset
(porque el split agrupado por sesión dejó clases enteras fuera del train) se
guardaba y se servía sin ningún error, prediciendo siempre lo mismo con
confianza 1.0 — y encima con la etiqueta equivocada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.app.modelo_lse import ModeloLSE  # noqa: E402
from backend.app.modelos.sklearn_adapter import SklearnJoblibAdapter  # noqa: E402
from common.features import FEATURE_VECTOR_LENGTH, FEATURE_VERSION  # noqa: E402

CLASES = ["a", "b", "c"]


def _datos(n_por_clase=12, clases=CLASES, semilla=0):
    rng = np.random.default_rng(semilla)
    X, y = [], []
    for i, clase in enumerate(clases):
        centro = np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32)
        centro[i * 10:(i + 1) * 10] = 1.0  # cada clase vive en otra zona
        for _ in range(n_por_clase):
            X.append(centro + rng.normal(0, 0.05, FEATURE_VECTOR_LENGTH).astype(np.float32))
            y.append(clase)
    return np.array(X, dtype=np.float32), np.array(y)


def _guardar(tmp_path, model, label_encoder, **extra):
    payload = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_version": FEATURE_VERSION,
        "feature_vector_length": FEATURE_VECTOR_LENGTH,
    }
    payload.update(extra)
    ruta = tmp_path / "modelo.joblib"
    joblib.dump(payload, ruta)
    return ruta


class TestCargar:
    def test_carga_modelo_sano(self, tmp_path):
        X, y_raw = _datos()
        le = LabelEncoder().fit(CLASES)
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, le.transform(y_raw))

        modelo = ModeloLSE(ruta_modelo=str(_guardar(tmp_path, rf, le)))
        modelo.cargar()
        assert modelo.clases == CLASES

    def test_rechaza_modelo_con_clases_faltantes(self, tmp_path):
        """REGRESIÓN: el modelo entrenado con una sola clase se cargaba sin
        chistar y predecía esa clase siempre, con confianza 1.0."""
        X, y_raw = _datos(clases=["b"])          # solo se entrenó con 'b'...
        le = LabelEncoder().fit(CLASES)          # ...pero el encoder conoce a, b, c
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, le.transform(y_raw))

        modelo = ModeloLSE(ruta_modelo=str(_guardar(tmp_path, rf, le)))
        with pytest.raises(RuntimeError, match="entrenado con solo"):
            modelo.cargar()

    def test_rechaza_feature_version_distinta(self, tmp_path):
        X, y_raw = _datos()
        le = LabelEncoder().fit(CLASES)
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, le.transform(y_raw))

        ruta = _guardar(tmp_path, rf, le, feature_version="version-vieja")
        modelo = ModeloLSE(ruta_modelo=str(ruta))
        with pytest.raises(RuntimeError, match="Desalineación de esquema"):
            modelo.cargar()

    def test_rechaza_longitud_de_vector_distinta(self, tmp_path):
        X, y_raw = _datos()
        le = LabelEncoder().fit(CLASES)
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, le.transform(y_raw))

        ruta = _guardar(tmp_path, rf, le, feature_vector_length=FEATURE_VECTOR_LENGTH + 1)
        modelo = ModeloLSE(ruta_modelo=str(ruta))
        with pytest.raises(RuntimeError, match="longitud"):
            modelo.cargar()

    def test_error_claro_si_no_existe_el_archivo(self, tmp_path):
        modelo = ModeloLSE(ruta_modelo=str(tmp_path / "no-existe.joblib"))
        with pytest.raises(FileNotFoundError):
            modelo.cargar()

    def test_desactiva_el_paralelismo_del_random_forest(self, tmp_path):
        """REGRESIÓN de rendimiento: el modelo se guarda con `n_jobs=-1`, pero
        en producción se predice de a UNA muestra por frame y ahí el overhead de
        repartir los árboles entre hilos supera al cómputo. Medido sobre el
        Random Forest de 200 árboles del proyecto: 55 ms contra 19,6 ms con
        `n_jobs=1` — más que todo el pipeline de MediaPipe junto."""
        X, y_raw = _datos()
        le = LabelEncoder().fit(CLASES)
        rf = RandomForestClassifier(n_estimators=10, n_jobs=-1,
                                    random_state=0).fit(X, le.transform(y_raw))

        modelo = ModeloLSE(ruta_modelo=str(_guardar(tmp_path, rf, le)))
        modelo.cargar()
        assert modelo.model.n_jobs == 1

    def test_rechaza_mlp_de_una_sola_clase(self, tmp_path):
        """REGRESIÓN: MLPClassifier entrenado con una sola clase devuelve un
        `predict_proba` de 2 columnas pero un `classes_` de 1 elemento. Traducir
        el argmax a etiqueta indexaba fuera de rango (IndexError en pleno
        WebSocket) o devolvía cualquier cosa."""
        from sklearn.neural_network import MLPClassifier

        X, y_raw = _datos(clases=["b"])
        le = LabelEncoder().fit(["b"])
        mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=120,
                            random_state=0).fit(X, le.transform(y_raw))
        assert mlp.predict_proba(X[:1]).shape[1] == 2, "el fixture debe reproducir el caso"
        assert len(mlp.classes_) == 1

        modelo = ModeloLSE(ruta_modelo=str(_guardar(tmp_path, mlp, le)))
        with pytest.raises(RuntimeError, match="probabilidades pero declara"):
            modelo.cargar()
        assert modelo.model is None

    @pytest.mark.parametrize(
        "extra, clases_entrenadas",
        [
            ({"feature_version": "version-vieja"}, CLASES),
            ({"feature_vector_length": FEATURE_VECTOR_LENGTH + 1}, CLASES),
            ({}, ["b"]),
        ],
        ids=["feature_version", "longitud_vector", "clases_faltantes"],
    )
    def test_un_modelo_rechazado_no_queda_cargado(self, tmp_path, extra, clases_entrenadas):
        """REGRESIÓN: `cargar()` asignaba `self.model` ANTES de validar, así que
        un modelo rechazado quedaba igualmente disponible: /health informaba
        `modelo_cargado: true` y el backend lo servía. Los chequeos no protegían
        nada. Ahora el estado se compromete recién después de validar todo."""
        X, y_raw = _datos(clases=clases_entrenadas)
        le = LabelEncoder().fit(CLASES)
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, le.transform(y_raw))

        modelo = ModeloLSE(ruta_modelo=str(_guardar(tmp_path, rf, le, **extra)))
        with pytest.raises(RuntimeError):
            modelo.cargar()

        assert modelo.model is None, "un modelo rechazado no puede quedar cargado"
        assert modelo.label_encoder is None
        assert modelo.clases == []


class TestPredecir:
    """Se construye el adaptador a mano (`desde_objetos`) en vez de pasar por
    `cargar()`: varios de estos casos son modelos deliberadamente rotos que la
    validación de `cargar()` rechazaría antes de poder probar el mapeo."""

    def _modelo_entrenado(self):
        X, y_raw = _datos()
        le = LabelEncoder().fit(CLASES)
        rf = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, le.transform(y_raw))
        modelo = ModeloLSE(ruta_modelo="(en memoria)", umbral_confianza=0.5)
        modelo.adaptador = SklearnJoblibAdapter.desde_objetos(rf, le)
        return modelo

    def test_predice_la_clase_correcta(self):
        modelo = self._modelo_entrenado()
        for i, esperada in enumerate(CLASES):
            v = np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32)
            v[i * 10:(i + 1) * 10] = 1.0
            assert modelo.predecir(v).etiqueta == esperada

    def test_etiqueta_correcta_con_classes_subconjunto(self):
        """REGRESIÓN del bug de mapeo de índices.

        Las columnas de `predict_proba` corresponden a `model.classes_`, no a
        `0..n-1`. Cuando el modelo se entrenó con un subconjunto, el código viejo
        hacía `label_encoder.inverse_transform([argmax])` y devolvía la etiqueta
        de OTRA clase, sin ningún error. Acá el modelo solo conoce la clase 'b'
        (índice 1), así que un argmax de 0 tiene que seguir dando 'b'.
        """
        X, y_raw = _datos(clases=["b"])
        le = LabelEncoder().fit(CLASES)
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, le.transform(y_raw))

        modelo = ModeloLSE(ruta_modelo="(en memoria)")
        modelo.adaptador = SklearnJoblibAdapter.desde_objetos(rf, le)

        assert list(rf.classes_) == [1], "el fixture debe reproducir el caso del bug"
        pred = modelo.predecir(np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32))
        assert pred.etiqueta == "b"   # con el bug daba "a"

    def test_top_n_ordenado_y_consistente(self):
        modelo = self._modelo_entrenado()
        v = np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32)
        v[0:10] = 1.0

        top = modelo.top_n(v, n=3)
        assert len(top) == 3
        assert top[0].etiqueta == "a"
        assert [p.confianza for p in top] == sorted((p.confianza for p in top), reverse=True)
        assert {p.etiqueta for p in top} == set(CLASES)

    def test_predecir_sin_cargar_falla(self):
        with pytest.raises(RuntimeError, match="cargar"):
            ModeloLSE(ruta_modelo="x").predecir(np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
