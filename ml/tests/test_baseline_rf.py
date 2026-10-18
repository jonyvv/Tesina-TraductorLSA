# -*- coding: utf-8 -*-
"""
Tests de la linea base Random Forest (ml/baseline_rf.py).

Lo que se prueba es la representacion: el resumen por secuencia es lo unico
propio de este baseline, porque los folds y el cache los reusa del LOSO de la
BiLSTM (ya cubiertos por test_lsa64_loso.py).

    python -m pytest ml/tests/test_baseline_rf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ml"))

from baseline_rf import construir_matriz, resumir_secuencia


def test_largo_del_vector_es_cuatro_veces_las_features():
    secuencia = np.random.rand(17, 138).astype(np.float32)
    assert resumir_secuencia(secuencia).shape == (552,)


def test_los_agregados_quedan_en_el_orden_documentado():
    """media, desvio, minimo, maximo — en ese orden, bloque por bloque."""
    secuencia = np.array([[0.0, 10.0], [2.0, 20.0]], dtype=np.float32)
    resumen = resumir_secuencia(secuencia)

    media, desvio, minimo, maximo = resumen[:2], resumen[2:4], resumen[4:6], resumen[6:]
    assert np.allclose(media, [1.0, 15.0])
    assert np.allclose(desvio, [1.0, 5.0])
    assert np.allclose(minimo, [0.0, 10.0])
    assert np.allclose(maximo, [2.0, 20.0])


def test_una_secuencia_de_un_frame_da_desvio_cero_y_no_nan():
    """min_seq_len podria bajarse a 1; un NaN se propagaria al bosque sin aviso."""
    resumen = resumir_secuencia(np.array([[3.0, 4.0]], dtype=np.float32))
    assert not np.isnan(resumen).any()
    assert np.allclose(resumen[2:4], [0.0, 0.0])


def test_el_resumen_descarta_el_orden_temporal():
    """Es la limitacion que la comparacion contra la BiLSTM busca medir: dos
    secuencias con los mismos frames en distinto orden son indistinguibles."""
    secuencia = np.random.rand(12, 138).astype(np.float32)
    invertida = secuencia[::-1].copy()
    assert np.allclose(resumir_secuencia(secuencia), resumir_secuencia(invertida))


def test_secuencias_de_distinto_largo_dan_vectores_del_mismo_largo():
    """Es lo que permite que un modelo tabular coma secuencias de largo variable."""
    matriz = construir_matriz([
        np.random.rand(8, 138).astype(np.float32),
        np.random.rand(45, 138).astype(np.float32),
        np.random.rand(120, 138).astype(np.float32),
    ])
    assert matriz.shape == (3, 552)


def test_rechaza_secuencia_vacia():
    with pytest.raises(ValueError, match="vacia"):
        resumir_secuencia(np.zeros((0, 138), dtype=np.float32))


def test_rechaza_secuencia_que_no_es_2d():
    with pytest.raises(ValueError, match="2D"):
        resumir_secuencia(np.zeros(138, dtype=np.float32))
