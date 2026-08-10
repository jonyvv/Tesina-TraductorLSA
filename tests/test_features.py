# -*- coding: utf-8 -*-
"""
Tests del módulo compartido de features (common/features.py).

Verifican las propiedades que el esquema promete y de las que depende todo el
resto del pipeline: longitud fija, invarianza a traslación y escala, y slots
estables por mano.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.features import (  # noqa: E402
    FEATURES_PER_HAND,
    FEATURE_VECTOR_LENGTH,
    HAND_CONNECTIONS,
    NUM_LANDMARKS,
    build_feature_vector,
    calculate_angles,
    landmarks_para_overlay,
    normalize_landmarks,
)


def _mano_sintetica(offset=(0.0, 0.0, 0.0), escala=1.0):
    """21 landmarks con forma de mano, trasladados y escalados a gusto."""
    rng = np.random.default_rng(7)
    base = rng.random((21, 3)).astype(np.float32)
    base[0] = [0.5, 0.5, 0.0]   # muñeca
    base[9] = [0.5, 0.3, 0.0]   # base del dedo medio (referencia de escala)
    return base * escala + np.array(offset, dtype=np.float32)


def _a_landmarks(arr):
    return [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in arr]


def _resultado_mp(manos):
    """Imita un HandLandmarkerResult: `manos` es [(landmarks, "Left"/"Right"), ...]"""
    return SimpleNamespace(
        hand_landmarks=[_a_landmarks(lm) for lm, _ in manos],
        handedness=[[SimpleNamespace(category_name=lado, score=0.99)] for _, lado in manos],
    )


class TestNormalizacion:
    def test_invariante_a_traslacion(self):
        a = normalize_landmarks(_mano_sintetica())
        b = normalize_landmarks(_mano_sintetica(offset=(0.3, -0.2, 0.1)))
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_invariante_a_escala(self):
        # La invarianza a escala no es EXACTA a propósito: `normalize_landmarks`
        # divide por `norm(centered[9]) + 1e-6`, y ese epsilon (que evita la
        # división por cero cuando la mano se detecta degenerada) pesa distinto
        # según el tamaño de la mano. El desvío resultante es del orden de 1e-6
        # relativo, muy por debajo del ruido de detección de MediaPipe, así que
        # se compara con tolerancia relativa y no con igualdad estricta.
        a = normalize_landmarks(_mano_sintetica())
        b = normalize_landmarks(_mano_sintetica(escala=2.5))
        np.testing.assert_allclose(a, b, rtol=1e-4)

    def test_muñeca_queda_en_el_origen(self):
        norm = normalize_landmarks(_mano_sintetica(offset=(0.9, 0.4, 0.2)))
        np.testing.assert_allclose(norm[0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_angulos_en_rango_valido(self):
        angulos = calculate_angles(normalize_landmarks(_mano_sintetica()))
        assert len(angulos) == 5
        assert all(0.0 <= a <= np.pi for a in angulos)


class TestVectorDeFeatures:
    def test_longitud_fija_sin_manos(self):
        res = build_feature_vector(_resultado_mp([]))
        assert res.vector.shape == (FEATURE_VECTOR_LENGTH,)
        assert not res.any_hand_present()
        assert np.all(res.vector == 0.0)

    def test_longitud_fija_con_una_y_dos_manos(self):
        mano = _mano_sintetica()
        una = build_feature_vector(_resultado_mp([(mano, "Right")]))
        dos = build_feature_vector(_resultado_mp([(mano, "Left"), (mano, "Right")]))
        assert una.vector.shape == (FEATURE_VECTOR_LENGTH,)
        assert dos.vector.shape == (FEATURE_VECTOR_LENGTH,)

    def test_slots_estables_por_handedness(self):
        """La mano derecha SIEMPRE va al segundo bloque, sin importar el orden
        en que MediaPipe la haya detectado. Es lo que permite señas bimanuales."""
        izq, der = _mano_sintetica(), _mano_sintetica(offset=(0.2, 0.0, 0.0))

        orden_a = build_feature_vector(_resultado_mp([(izq, "Left"), (der, "Right")]))
        orden_b = build_feature_vector(_resultado_mp([(der, "Right"), (izq, "Left")]))

        np.testing.assert_allclose(orden_a.vector, orden_b.vector, atol=1e-6)

    def test_flags_de_presencia(self):
        mano = _mano_sintetica()

        solo_der = build_feature_vector(_resultado_mp([(mano, "Right")]))
        assert solo_der.right_present and not solo_der.left_present
        assert solo_der.vector[0] == 0.0                    # flag mano izquierda
        assert solo_der.vector[FEATURES_PER_HAND] == 1.0    # flag mano derecha

        solo_izq = build_feature_vector(_resultado_mp([(mano, "Left")]))
        assert solo_izq.left_present and not solo_izq.right_present
        assert solo_izq.vector[0] == 1.0
        assert solo_izq.vector[FEATURES_PER_HAND] == 0.0

    def test_mano_ausente_se_distingue_de_mano_en_el_origen(self):
        """El flag de presencia existe justamente para que "no hay mano" no se
        confunda con "hay una mano cuyos landmarks dieron cerca de cero"."""
        sin_manos = build_feature_vector(_resultado_mp([]))
        con_mano = build_feature_vector(_resultado_mp([(_mano_sintetica(), "Left")]))
        assert sin_manos.vector[0] != con_mano.vector[0]


class TestLandmarksParaOverlay:
    """Serialización de landmarks que se manda al cliente para dibujar."""

    def test_sin_manos_devuelve_lista_vacia(self):
        assert landmarks_para_overlay(_resultado_mp([])) == []

    def test_estructura_por_mano(self):
        salida = landmarks_para_overlay(_resultado_mp([(_mano_sintetica(), "Right")]))
        assert len(salida) == 1
        assert salida[0]["mano"] == "Right"
        assert len(salida[0]["puntos"]) == NUM_LANDMARKS
        assert all(len(p) == 2 for p in salida[0]["puntos"]), "solo x,y: z no aporta al overlay 2D"

    def test_dos_manos(self):
        mano = _mano_sintetica()
        salida = landmarks_para_overlay(_resultado_mp([(mano, "Left"), (mano, "Right")]))
        assert [m["mano"] for m in salida] == ["Left", "Right"]

    def test_redondeo_acota_el_tamaño_del_payload(self):
        salida = landmarks_para_overlay(_resultado_mp([(_mano_sintetica(), "Left")]), decimales=4)
        for x, y in salida[0]["puntos"]:
            assert x == round(x, 4) and y == round(y, 4)

    def test_indices_de_conexiones_son_validos(self):
        """El frontend usa estos pares para indexar los puntos: si alguno se
        fuera de rango, el esqueleto se dibujaría mal o rompería."""
        for a, b in HAND_CONNECTIONS:
            assert 0 <= a < NUM_LANDMARKS
            assert 0 <= b < NUM_LANDMARKS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
