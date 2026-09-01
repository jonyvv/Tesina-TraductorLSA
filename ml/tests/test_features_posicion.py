# -*- coding: utf-8 -*-
"""
Tests de la variante v2 (posicion de la muneca en el vector de features).

El riesgo de este cambio es silencioso: si algun lugar sigue indexando con la
constante FEATURES_PER_HAND (69) sobre un vector de 144, lee la columna
equivocada y no falla — simplemente entrena con basura. Estos tests recorren
todos los puntos donde se indexa por bloque de mano.

    python ml/tests/test_features_posicion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ml"))

from common.features import (
    ANGLES_PER_HAND,
    COORDS_PER_LANDMARK,
    FEATURES_PER_HAND,
    FEATURES_PER_HAND_CON_POSICION,
    FEATURE_VECTOR_LENGTH,
    FEATURE_VECTOR_LENGTH_CON_POSICION,
    FEATURE_VERSION,
    FEATURE_VERSION_CON_POSICION,
    NUM_HANDS,
    NUM_LANDMARKS,
    POSITION_PER_HAND,
    PRESENCE_FLAG_LEN,
    build_feature_vector,
    coord_slices,
    feature_vector_length_de,
    feature_version_de,
    incluye_posicion,
    per_hand_width,
    presence_indices,
)
from lsa64.augment import augmentar, aplicar_noise

V1, V2 = FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_LENGTH_CON_POSICION


class _LM:
    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z


class _Cat:
    def __init__(self, nombre): self.category_name = nombre


class _Result:
    def __init__(self, manos): self.hand_landmarks, self.handedness = \
        [m[1] for m in manos], [[_Cat(m[0])] for m in manos]


def _landmarks(seed=0):
    rng = np.random.RandomState(seed)
    pts = rng.uniform(0.1, 0.9, size=(NUM_LANDMARKS, 3))
    return [_LM(*p) for p in pts], pts


def _seq(n=20, largo=V1, vacios=()):
    rng = np.random.RandomState(0)
    s = rng.uniform(-1, 1, size=(n, largo)).astype(np.float32)
    for i in presence_indices(largo):
        s[:, i] = 1.0
    for i in vacios:
        s[i, :] = 0.0
    return s


def test_largos_y_versiones():
    assert V1 == 138 and V2 == 144
    assert FEATURES_PER_HAND == 69 and FEATURES_PER_HAND_CON_POSICION == 72
    assert feature_vector_length_de(False) == V1 and feature_vector_length_de(True) == V2
    assert feature_version_de(False) == FEATURE_VERSION == "v1"
    assert feature_version_de(True) == FEATURE_VERSION_CON_POSICION == "v2"
    assert not incluye_posicion(V1) and incluye_posicion(V2)


def test_layout_por_ancho():
    assert per_hand_width(V1) == 69 and per_hand_width(V2) == 72
    assert presence_indices(V1) == [0, 69]
    assert presence_indices(V2) == [0, 72], "en v2 la 2da mano arranca en 72, no en 69"
    for largo in (V1, V2):
        for sl in coord_slices(largo):
            assert sl.stop - sl.start == NUM_LANDMARKS * COORDS_PER_LANDMARK
            assert sl.stop <= largo


def test_coords_no_pisan_presencia_posicion_ni_angulos():
    for largo in (V1, V2):
        pres = set(presence_indices(largo))
        cubierto = set()
        for sl in coord_slices(largo):
            cubierto |= set(range(sl.start, sl.stop))
        assert not (pres & cubierto), "el bloque de coords no puede incluir la presencia"
        ancho = per_hand_width(largo)
        angulos = {m * ancho + ancho - 1 - k for m in range(NUM_HANDS) for k in range(ANGLES_PER_HAND)}
        assert not (angulos & cubierto), "el bloque de coords no puede incluir los angulos"
        if incluye_posicion(largo):
            pos = {m * ancho + PRESENCE_FLAG_LEN + k for m in range(NUM_HANDS) for k in range(POSITION_PER_HAND)}
            assert not (pos & cubierto), "la posicion se excluye del ruido a proposito"


def test_v2_guarda_la_muneca_cruda():
    lms, pts = _landmarks()
    r = build_feature_vector(_Result([("Right", lms)]), incluir_posicion=True)
    v = r.vector
    assert len(v) == V2
    off = per_hand_width(V2) + PRESENCE_FLAG_LEN  # mano derecha, despues de la presencia
    assert np.allclose(v[off:off + 3], pts[0], atol=1e-5), \
        "v2 debe guardar la muneca SIN normalizar: es lo que normalize_landmarks descarta"


def test_v1_no_guarda_posicion():
    lms, pts = _landmarks()
    v = build_feature_vector(_Result([("Right", lms)]), incluir_posicion=False).vector
    assert len(v) == V1
    off = per_hand_width(V1) + PRESENCE_FLAG_LEN
    assert not np.allclose(v[off:off + 3], pts[0], atol=1e-3), "v1 no debe traer la muneca cruda"


def test_v1_es_v2_sin_el_bloque_de_posicion():
    """La unica diferencia entre versiones deben ser esos 3 numeros por mano."""
    lms, _ = _landmarks(3)
    res = _Result([("Right", lms), ("Left", _landmarks(4)[0])])
    v1 = build_feature_vector(res, incluir_posicion=False).vector
    v2 = build_feature_vector(res, incluir_posicion=True).vector
    sin_pos = np.concatenate([
        np.delete(v2[m * 72:(m + 1) * 72], slice(PRESENCE_FLAG_LEN, PRESENCE_FLAG_LEN + POSITION_PER_HAND))
        for m in range(NUM_HANDS)
    ])
    assert np.allclose(v1, sin_pos, atol=1e-6)


def test_slots_de_mano_se_respetan_en_v2():
    lms, pts = _landmarks(9)
    v = build_feature_vector(_Result([("Left", lms)]), incluir_posicion=True).vector
    assert v[0] == 1.0, "mano izquierda presente"
    assert v[72] == 0.0, "mano derecha ausente"
    assert np.allclose(v[1:4], pts[0], atol=1e-5)
    assert np.array_equal(v[72:], np.zeros(72, dtype=np.float32))


def test_frame_sin_manos_es_todo_ceros_en_v2():
    v = build_feature_vector(_Result([]), incluir_posicion=True).vector
    assert len(v) == V2 and not v.any()


def test_augmentation_respeta_el_layout_v2():
    s = _seq(30, V2, vacios=(3, 8))
    fuera = augmentar(s, np.random.RandomState(5), noise=0.05, frame_drop=0.2, time_scale=0.3)
    for col in presence_indices(V2):
        assert set(np.unique(fuera[:, col])) <= {0.0, 1.0}
    assert fuera.shape[1] == V2


def test_noise_en_v2_no_toca_la_posicion():
    s = _seq(12, V2)
    fuera = aplicar_noise(s, 0.1, np.random.RandomState(1))
    for m in range(NUM_HANDS):
        ini = m * 72 + PRESENCE_FLAG_LEN
        assert np.array_equal(fuera[:, ini:ini + POSITION_PER_HAND], s[:, ini:ini + POSITION_PER_HAND]), \
            "la posicion no se perturba: se mide si sirve, no si tolera ruido"
    for sl in coord_slices(V2):
        assert not np.allclose(fuera[:, sl], s[:, sl]), "las coordenadas si deben recibir ruido"


def test_noise_en_v2_no_ensucia_frames_vacios():
    s = _seq(15, V2, vacios=(2, 7))
    fuera = aplicar_noise(s, 0.1, np.random.RandomState(2))
    for i in (2, 7):
        assert not fuera[i].any(), f"el frame vacio {i} se lleno de ruido"


def test_mascara_con_mano_funciona_en_ambas_versiones():
    from lsa64.training import _frames_con_mano  # noqa: PLC0415

    for largo in (V1, V2):
        s = _seq(10, largo, vacios=(1, 4))
        m = _frames_con_mano(s)
        assert m.sum() == 8 and not m[1] and not m[4], f"mascara mal en largo {largo}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fallos = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except AssertionError as exc:
            fallos += 1
            print(f"  FALLO {t.__name__}: {exc}")
        except Exception as exc:
            fallos += 1
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - fallos}/{len(tests)} tests OK")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
