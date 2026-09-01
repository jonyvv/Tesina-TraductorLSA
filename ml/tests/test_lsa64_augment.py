# -*- coding: utf-8 -*-
"""
Tests de la augmentation de landmarks y de la regularizacion.

El test que mas importa es el primero: con los flags en su default, TODO tiene
que comportarse igual que antes. Si eso se rompe, el baseline de 79,8 % deja de
ser comparable con cualquier experimento nuevo y hay que remedir todo.

    python ml/tests/test_lsa64_augment.py
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
    FEATURE_VECTOR_LENGTH,
    NUM_LANDMARKS,
)
from lsa64.augment import (
    MIN_FRAMES_TRAS_AUGMENTAR,
    augmentar,
    aplicar_frame_drop,
    aplicar_noise,
    aplicar_time_scale,
    hay_augmentation,
)
from lsa64.config import LSA64TrainingConfig

_COORD_INI = 1
_COORD_FIN = 1 + NUM_LANDMARKS * COORDS_PER_LANDMARK
_ANG_INI, _ANG_FIN = _COORD_FIN, _COORD_FIN + ANGLES_PER_HAND


def _seq(n_frames: int = 20, vacios: tuple[int, ...] = ()) -> np.ndarray:
    """Secuencia sintetica: presencia en 1 salvo en los frames de `vacios`."""
    rng = np.random.RandomState(0)
    seq = rng.uniform(-1, 1, size=(n_frames, FEATURE_VECTOR_LENGTH)).astype(np.float32)
    seq[:, 0] = 1.0
    seq[:, FEATURES_PER_HAND] = 1.0
    for i in vacios:
        seq[i, :] = 0.0
    return seq


def _rng():
    return np.random.RandomState(123)


def test_defaults_del_config_reproducen_el_baseline():
    """dropout 0,2 es el que el modelo ya tenia fijo; el resto va apagado."""
    c = LSA64TrainingConfig(dataset_dir=Path("."), annotations=None, output=Path("x.pt"))
    assert c.dropout == 0.2, "el modelo tenia nn.Dropout(0.2) hardcodeado"
    assert c.weight_decay == 0.0
    assert c.aug_noise == 0.0 and c.aug_frame_drop == 0.0 and c.aug_time_scale == 0.0
    assert not hay_augmentation(c), "sin flags no se debe activar augmentation"


def test_sin_flags_devuelve_la_misma_secuencia():
    seq = _seq()
    fuera = augmentar(seq, _rng())
    assert fuera is seq, "sin augmentation no deberia ni copiar el array"


def test_nunca_toca_el_flag_de_presencia():
    seq = _seq(30, vacios=(3, 4, 11))
    fuera = augmentar(seq, _rng(), noise=0.05, frame_drop=0.2, time_scale=0.3)
    for col in (0, FEATURES_PER_HAND):
        assert set(np.unique(fuera[:, col])) <= {0.0, 1.0}, "la presencia debe seguir siendo 0 o 1"


def test_noise_no_ensucia_los_frames_vacios():
    seq = _seq(15, vacios=(2, 7))
    fuera = aplicar_noise(seq, 0.1, _rng())
    for i in (2, 7):
        assert np.array_equal(fuera[i], np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32)), (
            f"el frame vacio {i} se lleno de ruido: seria una mano fantasma"
        )


def test_noise_toca_las_coordenadas_y_deja_los_angulos():
    seq = _seq(10)
    fuera = aplicar_noise(seq, 0.05, _rng())
    assert not np.allclose(fuera[:, _COORD_INI:_COORD_FIN], seq[:, _COORD_INI:_COORD_FIN])
    assert np.array_equal(fuera[:, _ANG_INI:_ANG_FIN], seq[:, _ANG_INI:_ANG_FIN]), (
        "los angulos no se perturban (ver docstring de aplicar_noise)"
    )


def test_noise_cero_es_identidad():
    seq = _seq()
    assert aplicar_noise(seq, 0.0, _rng()) is seq


def test_frame_drop_conserva_el_orden():
    seq = _seq(60)
    fuera = aplicar_frame_drop(seq, 0.3, _rng())
    assert len(fuera) < len(seq), "con p=0,3 sobre 60 frames tiene que descartar algo"
    # cada frame de salida existe en la entrada, y en el mismo orden relativo
    pos = [next(j for j in range(len(seq)) if np.array_equal(seq[j], f)) for f in fuera]
    assert pos == sorted(pos), "el drop no puede reordenar la secuencia"


def test_frame_drop_no_degenera_secuencias_cortas():
    corta = _seq(MIN_FRAMES_TRAS_AUGMENTAR)
    assert aplicar_frame_drop(corta, 0.9, _rng()) is corta
    for _ in range(50):  # con p altisimo nunca debe bajar del minimo
        fuera = aplicar_frame_drop(_seq(12), 0.95, np.random.RandomState(None))
        assert len(fuera) >= MIN_FRAMES_TRAS_AUGMENTAR


def test_time_scale_cambia_el_largo_sin_inventar_frames():
    seq = _seq(40)
    fuera = aplicar_time_scale(seq, 0.4, _rng())
    assert len(fuera) != len(seq), "con s=0,4 el largo tiene que moverse"
    filas = {tuple(f) for f in seq}
    assert all(tuple(f) in filas for f in fuera), "no puede interpolar poses nuevas"


def test_time_scale_respeta_el_minimo():
    for _ in range(50):
        fuera = aplicar_time_scale(_seq(6), 0.9, np.random.RandomState(None))
        assert len(fuera) >= MIN_FRAMES_TRAS_AUGMENTAR


def test_es_determinista_con_la_misma_seed():
    seq = _seq(30)
    kw = dict(noise=0.03, frame_drop=0.15, time_scale=0.2)
    a = augmentar(seq, np.random.RandomState(7), **kw)
    b = augmentar(seq, np.random.RandomState(7), **kw)
    assert np.array_equal(a, b), "misma seed tiene que dar la misma augmentation"


def test_varia_entre_llamadas_del_mismo_rng():
    """Si no variara, cada epoch veria exactamente los mismos datos y no serviria."""
    seq = _seq(40)
    rng = np.random.RandomState(7)
    kw = dict(noise=0.03, frame_drop=0.15, time_scale=0.2)
    a = augmentar(seq, rng, **kw)
    b = augmentar(seq, rng, **kw)
    assert not (a.shape == b.shape and np.array_equal(a, b)), "cada llamada debe dar algo distinto"


def test_no_muta_la_secuencia_original():
    seq = _seq(25)
    copia = seq.copy()
    augmentar(seq, _rng(), noise=0.1, frame_drop=0.2, time_scale=0.3)
    assert np.array_equal(seq, copia), "la augmentation no puede pisar el cache en memoria"


def test_salida_es_float32_contigua():
    fuera = augmentar(_seq(20), _rng(), noise=0.02, frame_drop=0.1, time_scale=0.1)
    assert fuera.dtype == np.float32 and fuera.flags["C_CONTIGUOUS"], "torch.from_numpy lo necesita"


def test_augmentation_solo_en_train():
    """val y test nunca se augmentan: augmentarlos cambiaria lo que se mide."""
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print("       (salteado: torch no esta instalado en este entorno)")
        return
    from lsa64.fitting import _build_loaders

    seqs = [_seq(20) for _ in range(9)]
    labels = np.array([0, 1, 2] * 3)
    config = LSA64TrainingConfig(
        dataset_dir=Path("."), annotations=None, output=Path("x.pt"),
        batch_size=2, seed=42, aug_frame_drop=0.5, aug_noise=0.05,
    )
    train, val, test = _build_loaders(seqs, labels, ([0, 1, 2], [3, 4, 5], [6, 7, 8]), config)
    assert train.dataset.augmentar_datos is True, "train tiene que augmentarse"
    assert val.dataset.augmentar_datos is False, "val NO se augmenta"
    assert test.dataset.augmentar_datos is False, "test NO se augmenta"


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
