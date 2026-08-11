# -*- coding: utf-8 -*-
"""
Tests de la etapa de entrenamiento que no necesitan torch ni MediaPipe:
lectura del cache, deteccion de fuga de sujetos y metricas por clase.

    python ml/tests/test_lsa64_training.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ml"))

import numpy as np

from lsa64.cache import FeatureCache, build_meta, save_cache
from lsa64.config import LSA64TrainingConfig
from lsa64.data import VideoSample, train_val_test_split
from lsa64.training import _check_subject_leakage, _per_class_accuracy, load_features


def _cache_sintetico(n_clases=8, n_sujetos=6, n_reps=3, seed=0) -> FeatureCache:
    rng = np.random.default_rng(seed)
    sequences, labels, subjects, paths = [], [], [], []
    for clase in range(1, n_clases + 1):
        for sujeto in range(1, n_sujetos + 1):
            for rep in range(1, n_reps + 1):
                largo = int(rng.integers(10, 60))
                sequences.append(rng.random((largo, 138), dtype=np.float32))
                labels.append(f"clase_{clase:02d}")
                subjects.append(f"sujeto_{sujeto:02d}")
                paths.append(f"{clase:03d}_{sujeto:03d}_{rep:03d}.mp4")
    return FeatureCache(
        sequences=sequences,
        labels=labels,
        subjects=subjects,
        paths=paths,
        splits=[None] * len(sequences),
        meta=build_meta(
            feature_version="v1", feature_vector_length=138,
            frame_step=2, max_frames=120, keep_empty_frames=False,
        ),
    )


def test_load_features_usa_el_cache_sin_tocar_el_dataset():
    """Si el cache es compatible, no debe hacer falta que exista el dataset
    de videos: ese es justamente el punto de separar las dos etapas."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "features.npz"
        save_cache(cache_path, _cache_sintetico())

        config = LSA64TrainingConfig(
            dataset_dir=Path(tmp) / "no-existe",
            annotations=None,
            output=Path(tmp) / "modelo.pt",
            cache_path=cache_path,
        )
        cache = load_features(config)
        assert len(cache) == 8 * 6 * 3
        assert len(set(cache.labels)) == 8


def test_load_features_rechaza_cache_incompatible():
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "features.npz"
        save_cache(cache_path, _cache_sintetico())

        # frame_step distinto => las features no son las mismas => hay que
        # re-extraer, y como el dataset no existe debe fallar explicitamente.
        config = LSA64TrainingConfig(
            dataset_dir=Path(tmp) / "no-existe",
            annotations=None,
            output=Path(tmp) / "modelo.pt",
            cache_path=cache_path,
            frame_step=1,
        )
        try:
            load_features(config)
        except FileNotFoundError:
            return
        raise AssertionError("deberia haber intentado re-extraer y fallado")


def test_min_seq_len_se_aplica_al_cargar_sin_reextraer():
    """Cambiar min_seq_len no debe invalidar el cache (solo filtra), pero si
    debe aplicarse: antes se ignoraba en silencio al leer del cache."""
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(1)
        largos = [5, 20, 7, 40, 9]
        cache = FeatureCache(
            sequences=[rng.random((n, 138), dtype=np.float32) for n in largos],
            labels=[f"clase_{i:02d}" for i in range(5)],
            subjects=[f"sujeto_{i:02d}" for i in range(5)],
            paths=[f"v{i}.mp4" for i in range(5)],
            splits=[None] * 5,
            meta=build_meta(
                feature_version="v1", feature_vector_length=138,
                frame_step=2, max_frames=120, keep_empty_frames=False,
                min_seq_len=1,
            ),
        )
        cache_path = Path(tmp) / "features.npz"
        save_cache(cache_path, cache)

        config = LSA64TrainingConfig(
            dataset_dir=Path(tmp) / "no-existe",
            annotations=None,
            output=Path(tmp) / "modelo.pt",
            cache_path=cache_path,
            min_seq_len=10,
        )
        filtrado = load_features(config)
        assert [len(s) for s in filtrado.sequences] == [20, 40]
        assert filtrado.labels == ["clase_01", "clase_03"]


def test_deteccion_de_fuga_de_sujetos():
    metadata = [
        VideoSample(Path("a.mp4"), "clase_01", "sujeto_01"),
        VideoSample(Path("b.mp4"), "clase_01", "sujeto_02"),
        VideoSample(Path("c.mp4"), "clase_02", "sujeto_03"),
    ]
    assert _check_subject_leakage(metadata, [0], [1], [2]) == []

    fugado = _check_subject_leakage(metadata, [0, 1], [], [1, 2])
    assert fugado and "sujeto_02" in fugado[0]


def test_split_sobre_cache_sintetico_no_tiene_fuga():
    cache = _cache_sintetico()
    metadata = [
        VideoSample(Path(p), lbl, subj, split)
        for p, lbl, subj, split in zip(cache.paths, cache.labels, cache.subjects, cache.splits)
    ]
    train_idx, val_idx, test_idx = train_val_test_split(metadata, seed=42)
    assert _check_subject_leakage(metadata, train_idx, val_idx, test_idx) == []
    assert train_idx and test_idx


def test_per_class_accuracy():
    clases = np.array(["clase_01", "clase_02"])
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 1]
    assert _per_class_accuracy(y_true, y_pred, clases) == {"clase_01": 0.5, "clase_02": 1.0}


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fallos = 0
    for test in tests:
        try:
            test()
            print(f"  OK   {test.__name__}")
        except AssertionError as exc:
            fallos += 1
            print(f"  FALLO {test.__name__}: {exc}")
        except Exception as exc:
            fallos += 1
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - fallos}/{len(tests)} tests OK")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
