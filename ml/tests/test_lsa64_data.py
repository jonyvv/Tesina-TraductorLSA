# -*- coding: utf-8 -*-
"""
Tests del pipeline de datos de LSA64.

Se puede correr con pytest o directamente:
    python ml/tests/test_lsa64_data.py

No requiere MediaPipe, torch ni los videos reales: arma un dataset falso con
la misma convencion de nombres (CCC_SSS_RRR.mp4) y archivos vacios.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ml"))

import numpy as np

from lsa64.cache import FeatureCache, build_meta, incompatibilities, load_cache, save_cache
from lsa64.data import (
    build_samples,
    infer_label_from_path,
    infer_subject_from_stem,
    load_label_map,
    parse_lsa64_stem,
    train_val_test_split,
)

N_CLASES, N_SUJETOS, N_REPS = 64, 10, 5


def _fake_dataset(root: Path, n_clases=N_CLASES, n_sujetos=N_SUJETOS, n_reps=N_REPS) -> Path:
    videos = root / "all"
    videos.mkdir(parents=True, exist_ok=True)
    for clase in range(1, n_clases + 1):
        for sujeto in range(1, n_sujetos + 1):
            for rep in range(1, n_reps + 1):
                (videos / f"{clase:03d}_{sujeto:03d}_{rep:03d}.mp4").touch()
    return videos


def test_parse_lsa64_stem():
    assert parse_lsa64_stem("001_001_001") == {
        "clase": "001", "sujeto": "001", "repeticion": "001",
    }
    assert parse_lsa64_stem("064_010_005")["clase"] == "064"
    # Nombres que no son de LSA64 caen al parseo generico.
    assert parse_lsa64_stem("hola_subj03") is None
    assert parse_lsa64_stem("01_01_01") is None


def test_label_es_la_clase_no_el_sujeto():
    """Regresion: el parseo generico tomaba tokens[1:] y armaba '001 001',
    es decir sujeto+repeticion, descartando la clase real de la sena."""
    with tempfile.TemporaryDirectory() as tmp:
        videos = _fake_dataset(Path(tmp), n_clases=2, n_sujetos=2, n_reps=1)
        label = infer_label_from_path(videos / "001_002_001.mp4", videos)
        assert label == "clase_01", label
        assert infer_label_from_path(videos / "064_010_005.mp4", videos) == "clase_64"


def test_subject_se_infiere():
    """Regresion: devolvia None, y entonces el split agrupaba por archivo."""
    assert infer_subject_from_stem(Path("001_007_003.mp4")) == "sujeto_07"
    assert infer_subject_from_stem(Path("064_010_005.mp4")) == "sujeto_10"
    # El parseo generico previo sigue funcionando para capturas propias.
    assert infer_subject_from_stem(Path("hola_subj03_1.mp4")) == "03"


def test_build_samples_cuenta_clases_y_sujetos():
    with tempfile.TemporaryDirectory() as tmp:
        videos = _fake_dataset(Path(tmp))
        samples = build_samples(videos, annotations={})

        assert len(samples) == N_CLASES * N_SUJETOS * N_REPS == 3200
        assert len({s.label for s in samples}) == N_CLASES, "deben ser 64 clases, no 640"
        assert len({s.subject for s in samples}) == N_SUJETOS
        # Cada clase tiene exactamente sujetos x repeticiones muestras.
        por_clase = {}
        for sample in samples:
            por_clase[sample.label] = por_clase.get(sample.label, 0) + 1
        assert set(por_clase.values()) == {N_SUJETOS * N_REPS}


def test_split_no_comparte_sujetos():
    with tempfile.TemporaryDirectory() as tmp:
        videos = _fake_dataset(Path(tmp))
        samples = build_samples(videos, annotations={})
        train_idx, val_idx, test_idx = train_val_test_split(samples, seed=42)

        def sujetos(indices):
            return {samples[i].subject for i in indices}

        train_s, val_s, test_s = sujetos(train_idx), sujetos(val_idx), sujetos(test_idx)
        assert train_s and test_s
        assert not (train_s & test_s), f"fuga train/test: {train_s & test_s}"
        assert not (train_s & val_s), f"fuga train/val: {train_s & val_s}"
        assert not (val_s & test_s), f"fuga val/test: {val_s & test_s}"
        assert len(train_s | val_s | test_s) == N_SUJETOS
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(samples)


def test_todas_las_clases_estan_en_train_y_test():
    """Con split por sujeto, cada clase debe seguir apareciendo en ambos lados:
    si no, hay clases imposibles de predecir en test."""
    with tempfile.TemporaryDirectory() as tmp:
        videos = _fake_dataset(Path(tmp))
        samples = build_samples(videos, annotations={})
        train_idx, _, test_idx = train_val_test_split(samples, seed=42)
        assert {samples[i].label for i in train_idx} == {samples[i].label for i in test_idx}


def test_labels_map():
    with tempfile.TemporaryDirectory() as tmp:
        mapa = Path(tmp) / "labels.json"
        mapa.write_text(json.dumps({"1": "Opaco", "002": "Rojo"}), encoding="utf-8")
        label_map = load_label_map(mapa)
        assert label_map == {"001": "Opaco", "002": "Rojo"}

        videos = _fake_dataset(Path(tmp), n_clases=3, n_sujetos=2, n_reps=1)
        samples = build_samples(videos, annotations={}, label_map=label_map)
        etiquetas = {s.label for s in samples}
        assert etiquetas == {"Opaco", "Rojo", "clase_03"}


def test_cache_roundtrip():
    rng = np.random.default_rng(0)
    sequences = [rng.random((n, 138), dtype=np.float32) for n in (12, 45, 8, 120)]
    cache = FeatureCache(
        sequences=sequences,
        labels=["clase_01", "clase_02", "clase_01", "clase_03"],
        subjects=["sujeto_01", "sujeto_02", "sujeto_01", None],
        paths=[f"v{i}.mp4" for i in range(4)],
        splits=[None, None, "test", None],
        meta=build_meta(
            feature_version="v1",
            feature_vector_length=138,
            frame_step=2,
            max_frames=120,
            keep_empty_frames=False,
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "features.npz"
        save_cache(path, cache)
        cargado = load_cache(path)

        assert len(cargado) == 4
        assert [len(s) for s in cargado.sequences] == [12, 45, 8, 120]
        for original, recuperada in zip(sequences, cargado.sequences):
            assert np.array_equal(original, recuperada)
        assert cargado.labels == cache.labels
        assert cargado.subjects == ["sujeto_01", "sujeto_02", "sujeto_01", None]
        assert cargado.splits == [None, None, "test", None]
        assert cargado.meta["frame_step"] == 2


def test_cache_se_invalida_si_cambian_los_parametros():
    base = build_meta(
        feature_version="v1", feature_vector_length=138,
        frame_step=2, max_frames=120, keep_empty_frames=False,
    )
    igual = build_meta(
        feature_version="v1", feature_vector_length=138,
        frame_step=2, max_frames=120, keep_empty_frames=False,
        dataset_dir="/otra/ruta",  # metadata informativa: no invalida
    )
    distinto = build_meta(
        feature_version="v1", feature_vector_length=138,
        frame_step=1, max_frames=120, keep_empty_frames=False,
    )
    assert incompatibilities(base, igual) == []
    assert incompatibilities(base, distinto)


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
