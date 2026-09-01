# -*- coding: utf-8 -*-
"""
Tests del mapa de nombres de senas (--labels-map).

La invariante que protegen: el nombre de la sena es PRESENTACION y no puede
entrar al entrenamiento. Si entrara, el LabelEncoder (que ordena alfabeticamente)
le asignaria otro indice a cada clase y el modelo saldria distinto segun el mapa
que se pase. Se midio: 0,8125 -> 0,8159 y best_epoch 11 -> 32 sobre datos
identicos.

    python ml/tests/test_lsa64_labels_map.py
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

from lsa64.cache import FeatureCache, build_meta, save_cache
from lsa64.config import LSA64TrainingConfig
from lsa64.training import load_features, nombres_de_clases


def _mapa_temporal(contenido: dict, carpeta: Path) -> Path:
    ruta = carpeta / "labels.json"
    with open(ruta, "w", encoding="utf-8") as handle:
        json.dump(contenido, handle)
    return ruta


def test_sin_mapa_devuelve_los_ids_canonicos():
    clases = ["clase_01", "clase_02", "clase_03"]
    assert nombres_de_clases(clases, None) == clases


def test_traduce_por_numero_de_clase():
    with tempfile.TemporaryDirectory() as tmp:
        ruta = _mapa_temporal({"001": "Opaque", "002": "Red"}, Path(tmp))
        assert nombres_de_clases(["clase_01", "clase_02"], ruta) == ["Opaque", "Red"]


def test_clase_sin_entrada_en_el_mapa_conserva_el_id():
    with tempfile.TemporaryDirectory() as tmp:
        ruta = _mapa_temporal({"001": "Opaque"}, Path(tmp))
        assert nombres_de_clases(["clase_01", "clase_07"], ruta) == ["Opaque", "clase_07"]


def test_es_idempotente_sobre_nombres_ya_traducidos():
    with tempfile.TemporaryDirectory() as tmp:
        ruta = _mapa_temporal({"001": "Opaque"}, Path(tmp))
        assert nombres_de_clases(["Opaque"], ruta) == ["Opaque"]


def test_conserva_el_orden_recibido():
    """No reordena: el orden lo fija el LabelEncoder sobre los ids canonicos, y
    los indices del modelo dependen de el."""
    with tempfile.TemporaryDirectory() as tmp:
        ruta = _mapa_temporal({"001": "Zeta", "002": "Alfa"}, Path(tmp))
        assert nombres_de_clases(["clase_01", "clase_02"], ruta) == ["Zeta", "Alfa"]


def test_el_cache_nunca_se_traduce():
    """LA invariante: --labels-map no toca las etiquetas con las que se entrena."""
    with tempfile.TemporaryDirectory() as tmp:
        carpeta = Path(tmp)
        secuencias = [np.ones((12, 138), dtype=np.float32) for _ in range(4)]
        cache = FeatureCache(
            sequences=secuencias,
            labels=["clase_01", "clase_02", "clase_01", "clase_02"],
            subjects=["sujeto_01", "sujeto_01", "sujeto_02", "sujeto_02"],
            paths=[f"00{i}.mp4" for i in range(4)],
            splits=[None] * 4,
            meta=build_meta(
                feature_version="v1",
                feature_vector_length=138,
                frame_step=2,
                max_frames=120,
                keep_empty_frames=True,
                espejado=True,
            ),
        )
        ruta_cache = carpeta / "features.npz"
        save_cache(ruta_cache, cache)

        config = LSA64TrainingConfig(
            dataset_dir=carpeta / "no-existe",
            annotations=None,
            output=carpeta / "modelo.pt",
            cache_path=ruta_cache,
            min_seq_len=8,
            labels_map=_mapa_temporal({"001": "Opaque", "002": "Red"}, carpeta),
        )
        cargado = load_features(config)
        assert set(cargado.labels) == {"clase_01", "clase_02"}, cargado.labels[:4]


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
