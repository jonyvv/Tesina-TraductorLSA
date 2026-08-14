# -*- coding: utf-8 -*-
"""
Cache de features de LSA64.

Motivacion
----------
Extraer los landmarks con MediaPipe es la parte cara del pipeline: ~190.000
inferencias sobre frames de 1080p para los 3200 videos del dataset. Entrenar
la BiLSTM sobre esos landmarks, en cambio, son segundos.

Antes, `train_lsa64_model` hacia las dos cosas juntas, asi que cambiar el
learning rate obligaba a repetir horas de extraccion. Este modulo separa las
dos etapas: la extraccion se corre una vez y se guarda en un .npz, y el
entrenamiento lo lee en segundos.

Formato del .npz
----------------
Las secuencias tienen largo variable, asi que se guardan concatenadas en un
unico array 2D mas un array de offsets (formato CSR):

    frames    float32 (total_frames, feature_vector_length)
    offsets   int64   (n_muestras + 1,)   -> secuencia i = frames[o[i]:o[i+1]]
    labels    <U      (n_muestras,)
    subjects  <U      (n_muestras,)
    paths     <U      (n_muestras,)
    splits    <U      (n_muestras,)       -> "" si no habia split explicito
    meta      json    parametros de extraccion, para validar compatibilidad
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE_FORMAT_VERSION = 1

# Parametros que cambian el contenido de las features. Si alguno difiere entre
# el cache y lo que se pide al entrenar, el cache no sirve y hay que re-extraer.
INVALIDATING_KEYS = (
    "cache_format_version",
    "feature_version",
    "feature_vector_length",
    "frame_step",
    "max_frames",
    # Orientacion de la imagen (contrato ESPEJADO_CANONICO de common/features.py).
    # Leandro documento que este contrato "no se puede validar automaticamente":
    # con esta clave si se puede, al menos para el cache. Un cache extraido con
    # otra orientacion tiene los slots de mano invertidos y no sirve.
    "espejado",
)

# keep_empty_frames NO invalida el cache: la extraccion guarda siempre la
# secuencia completa y el entrenamiento decide si descarta los frames sin mano.
# Se puede endurecer (tirar frames), nunca aflojar (inventarlos), asi que el
# unico caso incompatible se chequea aparte en puede_servir_frames_completos().


@dataclass
class FeatureCache:
    sequences: list[np.ndarray]
    labels: list[str]
    subjects: list[str | None]
    paths: list[str]
    splits: list[str | None]
    meta: dict

    def __len__(self) -> int:
        return len(self.sequences)

    def summary(self) -> str:
        total_frames = sum(len(seq) for seq in self.sequences)
        n_labels = len(set(self.labels))
        n_subjects = len({s for s in self.subjects if s})
        return (
            f"{len(self)} secuencias | {total_frames} frames | "
            f"{n_labels} clases | {n_subjects} sujetos"
        )


def build_meta(
    *,
    feature_version: str,
    feature_vector_length: int,
    frame_step: int,
    max_frames: int,
    keep_empty_frames: bool,
    espejado: bool = True,
    **extra,
) -> dict:
    meta = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "feature_version": feature_version,
        "feature_vector_length": int(feature_vector_length),
        "frame_step": int(frame_step),
        "max_frames": int(max_frames),
        "keep_empty_frames": bool(keep_empty_frames),
        "espejado": bool(espejado),
    }
    meta.update(extra)
    return meta


def incompatibilities(cache_meta: dict, expected_meta: dict) -> list[str]:
    """Lista de parametros en los que el cache difiere de lo pedido.
    Vacia => el cache se puede usar tal cual."""
    diffs = []
    for key in INVALIDATING_KEYS:
        got = cache_meta.get(key)
        want = expected_meta.get(key)
        if got != want:
            diffs.append(f"{key}: cache={got!r} pedido={want!r}")
    return diffs


def puede_servir_frames_completos(cache_meta: dict) -> bool:
    """True si el cache guarda la secuencia entera (incluidos los frames sin
    mano). Si se extrajo descartandolos, esos frames se perdieron y no hay forma
    de entrenar en modo completo sin volver a extraer."""
    return bool(cache_meta.get("keep_empty_frames", False))


def save_cache(path: Path, cache: FeatureCache) -> Path:
    if not cache.sequences:
        raise ValueError("No hay secuencias para guardar en el cache.")

    lengths = [len(seq) for seq in cache.sequences]
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    frames = np.concatenate(cache.sequences, axis=0).astype(np.float32, copy=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=frames,
        offsets=offsets,
        labels=np.array(cache.labels, dtype=object).astype("U"),
        subjects=np.array([s or "" for s in cache.subjects], dtype=object).astype("U"),
        paths=np.array(cache.paths, dtype=object).astype("U"),
        splits=np.array([s or "" for s in cache.splits], dtype=object).astype("U"),
        meta=np.array(json.dumps(cache.meta, ensure_ascii=False)),
    )
    return path


def load_cache(path: Path) -> FeatureCache:
    with np.load(path, allow_pickle=False) as data:
        frames = data["frames"]
        offsets = data["offsets"]
        sequences = [
            np.ascontiguousarray(frames[offsets[i]:offsets[i + 1]])
            for i in range(len(offsets) - 1)
        ]
        return FeatureCache(
            sequences=sequences,
            labels=[str(x) for x in data["labels"]],
            subjects=[str(x) or None for x in data["subjects"]],
            paths=[str(x) for x in data["paths"]],
            splits=[str(x) or None for x in data["splits"]],
            meta=json.loads(str(data["meta"])),
        )


def load_cache_if_compatible(path: Path, expected_meta: dict) -> tuple[FeatureCache | None, list[str]]:
    """Devuelve (cache, motivos). Si el cache no existe o es incompatible,
    devuelve (None, motivos) para que el llamador decida re-extraer."""
    if not path.exists():
        return None, [f"no existe {path}"]
    try:
        cache = load_cache(path)
    except Exception as exc:  # cache corrupto o de otra version de numpy
        return None, [f"no se pudo leer ({exc})"]

    diffs = incompatibilities(cache.meta, expected_meta)
    if diffs:
        return None, diffs
    return cache, []
