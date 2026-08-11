# -*- coding: utf-8 -*-
"""
Extraccion paralela de landmarks para LSA64.

MediaPipe HandLandmarker corre en CPU y es single-thread por detector, asi que
la forma de acelerar es repartir los videos entre procesos, cada uno con su
propio detector. En un Ryzen 7 (8 nucleos) esto baja la extraccion de las
~2 horas del pipeline secuencial a ~15-25 minutos.

Notas de implementacion
-----------------------
* Un detector por proceso, creado en el `initializer` del Pool. Crear el
  HandLandmarker cuesta ~1s, no se puede pagar por video.
* `cv2.setNumThreads(1)` en cada worker: OpenCV por defecto paraleliza el
  decode entre todos los nucleos, lo que compite con nuestros propios procesos
  y termina siendo mas lento (oversubscription).
* Las funciones son de nivel de modulo porque Windows usa `spawn`: el proceso
  hijo re-importa el modulo y necesita poder resolverlas por nombre.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import VideoSample, sequence_from_video

_detector = None
_options: dict = {}


@dataclass
class ExtractionResult:
    index: int
    sequence: np.ndarray | None
    reason: str | None = None


def _init_worker(options: dict) -> None:
    """Corre una vez por proceso hijo: crea el detector y limita los threads."""
    global _detector, _options

    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass

    # MediaPipe/TensorFlow Lite tambien intentan usar todos los nucleos.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from common.features import new_hands_detector

    _options = options
    _detector = new_hands_detector(max_num_hands=2)


def _extract_one(job: tuple[int, str]) -> ExtractionResult:
    index, video_path = job
    if _detector is None:
        raise RuntimeError("El worker no fue inicializado con _init_worker.")

    try:
        sequence = sequence_from_video(
            Path(video_path),
            detector=_detector,
            frame_step=_options["frame_step"],
            max_frames=_options["max_frames"],
            keep_empty_frames=_options["keep_empty_frames"],
        )
    except Exception as exc:  # un video corrupto no debe tirar toda la corrida
        return ExtractionResult(index=index, sequence=None, reason=f"error: {exc}")

    if sequence is None:
        return ExtractionResult(index=index, sequence=None, reason="sin frames con manos")
    if len(sequence) < _options["min_seq_len"]:
        return ExtractionResult(
            index=index,
            sequence=None,
            reason=f"secuencia corta ({len(sequence)} < {_options['min_seq_len']})",
        )
    return ExtractionResult(index=index, sequence=sequence.astype(np.float32, copy=False))


def default_workers() -> int:
    """Nucleos fisicos aproximados, dejando uno libre para no congelar la maquina."""
    logical = os.cpu_count() or 2
    return max(1, min(logical - 1, max(1, logical // 2)))


def extract_sequences(
    samples: list[VideoSample],
    *,
    frame_step: int,
    max_frames: int,
    min_seq_len: int,
    keep_empty_frames: bool,
    workers: int | None = None,
    progress_every: int = 25,
) -> tuple[dict[int, np.ndarray], list[tuple[str, str]]]:
    """Extrae la secuencia de features de cada video.

    Devuelve ({indice_en_samples: secuencia}, [(path, motivo_descarte)]).
    Con workers=1 corre en el proceso actual (util para tests y debug).
    """
    options = {
        "frame_step": frame_step,
        "max_frames": max_frames,
        "min_seq_len": min_seq_len,
        "keep_empty_frames": keep_empty_frames,
    }
    jobs = [(idx, str(sample.path)) for idx, sample in enumerate(samples)]
    sequences: dict[int, np.ndarray] = {}
    descartes: list[tuple[str, str]] = []
    workers = workers or default_workers()

    def _register(result: ExtractionResult) -> None:
        if result.sequence is not None:
            sequences[result.index] = result.sequence
        else:
            descartes.append((str(samples[result.index].path), result.reason or "desconocido"))

    if workers <= 1:
        _init_worker(options)
        try:
            for done, job in enumerate(jobs, start=1):
                _register(_extract_one(job))
                if progress_every and done % progress_every == 0:
                    print(f"  {done}/{len(jobs)} videos", flush=True)
        finally:
            _close_worker()
        return sequences, descartes

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(options,)) as pool:
        for done, result in enumerate(pool.imap_unordered(_extract_one, jobs, chunksize=4), start=1):
            _register(result)
            if progress_every and done % progress_every == 0:
                print(f"  {done}/{len(jobs)} videos", flush=True)

    return sequences, descartes


def _close_worker() -> None:
    global _detector
    if _detector is not None:
        _detector.close()
        _detector = None
