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
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import VideoSample, sequence_from_video

# Si en este tiempo ningun worker devuelve un resultado, algo se rompio: un
# video de ~2 s tarda ~2,5 s en procesarse, asi que 5 min es varias ordenes de
# magnitud mas de lo normal.
WORKER_TIMEOUT_S = 300

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
        # Siempre se guarda la secuencia COMPLETA: descartar frames es barato y
        # reversible al entrenar, pero recuperarlos cuesta re-extraer todo.
        sequence = sequence_from_video(
            Path(video_path),
            detector=_detector,
            frame_step=_options["frame_step"],
            max_frames=_options["max_frames"],
            keep_empty_frames=True,
            incluir_posicion=_options.get("incluir_posicion", False),
        )
    except Exception as exc:  # un video corrupto no debe tirar toda la corrida
        return ExtractionResult(index=index, sequence=None, reason=f"error: {exc}")

    if sequence is None or not len(sequence):
        return ExtractionResult(index=index, sequence=None, reason="video ilegible o vacio")

    # Los criterios de descarte se siguen midiendo sobre los frames CON mano,
    # para que sean comparables con las extracciones anteriores.
    # Indexar por el ancho REAL del vector, no por la constante: en v2 el bloque
    # de cada mano mide 72 y no 69.
    from common.features import presence_indices

    idx = presence_indices(sequence.shape[1])
    con_mano = int(np.logical_or.reduce([sequence[:, i] > 0 for i in idx]).sum())
    if con_mano == 0:
        return ExtractionResult(index=index, sequence=None, reason="sin frames con manos")
    if con_mano < _options["min_seq_len"]:
        return ExtractionResult(
            index=index,
            sequence=None,
            reason=f"secuencia corta ({con_mano} < {_options['min_seq_len']})",
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
    incluir_posicion: bool = False,
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
        "incluir_posicion": incluir_posicion,
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

    total = len(jobs)
    inicio = time.perf_counter()

    def _progreso(done: int) -> None:
        # Al principio se informa seguido para que se vea que arranco; despues
        # cada progress_every para no inundar la consola.
        cada = 5 if done <= 25 else progress_every
        if not cada or done % cada:
            return
        transcurrido = time.perf_counter() - inicio
        restante = transcurrido / done * (total - done)
        print(
            f"  {done}/{total} videos"
            f" | {transcurrido / 60:.1f} min transcurridos"
            f" | faltan ~{restante / 60:.0f} min",
            flush=True,
        )

    if workers <= 1:
        print("Procesando en un solo proceso (sin paralelismo)...", flush=True)
        _init_worker(options)
        try:
            for done, job in enumerate(jobs, start=1):
                _register(_extract_one(job))
                _progreso(done)
        finally:
            _close_worker()
        return sequences, descartes

    import multiprocessing as mp

    # Este aviso importa: levantar los procesos implica importar MediaPipe en
    # cada uno (spawn en Windows), lo que puede tardar ~30-60 s en silencio.
    # Sin el mensaje, parece que el script se colgo y uno lo mata con Ctrl+C.
    print(
        f"Levantando {workers} procesos (cada uno importa MediaPipe, "
        f"puede tardar ~1 min antes del primer avance)...",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=workers, initializer=_init_worker, initargs=(options,))
    try:
        # chunksize=1 a proposito: con chunksize>1, imap_unordered envuelve el
        # resultado en un generador que no expone .next(timeout=...), y ademas
        # reparte peor la carga. Cada tarea tarda ~2,5 s, asi que el overhead
        # de IPC por tarea es despreciable.
        iterador = pool.imap_unordered(_extract_one, jobs, chunksize=1)
        for done in range(1, total + 1):
            try:
                # Con timeout en vez de espera infinita: si un worker muere
                # (MediaPipe puede caerse con algun video), imap_unordered se
                # quedaria colgado para siempre sin decir nada.
                result = iterador.next(timeout=WORKER_TIMEOUT_S)
            except mp.TimeoutError:
                print(
                    f"\n[!] Ningun worker respondio en {WORKER_TIMEOUT_S} s "
                    f"(iban {done - 1}/{total}).",
                    flush=True,
                )
                print("[!] Reintenta con --workers 1 para ver que video lo traba.", flush=True)
                raise
            _register(result)
            _progreso(done)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("\n[!] Interrumpido. No se guardo nada; hay que correrlo de nuevo.", flush=True)
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    return sequences, descartes


def _close_worker() -> None:
    global _detector
    if _detector is not None:
        _detector.close()
        _detector = None
