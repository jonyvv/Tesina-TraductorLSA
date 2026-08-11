# -*- coding: utf-8 -*-
"""
Etapa 1 del pipeline: extraer los landmarks de LSA64 una sola vez.

    python ml/extract_features.py --dataset-dir .lsa64_cache/extracted/all

Deja un .npz con las secuencias de features. Despues, ml/train_lsa64.py lo lee
en segundos, asi se pueden probar hiperparametros sin repetir MediaPipe.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.features import FEATURE_VECTOR_LENGTH, FEATURE_VERSION
from lsa64.cache import FeatureCache, build_meta, save_cache
from lsa64.config import (
    DEFAULT_FRAME_STEP,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_SEQUENCE_LEN,
    default_cache_path,
)
from lsa64.data import ESPEJADO_CANONICO, build_samples, load_annotations, load_label_map
from lsa64.extract import default_workers, extract_sequences
from lsa64.source import resolve_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extraccion de features de LSA64 (etapa previa al entrenamiento)"
    )
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--dataset-archive", type=str, default=None)
    parser.add_argument("--download-url", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=str(Path.cwd() / ".lsa64_cache"))
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument(
        "--labels-map",
        type=str,
        default=None,
        help="JSON o CSV clase,nombre para etiquetas legibles (por defecto: clase_NN)",
    )
    parser.add_argument("--output", type=str, default=None, help="ruta del .npz de salida")
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--min-seq-len", type=int, default=DEFAULT_MIN_SEQUENCE_LEN)
    parser.add_argument("--keep-empty-frames", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"procesos en paralelo (por defecto {default_workers()} en esta maquina)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="procesar solo N videos, repartidos entre todas las clases (prueba rapida)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = resolve_dataset_root(
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        dataset_archive=Path(args.dataset_archive) if args.dataset_archive else None,
        download_url=args.download_url,
        work_dir=Path(args.work_dir),
    )
    output = Path(args.output) if args.output else default_cache_path(Path(args.work_dir))

    annotations = load_annotations(Path(args.annotations) if args.annotations else None)
    label_map = load_label_map(Path(args.labels_map) if args.labels_map else None)
    samples = build_samples(dataset_root, annotations, label_map)
    if not samples:
        raise SystemExit(f"No se encontraron videos en {dataset_root}")
    if args.limit and args.limit < len(samples):
        # Muestreo espaciado en vez de los primeros N: los archivos vienen
        # ordenados por clase, asi que samples[:N] daria una sola sena y el
        # smoke test no probaria nada del entrenamiento multiclase.
        paso = len(samples) / args.limit
        samples = [samples[int(i * paso)] for i in range(args.limit)]

    workers = args.workers or default_workers()
    clases = sorted({sample.label for sample in samples})
    sujetos = sorted({sample.subject for sample in samples if sample.subject})

    print(f"Dataset : {dataset_root}")
    print(f"Videos  : {len(samples)}")
    print(f"Clases  : {len(clases)}  -> {clases[:5]}{' ...' if len(clases) > 5 else ''}")
    print(f"Sujetos : {len(sujetos)} -> {sujetos}")
    print(f"Workers : {workers}")
    print(f"Espejado: {ESPEJADO_CANONICO} (contrato ESPEJADO_CANONICO de common/features.py)")
    if not sujetos:
        print("[!] Ningun video tiene sujeto identificable: el split no va a poder")
        print("    separar por persona y los resultados van a estar inflados.")
    print()

    start = time.perf_counter()
    sequences, descartes = extract_sequences(
        samples,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        min_seq_len=args.min_seq_len,
        keep_empty_frames=args.keep_empty_frames,
        workers=workers,
    )
    elapsed = time.perf_counter() - start

    if not sequences:
        raise SystemExit("No se pudo extraer ninguna secuencia valida.")

    valid = sorted(sequences.keys())
    cache = FeatureCache(
        sequences=[sequences[i] for i in valid],
        labels=[samples[i].label for i in valid],
        subjects=[samples[i].subject for i in valid],
        paths=[str(samples[i].path) for i in valid],
        splits=[samples[i].split for i in valid],
        meta=build_meta(
            feature_version=FEATURE_VERSION,
            feature_vector_length=FEATURE_VECTOR_LENGTH,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            keep_empty_frames=args.keep_empty_frames,
            espejado=ESPEJADO_CANONICO,
            min_seq_len=args.min_seq_len,
            dataset_dir=str(dataset_root),
            n_videos_encontrados=len(samples),
            n_descartados=len(descartes),
            extraccion_segundos=round(elapsed, 1),
        ),
    )
    save_cache(output, cache)

    print()
    print(f"Extraccion terminada en {elapsed / 60:.1f} min ({elapsed / max(len(samples), 1):.2f} s/video)")
    print(f"Cache    : {output}  ({output.stat().st_size / 1e6:.1f} MB)")
    print(f"Contenido: {cache.summary()}")
    if descartes:
        print(f"Descartados: {len(descartes)}")
        for path, motivo in descartes[:10]:
            print(f"  - {Path(path).name}: {motivo}")
        if len(descartes) > 10:
            print(f"  ... y {len(descartes) - 10} mas")
    print()
    print("Ahora entrena las veces que quieras sin repetir MediaPipe:")
    print(f"  python ml/train_lsa64.py --cache {output}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()  # necesario en Windows
    main()
