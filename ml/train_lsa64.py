# -*- coding: utf-8 -*-
"""
Etapa 2: entrenamiento de la BiLSTM sobre LSA64.

Flujo recomendado (la extraccion se paga una sola vez):

    python ml/extract_features.py --dataset-dir .lsa64_cache/extracted/all
    python ml/train_lsa64.py --epochs 60 --hidden-size 256

Si no hay cache, se puede pasar --dataset-dir y el script lo genera solo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lsa64.config import LSA64TrainingConfig, MODELS_DIR, default_cache_path
from lsa64.source import resolve_dataset_root
from lsa64.training import train_lsa64_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento LSA64 con BiLSTM")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--dataset-archive", type=str, default=None)
    parser.add_argument("--download-url", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=str(Path.cwd() / ".lsa64_cache"))
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--labels-map", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(MODELS_DIR / "modelo_lsa64_lstm.pt"))
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="ruta del .npz con features (por defecto .lsa64_cache/features_lsa64.npz)",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="ignorar el cache existente y volver a extraer landmarks",
    )
    parser.add_argument("--workers", type=int, default=None, help="procesos para extraer (si hace falta)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8, help="early stopping; 0 lo desactiva")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--min-seq-len", type=int, default=8)
    parser.add_argument("--keep-empty-frames", action="store_true")
    parser.add_argument(
        "--con-posicion",
        action="store_true",
        help="v2: agrega la posicion de la muneca al vector (138 -> 144). "
             "La ubicacion de la mano es un parametro fonologico que v1 descarta.",
    )
    # --- Regularizacion. Los defaults reproducen el baseline de 79,8 %.
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout antes de la capa final")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 desacoplado (AdamW)")
    parser.add_argument("--aug-noise", type=float, default=0.0,
                        help="sigma del ruido gaussiano sobre coordenadas, solo en train")
    parser.add_argument("--aug-frame-drop", type=float, default=0.0,
                        help="probabilidad de descartar cada frame, solo en train")
    parser.add_argument("--aug-time-scale", type=float, default=0.0,
                        help="reescalado temporal aleatorio +-s, solo en train")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    cache_path = Path(args.cache) if args.cache else default_cache_path(work_dir)

    # Si ya hay cache utilizable no hace falta el dataset de videos: el
    # entrenamiento solo necesita las features.
    necesita_dataset = args.refresh_cache or not cache_path.exists()
    if necesita_dataset:
        dataset_root = resolve_dataset_root(
            dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
            dataset_archive=Path(args.dataset_archive) if args.dataset_archive else None,
            download_url=args.download_url,
            work_dir=work_dir,
        )
    else:
        dataset_root = Path(args.dataset_dir) if args.dataset_dir else cache_path.parent

    config = LSA64TrainingConfig(
        dataset_dir=dataset_root,
        annotations=Path(args.annotations) if args.annotations else None,
        labels_map=Path(args.labels_map) if args.labels_map else None,
        output=Path(args.output),
        cache_path=cache_path,
        refresh_cache=args.refresh_cache,
        workers=args.workers,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        min_seq_len=args.min_seq_len,
        keep_empty_frames=args.keep_empty_frames,
        incluir_posicion=args.con_posicion,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        aug_noise=args.aug_noise,
        aug_frame_drop=args.aug_frame_drop,
        aug_time_scale=args.aug_time_scale,
        seed=args.seed,
    )
    train_lsa64_model(config)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()  # necesario en Windows si hay que extraer
    main()
