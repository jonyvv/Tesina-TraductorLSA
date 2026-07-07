# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lsa64.config import LSA64TrainingConfig, MODELS_DIR
from lsa64.source import resolve_dataset_root
from lsa64.training import train_lsa64_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento LSA64 con BiLSTM")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--dataset-archive", type=str, default=None)
    parser.add_argument("--download-url", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=str(Path.cwd() / ".lsa64_cache"))
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(MODELS_DIR / "modelo_lsa64_lstm.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--min-seq-len", type=int, default=8)
    parser.add_argument("--keep-empty-frames", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        dataset_archive=Path(args.dataset_archive) if args.dataset_archive else None,
        download_url=args.download_url,
        work_dir=Path(args.work_dir),
    )
    config = LSA64TrainingConfig(
        dataset_dir=dataset_root,
        annotations=Path(args.annotations) if args.annotations else None,
        output=Path(args.output),
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        min_seq_len=args.min_seq_len,
        keep_empty_frames=args.keep_empty_frames,
        seed=args.seed,
    )
    train_lsa64_model(config)


if __name__ == "__main__":
    main()
