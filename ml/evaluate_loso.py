# -*- coding: utf-8 -*-
"""
Validacion leave-one-subject-out sobre el cache de features de LSA64.

    python ml/evaluate_loso.py

Entrena 10 modelos (uno por sujeto dejado afuera) y reporta media +- desvio.
Es el resultado que se puede defender: un solo split con 2 sujetos de test da
un numero que varia varios puntos segun a quien le toque.

No re-extrae nada: trabaja sobre el .npz que genero ml/extract_features.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lsa64.config import LSA64TrainingConfig, REPORTS_DIR, default_cache_path
from lsa64.evaluation import evaluar_loso, imprimir_resumen
from lsa64.training import load_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-subject-out sobre LSA64")
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=str(Path.cwd() / ".lsa64_cache"))
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPORTS_DIR / "loso_lsa64.json"),
        help="JSON con el reporte completo (folds, accuracy por clase, predicciones)",
    )
    parser.add_argument(
        "--val-subjects",
        type=int,
        default=1,
        help="sujetos de validacion por fold, tomados de los de entrenamiento. 0 = sin early stopping",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-seq-len", type=int, default=8)
    parser.add_argument(
        "--keep-empty-frames",
        action="store_true",
        help="conservar los frames sin mano detectada, preservando la grilla temporal",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose-epochs", action="store_true", help="imprime cada epoch de cada fold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache) if args.cache else default_cache_path(Path(args.work_dir))
    if not cache_path.exists():
        raise SystemExit(
            f"No existe el cache {cache_path}.\n"
            f"Genéralo primero con: python ml/extract_features.py --dataset-dir <ruta>"
        )

    config = LSA64TrainingConfig(
        dataset_dir=cache_path.parent,
        annotations=None,
        output=Path(args.output),
        cache_path=cache_path,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        min_seq_len=args.min_seq_len,
        keep_empty_frames=args.keep_empty_frames,
        seed=args.seed,
    )

    cache = load_features(config)
    print()

    inicio = time.perf_counter()
    reporte = evaluar_loso(
        cache,
        config,
        val_subjects=args.val_subjects,
        verbose_epochs=args.verbose_epochs,
    )
    reporte["duracion_segundos"] = round(time.perf_counter() - inicio, 1)
    reporte["cache"] = str(cache_path)
    reporte["cache_meta"] = cache.meta

    imprimir_resumen(reporte)

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8") as handle:
        json.dump(reporte, handle, indent=2, ensure_ascii=False)

    print()
    print(f"[OK] Reporte guardado en: {salida}")
    print(f"     Duracion: {reporte['duracion_segundos'] / 60:.1f} min")
    print()
    print("Para citar en la tesina:")
    print(
        f"  accuracy signer-independent (LOSO, 10 sujetos): "
        f"{reporte['accuracy_media'] * 100:.1f}% +- {reporte['accuracy_desvio'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
