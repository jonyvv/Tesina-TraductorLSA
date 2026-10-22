# -*- coding: utf-8 -*-
"""
Linea base no neuronal: Random Forest sobre LSA64, evaluado con LOSO.

    python ml/baseline_rf.py

Por que hace falta
------------------
El planteo del proyecto promete comparar Random Forest contra una red neuronal.
Sin esa comparacion no hay forma de justificar la BiLSTM: puede que el orden
temporal no aporte nada y que un modelo tabular sobre un resumen de la
secuencia alcance el mismo numero. Ese tambien seria un resultado.

Como se compara de igual a igual
--------------------------------
- Mismo cache de features que `ml/train_lsa64.py` y `ml/evaluate_loso.py`.
- Mismos folds: `construir_folds(..., val_subjects=1)`, los mismos 10 que usa
  la BiLSTM, armados de forma determinista.
- Mismo conjunto de entrenamiento: los 8 sujetos de `fold.train_idx`. El sujeto
  de validacion queda afuera aunque el Random Forest no lo necesite (no hay
  early stopping), porque si entrenara con 9 sujetos estaria viendo mas datos
  que la red y la comparacion no mediria el modelo sino el tamano del train.
  Con --incluir-val se reporta ademas esa variante, como dato secundario.

La representacion
-----------------
Un Random Forest no toma secuencias de largo variable, asi que cada secuencia
se resume en estadisticos por feature: media, desvio, minimo y maximo. El
vector pasa de (T, 138) a 552 valores de largo fijo. Es deliberadamente el
resumen mas simple posible: el punto de la comparacion es cuanto se pierde al
descartar el orden temporal, y cualquier agregado mas elaborado (percentiles,
deltas, coeficientes) lo volveria menos interpretable como piso de referencia.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lsa64.config import LSA64TrainingConfig, REPORTS_DIR, default_cache_path
from lsa64.evaluation import construir_folds, verificar_fold
from lsa64.training import load_features

AGREGADOS = ("media", "desvio", "minimo", "maximo")


def resumir_secuencia(secuencia: np.ndarray) -> np.ndarray:
    """(T, D) -> (4*D,) con media, desvio, minimo y maximo por feature.

    El desvio de una secuencia de un solo frame es 0, no NaN: `np.std` sobre
    un eje de largo 1 ya devuelve 0, pero lo documentamos porque min_seq_len
    podria bajarse a 1 y el NaN se propagaria silenciosamente al bosque.
    """
    if secuencia.ndim != 2:
        raise ValueError(f"Se esperaba una secuencia 2D (T, D), llego {secuencia.shape}")
    if len(secuencia) == 0:
        raise ValueError("Secuencia vacia: no se puede resumir")

    return np.concatenate([
        secuencia.mean(axis=0),
        secuencia.std(axis=0),
        secuencia.min(axis=0),
        secuencia.max(axis=0),
    ]).astype(np.float32)


def construir_matriz(secuencias: list[np.ndarray]) -> np.ndarray:
    return np.vstack([resumir_secuencia(s) for s in secuencias])


def evaluar_rf_loso(
    cache,
    *,
    n_estimators: int,
    max_depth: int | None,
    seed: int,
    incluir_val: bool = False,
) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(cache.labels).astype(np.int64)
    clases = list(label_encoder.classes_)

    print("Resumiendo secuencias (media/desvio/min/max por feature)...", flush=True)
    X = construir_matriz(cache.sequences)
    print(f"  matriz: {X.shape[0]} muestras x {X.shape[1]} features")
    print()

    folds = construir_folds(cache.subjects, val_subjects=1)
    print(f"Folds  : {len(folds)} (uno por sujeto, los mismos que la BiLSTM)")
    print(f"Clases : {len(clases)}")
    print(f"Config : n_estimators={n_estimators} max_depth={max_depth} seed={seed}")
    print()

    resultados = []
    y_true_global: list[int] = []
    y_pred_global: list[int] = []

    for numero, fold in enumerate(folds, start=1):
        problemas = verificar_fold(fold, cache.subjects)
        if problemas:
            raise RuntimeError(f"Fold {numero} mal armado: {problemas}")

        train_idx = list(fold.train_idx)
        if incluir_val:
            train_idx += list(fold.val_idx)

        print(f"[Fold {numero}/{len(folds)}] {fold.resumen()}", flush=True)
        inicio = time.perf_counter()
        bosque = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
        )
        bosque.fit(X[train_idx], y[train_idx])
        y_pred = bosque.predict(X[fold.test_idx])
        y_true = y[fold.test_idx]
        accuracy = float((y_pred == y_true).mean())
        duracion = time.perf_counter() - inicio

        y_true_global.extend(int(v) for v in y_true)
        y_pred_global.extend(int(v) for v in y_pred)

        resultados.append({
            "fold": numero,
            "test_subject": fold.test_subject,
            "val_subjects": fold.val_subjects,
            "n_train": len(train_idx),
            "n_test": len(fold.test_idx),
            "test_accuracy": accuracy,
            "duracion_segundos": round(duracion, 1),
        })
        print(f"           accuracy: {accuracy:.4f}  ({duracion:.1f}s)", flush=True)

    accs = [r["test_accuracy"] for r in resultados]
    media = statistics.mean(accs)
    desvio = statistics.stdev(accs) if len(accs) > 1 else 0.0
    peor = min(resultados, key=lambda r: r["test_accuracy"])
    mejor = max(resultados, key=lambda r: r["test_accuracy"])

    totales = Counter(y_true_global)
    aciertos = Counter(t for t, p in zip(y_true_global, y_pred_global) if t == p)
    por_clase = {
        clases[label]: round(aciertos[label] / total, 4)
        for label, total in sorted(totales.items())
    }

    return {
        "modelo": "RandomForest",
        "protocolo": "leave-one-subject-out",
        "representacion": {
            "agregados": list(AGREGADOS),
            "n_features": int(X.shape[1]),
            "descripcion": "resumen por secuencia: media/desvio/min/max de cada feature",
        },
        "entrenado_con_sujeto_de_validacion": incluir_val,
        "n_folds": len(folds),
        "accuracy_media": media,
        "accuracy_desvio": desvio,
        "accuracy_min": peor["test_accuracy"],
        "accuracy_max": mejor["test_accuracy"],
        "sujeto_peor": peor["test_subject"],
        "sujeto_mejor": mejor["test_subject"],
        "accuracy_global_agrupada": sum(1 for t, p in zip(y_true_global, y_pred_global) if t == p)
        / max(len(y_true_global), 1),
        "n_predicciones": len(y_true_global),
        "hiperparametros": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "seed": seed,
        },
        "folds": resultados,
        "accuracy_por_clase": por_clase,
        "labels": clases,
        "predicciones": {"y_true": y_true_global, "y_pred": y_pred_global},
    }


def imprimir_resumen(reporte: dict) -> None:
    print()
    print("=" * 62)
    print("LINEA BASE RANDOM FOREST — LEAVE-ONE-SUBJECT-OUT")
    print("=" * 62)
    for r in reporte["folds"]:
        barra = "#" * int(round(r["test_accuracy"] * 40))
        print(f"  {r['test_subject']:<12} {r['test_accuracy']:.4f}  {barra}")
    print("-" * 62)
    print(f"  media +- desvio : {reporte['accuracy_media']:.4f} +- {reporte['accuracy_desvio']:.4f}")
    print(f"  rango           : {reporte['accuracy_min']:.4f} ({reporte['sujeto_peor']}) "
          f"a {reporte['accuracy_max']:.4f} ({reporte['sujeto_mejor']})")
    print(f"  agrupada        : {reporte['accuracy_global_agrupada']:.4f} "
          f"sobre {reporte['n_predicciones']} predicciones")
    print("=" * 62)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linea base Random Forest sobre LSA64 (LOSO)")
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=str(Path.cwd() / ".lsa64_cache"))
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPORTS_DIR / "baseline_rf_loso.json"),
        help="JSON con el reporte completo (folds, accuracy por clase, predicciones)",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-empty-frames",
        action="store_true",
        help="conservar los frames sin mano detectada, preservando la grilla temporal",
    )
    parser.add_argument(
        "--con-posicion",
        action="store_true",
        help="v2: agrega la posicion de la muneca al vector (138 -> 144)",
    )
    parser.add_argument(
        "--incluir-val",
        action="store_true",
        help="entrenar tambien con el sujeto de validacion (9 en vez de 8). "
             "Deja de ser comparable de igual a igual con la BiLSTM.",
    )
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
        min_seq_len=args.min_seq_len,
        keep_empty_frames=args.keep_empty_frames,
        incluir_posicion=args.con_posicion,
        seed=args.seed,
    )

    cache = load_features(config)
    print()

    inicio = time.perf_counter()
    reporte = evaluar_rf_loso(
        cache,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
        incluir_val=args.incluir_val,
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
        f"  linea base Random Forest (LOSO, {reporte['n_folds']} sujetos): "
        f"{reporte['accuracy_media'] * 100:.1f}% +- {reporte['accuracy_desvio'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
