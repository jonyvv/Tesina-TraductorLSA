# -*- coding: utf-8 -*-
"""
Validacion leave-one-subject-out (LOSO) sobre LSA64.

Por que hace falta
------------------
Un unico split con 2 sujetos de test da un numero muy inestable: en este
proyecto, tres corridas sobre datos practicamente identicos dieron 86,05 %,
78,55 % y 81,25 %. Ese spread no es ruido de medicion, es la varianza real de
estimar con dos personas. LOSO entrena 10 veces, dejando afuera a un sujeto
distinto cada vez, y reporta media +- desvio: eso si es un resultado.

Como se arman los folds
-----------------------
Para cada sujeto S:
    test  = S
    val   = el sujeto siguiente en orden ciclico (nunca S)
    train = los 8 restantes

La validacion sale de los sujetos de entrenamiento, nunca del de test. Es lo
que permite seguir usando early stopping sin contaminar la evaluacion.

Con --val-subjects 0 se desactiva la validacion y se entrenan todos los epochs
fijos: mas simple de explicar, pero sin seleccion de modelo.

Efecto lateral util: al terminar los 10 folds, CADA muestra del dataset fue
predicha exactamente una vez por un modelo que nunca vio a esa persona. Las
predicciones agrupadas sirven para armar la matriz de confusion sobre el
dataset completo.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import FeatureCache
from .config import LSA64TrainingConfig
from .fitting import ajustar_modelo


@dataclass
class Fold:
    test_subject: str
    val_subjects: list[str]
    train_subjects: list[str]
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]

    def resumen(self) -> str:
        val = ",".join(self.val_subjects) if self.val_subjects else "-"
        return (
            f"test={self.test_subject} val={val} "
            f"(train={len(self.train_idx)} val={len(self.val_idx)} test={len(self.test_idx)})"
        )


def construir_folds(subjects: list[str | None], val_subjects: int = 1) -> list[Fold]:
    """Un fold por sujeto. Determinista: no depende de ninguna semilla."""
    unicos = sorted({s for s in subjects if s})
    if len(unicos) < 2:
        raise ValueError(
            f"Hacen falta al menos 2 sujetos para LOSO, hay {len(unicos)}: {unicos}. "
            "Revisa que el parseo de sujeto este funcionando."
        )
    if val_subjects >= len(unicos) - 1:
        raise ValueError(
            f"val_subjects={val_subjects} deja sin sujetos de entrenamiento "
            f"(hay {len(unicos)} sujetos en total)."
        )

    indices_por_sujeto: dict[str, list[int]] = {}
    for idx, subject in enumerate(subjects):
        if subject:
            indices_por_sujeto.setdefault(subject, []).append(idx)

    folds: list[Fold] = []
    for posicion, test_subject in enumerate(unicos):
        # Los de validacion se toman en orden ciclico a partir del siguiente,
        # asi cada sujeto cumple los dos roles y la eleccion es reproducible.
        restantes = [unicos[(posicion + 1 + k) % len(unicos)] for k in range(len(unicos) - 1)]
        val = restantes[:val_subjects]
        train = restantes[val_subjects:]

        folds.append(
            Fold(
                test_subject=test_subject,
                val_subjects=val,
                train_subjects=train,
                train_idx=[i for s in train for i in indices_por_sujeto[s]],
                val_idx=[i for s in val for i in indices_por_sujeto[s]],
                test_idx=list(indices_por_sujeto[test_subject]),
            )
        )
    return folds


def verificar_fold(fold: Fold, subjects: list[str | None]) -> list[str]:
    """Devuelve los problemas encontrados; lista vacia = fold sano."""
    problemas = []

    def sujetos(indices: list[int]) -> set[str]:
        return {subjects[i] for i in indices if subjects[i]}

    train_s, val_s, test_s = sujetos(fold.train_idx), sujetos(fold.val_idx), sujetos(fold.test_idx)
    if test_s != {fold.test_subject}:
        problemas.append(f"test contiene {sorted(test_s)}, deberia ser solo {fold.test_subject}")
    if train_s & test_s:
        problemas.append(f"fuga train/test: {sorted(train_s & test_s)}")
    if val_s & test_s:
        problemas.append(f"fuga val/test: {sorted(val_s & test_s)}")
    if train_s & val_s:
        problemas.append(f"fuga train/val: {sorted(train_s & val_s)}")
    if not fold.train_idx:
        problemas.append("train vacio")
    if not fold.test_idx:
        problemas.append("test vacio")
    return problemas


def evaluar_loso(
    cache: FeatureCache,
    config: LSA64TrainingConfig,
    val_subjects: int = 1,
    verbose_epochs: bool = False,
) -> dict:
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    label_list = label_encoder.fit_transform(cache.labels).astype(np.int64)
    clases = list(label_encoder.classes_)
    folds = construir_folds(cache.subjects, val_subjects=val_subjects)

    print(f"Folds  : {len(folds)} (uno por sujeto)")
    print(f"Clases : {len(clases)}")
    print(f"Config : hidden={config.hidden_size} lr={config.learning_rate} "
          f"batch={config.batch_size} epochs<={config.epochs} patience={config.patience}")
    print()

    resultados = []
    y_true_global: list[int] = []
    y_pred_global: list[int] = []

    for numero, fold in enumerate(folds, start=1):
        problemas = verificar_fold(fold, cache.subjects)
        if problemas:
            raise RuntimeError(f"Fold {numero} mal armado: {problemas}")

        print(f"[Fold {numero}/{len(folds)}] {fold.resumen()}", flush=True)
        resultado = ajustar_modelo(
            cache.sequences,
            label_list,
            n_clases=len(clases),
            train_idx=fold.train_idx,
            val_idx=fold.val_idx,
            test_idx=fold.test_idx,
            config=config,
            verbose=verbose_epochs,
        )
        y_true_global.extend(resultado.y_true)
        y_pred_global.extend(resultado.y_pred)

        resultados.append({
            "fold": numero,
            "test_subject": fold.test_subject,
            "val_subjects": fold.val_subjects,
            "n_train": len(fold.train_idx),
            "n_val": len(fold.val_idx),
            "n_test": len(fold.test_idx),
            "test_accuracy": resultado.test_accuracy,
            "best_val_accuracy": resultado.best_val_accuracy,
            "best_epoch": resultado.best_epoch,
            "epochs_corridas": resultado.epochs_corridas,
        })
        print(
            f"           accuracy: {resultado.test_accuracy:.4f}"
            f"  (epochs: {resultado.epochs_corridas}, mejor: {resultado.best_epoch})",
            flush=True,
        )

    accs = [r["test_accuracy"] for r in resultados]
    media = statistics.mean(accs)
    desvio = statistics.stdev(accs) if len(accs) > 1 else 0.0
    peor = min(resultados, key=lambda r: r["test_accuracy"])
    mejor = max(resultados, key=lambda r: r["test_accuracy"])

    # Accuracy por clase sobre TODAS las predicciones: cada muestra del dataset
    # fue evaluada exactamente una vez, por un modelo que no vio a esa persona.
    from collections import Counter

    totales = Counter(y_true_global)
    aciertos = Counter(t for t, p in zip(y_true_global, y_pred_global) if t == p)
    por_clase = {
        clases[label]: round(aciertos[label] / total, 4)
        for label, total in sorted(totales.items())
    }

    return {
        "protocolo": "leave-one-subject-out",
        "n_folds": len(folds),
        "val_subjects_por_fold": val_subjects,
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
            "hidden_size": config.hidden_size,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "epochs_max": config.epochs,
            "patience": config.patience,
            "seed": config.seed,
        },
        "folds": resultados,
        "accuracy_por_clase": por_clase,
        "labels": clases,
        "predicciones": {"y_true": y_true_global, "y_pred": y_pred_global},
    }


def imprimir_resumen(reporte: dict) -> None:
    print()
    print("=" * 62)
    print("RESULTADO LEAVE-ONE-SUBJECT-OUT")
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

    peores = sorted(reporte["accuracy_por_clase"].items(), key=lambda kv: kv[1])[:8]
    print("Clases mas dificiles (sobre el dataset completo):")
    for clase, acc in peores:
        print(f"  {clase:<12} {acc:.2f}")
