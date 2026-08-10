# -*- coding: utf-8 -*-
"""
ml/train.py

Entrena y compara dos modelos sobre las muestras ESTÁTICAS del dataset
(abecedario / poses fijas):

  1. RandomForestClassifier -> baseline clásico (rápido, interpretable)
  2. MLPClassifier          -> "red neuronal", tal como pide el anteproyecto

Ambos comparten el mismo split train/test (agrupado por sesión, para evitar
fuga de datos) y se evalúan con las mismas métricas, para poder incluir la
comparación directamente como experimento en la tesina (ver docs/ARQUITECTURA.md,
sección "Experimentos sugeridos").

El modelo con mejor F1 macro en test se guarda como `modelo_lse.joblib` para que
lo cargue el backend. El otro se guarda igual, con sufijo, para referencia/comparación.

Para las muestras DINÁMICAS (secuencias, palabras con movimiento) usar en cambio
`train_dynamic_lstm.py`, que entrena un modelo secuencial (LSTM) — un
RandomForest/MLP tabular no puede modelar movimiento (ver docs/ARQUITECTURA.md).

Uso:
    python train.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # para poder correr sin display (servidores, CI)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.features import FEATURE_VECTOR_LENGTH, FEATURE_VERSION  # noqa: E402

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "sign_language_dataset"
MODELS_DIR = Path(__file__).resolve().parents[1] / "backend" / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "ml" / "reports"


def cargar_muestras_estaticas():
    """Carga solo `type == "static"`. Devuelve X, y, groups (para split por sesión)."""
    X, y, groups = [], [], []
    descartadas = 0

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"No existe {DATASET_DIR}. Corré ml/capture_dataset.py primero."
        )

    for path in DATASET_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                sample = json.load(f)
        except Exception:
            descartadas += 1
            continue

        if sample.get("type") != "static":
            continue
        if sample.get("feature_version") != FEATURE_VERSION:
            print(f"[!] Saltando {path.name}: feature_version distinta "
                  f"({sample.get('feature_version')} != {FEATURE_VERSION})")
            descartadas += 1
            continue

        vector = sample.get("payload")
        if not isinstance(vector, list) or len(vector) != FEATURE_VECTOR_LENGTH:
            print(f"[!] Saltando {path.name}: longitud de vector inesperada")
            descartadas += 1
            continue

        X.append(vector)
        y.append(sample["label"])
        # Agrupar por sujeto+sesión: así el split evita que frames de la misma
        # sesión terminen repartidos entre train y test (fuga de datos).
        groups.append(f"{sample.get('sujeto', 'na')}_{sample.get('sesion', 'na')}")

    print(f"Muestras estáticas cargadas: {len(X)} (descartadas: {descartadas})")
    return np.array(X, dtype=np.float32), np.array(y), np.array(groups)


def evaluar_y_reportar(nombre: str, model, X_test, y_test, label_encoder, clases):
    y_pred = model.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"\n=== {nombre} ===")
    print(classification_report(y_test, y_pred, target_names=clases, zero_division=0))
    print(f"F1 macro: {f1_macro:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(clases) * 0.6), max(5, len(clases) * 0.6)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clases, yticklabels=clases, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {nombre}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"confusion_matrix_{nombre.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    return f1_macro


def guardar_modelo(model, label_encoder, nombre_archivo: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_version": FEATURE_VERSION,
            "feature_vector_length": FEATURE_VECTOR_LENGTH,
        },
        MODELS_DIR / nombre_archivo,
    )
    print(f"[OK] Modelo guardado: {MODELS_DIR / nombre_archivo}")


def main() -> None:
    X, y_raw, groups = cargar_muestras_estaticas()
    if len(X) == 0:
        print("No hay muestras para entrenar.")
        return

    conteo = Counter(y_raw)
    print("Distribución de clases:", dict(conteo))
    if min(conteo.values()) < 5:
        print("[!] Advertencia: alguna clase tiene menos de 5 muestras. "
              "Los resultados no van a ser confiables todavía.")

    # --- Chequeo: hacen falta al menos 2 clases ---
    # Un clasificador entrenado con una sola clase no clasifica nada: responde
    # siempre esa etiqueta, con confianza 1.0, para cualquier entrada. Además
    # rompe de formas silenciosas: MLPClassifier con una sola clase devuelve un
    # `predict_proba` de DOS columnas pero un `classes_` de UN elemento, así que
    # el backend puede indexar fuera de rango al traducir el argmax a etiqueta.
    if len(conteo) < 2:
        unica = next(iter(conteo))
        print(
            f"\n[ERROR] El dataset tiene una sola seña ('{unica}').\n\n"
            f"Un clasificador necesita al menos dos clases para poder distinguir\n"
            f"algo: con una sola, el modelo responde siempre '{unica}' con\n"
            f"confianza 1.0 ante cualquier mano que vea, incluso una que no esté\n"
            f"haciendo esa seña. No sirve ni para probar el sistema de punta a\n"
            f"punta, porque no se puede saber si acierta o si siempre dice lo mismo.\n\n"
            f"Capturá una segunda seña (bien distinta de la primera, para que la\n"
            f"diferencia sea clara) y volvé a correr este script.\n\n"
            f"No se entrenó ningún modelo."
        )
        sys.exit(1)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    clases = list(label_encoder.classes_)

    # --- Chequeo previo: ¿el dataset permite un split agrupado sano? ---
    # El split agrupa por sujeto+sesión para evitar fuga de datos. Pero si una
    # etiqueta se capturó en UNA SOLA sesión, esa etiqueta entera cae toda de un
    # lado del split: o queda fuera del entrenamiento (el modelo nunca la
    # aprende) o fuera del test (no se puede evaluar). En el caso extremo, con
    # una sesión por clase, el split separa POR CLASE en vez de por sesión: se
    # entrena con una sola clase y se testea contra otra, F1 = 0.
    # Esto no lanzaba ningún error: guardaba el .joblib igual y el backend lo
    # servía prediciendo siempre lo mismo con confianza 1.0.
    sesiones_por_clase = {
        clase: {g for g, etq in zip(groups, y_raw) if etq == clase}
        for clase in conteo
    }
    clases_una_sesion = {c: s for c, s in sesiones_por_clase.items() if len(s) < 2}
    if clases_una_sesion:
        print("\n[ERROR] Estas señas están capturadas en una sola sesión:")
        for clase, sesiones in sorted(clases_una_sesion.items()):
            print(f"    - '{clase}': solo {sorted(sesiones)}")
        print(
            "\nCon un split agrupado por sujeto+sesión, cada una de esas señas cae\n"
            "entera de un solo lado (todo en train o todo en test), así que las\n"
            "métricas no significan nada. Capturá cada seña en AL MENOS 2 sesiones\n"
            "distintas (idealmente también con más de un sujeto) y volvé a correr.\n"
            "\nNo se entrenó ningún modelo."
        )
        sys.exit(1)

    # Split agrupado por sesión (no por muestra individual) para evitar fuga de datos.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Red de seguridad: incluso con ≥2 sesiones por clase, un split desafortunado
    # puede dejar una clase entera fuera de train. Mejor frenar que guardar un
    # modelo que no puede predecir esa clase.
    clases_en_train = set(y_train)
    if len(clases_en_train) < len(clases):
        faltantes = [clases[i] for i in range(len(clases)) if i not in clases_en_train]
        print(f"\n[ERROR] El split dejó estas clases fuera del entrenamiento: {faltantes}")
        print("El modelo nunca podría predecirlas. Revisá el balance de sesiones "
              "por clase. No se entrenó ningún modelo.")
        sys.exit(1)

    print(f"\nTrain: {len(X_train)} muestras | Test: {len(X_test)} muestras "
          f"(split agrupado por sujeto+sesión)")

    # --- Modelo 1: Random Forest (baseline) ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    f1_rf = evaluar_y_reportar("Random Forest", rf, X_test, y_test, label_encoder, clases)

    # IMPORTANTE: pasar cv=<int> a cross_val_score usa StratifiedKFold por
    # defecto, que IGNORA el parámetro `groups` de forma silenciosa (sklearn
    # ni siquiera tira error, solo un warning). Para que la validación
    # cruzada realmente respete el agrupamiento por sujeto+sesión (y no
    # vuelva a introducir la fuga de datos que se corrigió en el split de
    # train/test), hay que pasar un splitter GroupKFold explícito.
    n_grupos_disponibles = len(set(groups))
    n_splits_cv = min(5, n_grupos_disponibles)
    if n_splits_cv < 2:
        print("[!] No hay suficientes grupos (sujeto+sesión) distintos para "
              "hacer validación cruzada agrupada. Se omite este paso.")
    else:
        group_kfold = GroupKFold(n_splits=n_splits_cv)
        cv_rf = cross_val_score(rf, X, y, cv=group_kfold, groups=groups)
        print(f"Random Forest — CV score (GroupKFold, {n_splits_cv} folds): "
              f"{cv_rf.mean():.4f} (+/- {cv_rf.std():.4f})")

    # --- Modelo 2: MLP (red neuronal) ---
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        max_iter=2000,
        random_state=42,
        early_stopping=True,
    )
    mlp.fit(X_train, y_train)
    f1_mlp = evaluar_y_reportar("MLP", mlp, X_test, y_test, label_encoder, clases)

    # --- Guardar ambos, y marcar como modelo activo el de mejor F1 macro ---
    guardar_modelo(rf, label_encoder, "modelo_random_forest.joblib")
    guardar_modelo(mlp, label_encoder, "modelo_mlp.joblib")

    mejor_nombre, mejor_modelo, mejor_f1 = (
        ("Random Forest", rf, f1_rf) if f1_rf >= f1_mlp else ("MLP", mlp, f1_mlp)
    )
    guardar_modelo(mejor_modelo, label_encoder, "modelo_lse.joblib")  # el que carga el backend

    print(f"\n=== Resumen ===")
    print(f"Random Forest F1 macro: {f1_rf:.4f}")
    print(f"MLP F1 macro:           {f1_mlp:.4f}")
    print(f"Modelo activo (backend/models/modelo_lse.joblib): {mejor_nombre} (F1={mejor_f1:.4f})")
    print(f"Matrices de confusión guardadas en: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
