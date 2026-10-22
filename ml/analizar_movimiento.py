# -*- coding: utf-8 -*-
"""
Mide si la ventaja de la BiLSTM sobre el bosque aleatorio se concentra en las
senas con mas movimiento de la mano.

    python ml/analizar_movimiento.py

La hipotesis
------------
La comparacion contra la linea base (ml/comparar_modelos.py) dejo un patron a
simple vista: la BiLSTM aventaja al bosque en Apagar, Aceptar, Atrapar,
Cumpleanos y Aparecer, y pierde en Comida, Espumadera, Fideos, Agradecer y
Asado. Mirandolas, las primeras parecen senas de trayectoria y las segundas de
configuracion sostenida. Si eso fuera cierto, explicaria exactamente que aporta
la componente temporal: el bosque resume la secuencia y pierde el recorrido,
asi que deberia perder justamente donde el recorrido es el significado.

"Parecen" no es un resultado. Este script lo mide.

Como se mide el movimiento
--------------------------
Las features v1 estan centradas en la muneca, asi que **no contienen** donde
esta la mano: por construccion no se puede medir desplazamiento con ellas. El
cache v2 (--con-posicion) si guarda la muneca sin normalizar, 3 valores por
mano, y es el que usa este analisis.

Por cada secuencia se calcula el recorrido de la muneca: la suma de las
distancias entre frames consecutivos en los que la mano esta presente. Se
promedia por clase y se correlaciona con la diferencia de accuracy por clase
entre los dos modelos.

Importante: el recorrido esta en coordenadas del encuadre, no del cuerpo, y las
dos mediciones vienen de caches distintos (v1 para los modelos, v2 para el
movimiento). Sobre las mismas 64 clases y el mismo dataset, igual sirve para
ordenar las senas de menos a mas movimiento, que es todo lo que necesita la
correlacion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.features import (
    FEATURES_PER_HAND_CON_POSICION,
    NUM_HANDS,
    PRESENCE_FLAG_LEN,
)
from lsa64.cache import load_cache
from lsa64.config import REPORTS_DIR


def recorrido_de_secuencia(secuencia: np.ndarray) -> float:
    """Suma de los desplazamientos de la muneca entre frames consecutivos.

    Solo cuenta los tramos en que la mano esta presente en los dos frames: si
    la deteccion se pierde y vuelve, el "salto" no es movimiento de la persona
    sino un agujero de MediaPipe, y sumarlo inflaria justo a las senas con peor
    deteccion.
    """
    total = 0.0
    for mano in range(NUM_HANDS):
        base = mano * FEATURES_PER_HAND_CON_POSICION
        presencia = secuencia[:, base]
        munecas = secuencia[:, base + PRESENCE_FLAG_LEN: base + PRESENCE_FLAG_LEN + 3]

        presentes = presencia > 0.5
        if presentes.sum() < 2:
            continue
        pares = presentes[:-1] & presentes[1:]
        if not pares.any():
            continue
        deltas = np.linalg.norm(munecas[1:] - munecas[:-1], axis=1)
        total += float(deltas[pares].sum())
    return total


def cambio_de_configuracion(secuencia: np.ndarray) -> float:
    """Cuanto cambia la FORMA de la mano a lo largo de la sena.

    El recorrido de la muneca no es todo el movimiento: una sena puede ser
    inmovil en el espacio y aun asi cambiar de configuracion (abrir la mano,
    doblar los dedos). Esto mide lo otro: el cambio medio, entre frames
    consecutivos, del bloque de 21 landmarks ya normalizados respecto de la
    muneca, que es justamente la parte invariante a la posicion.
    """
    cambios = []
    for mano in range(NUM_HANDS):
        base = mano * FEATURES_PER_HAND_CON_POSICION
        presencia = secuencia[:, base]
        inicio = base + PRESENCE_FLAG_LEN + 3  # saltea presencia y muneca
        forma = secuencia[:, inicio: inicio + 63]

        presentes = presencia > 0.5
        if presentes.sum() < 2:
            continue
        pares = presentes[:-1] & presentes[1:]
        if not pares.any():
            continue
        deltas = np.linalg.norm(forma[1:] - forma[:-1], axis=1)
        cambios.append(float(deltas[pares].mean()))
    return float(np.mean(cambios)) if cambios else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Movimiento por clase contra ventaja de la BiLSTM")
    parser.add_argument("--cache-v2", type=str, default=".lsa64_cache/features_v2.npz")
    parser.add_argument("--comparacion", type=str, default=str(REPORTS_DIR / "comparacion_bilstm_rf.json"))
    parser.add_argument("--labels-map", type=str, default="ml/lsa64_labels.json")
    parser.add_argument("--output", type=str, default=str(REPORTS_DIR / "movimiento_por_clase.json"))
    args = parser.parse_args()

    from scipy import stats

    cache = load_cache(Path(args.cache_v2))
    ancho = cache.sequences[0].shape[1]
    if ancho != FEATURES_PER_HAND_CON_POSICION * NUM_HANDS:
        raise SystemExit(
            f"El cache {args.cache_v2} tiene vectores de {ancho} valores: se esperaba v2 "
            f"({FEATURES_PER_HAND_CON_POSICION * NUM_HANDS}). "
            f"Generalo con: python ml/extract_features.py --con-posicion --output {args.cache_v2}"
        )
    print(f"Cache v2: {cache.summary()}")

    recorridos: dict[str, list[float]] = {}
    cambios: dict[str, list[float]] = {}
    for secuencia, label in zip(cache.sequences, cache.labels):
        recorridos.setdefault(label, []).append(recorrido_de_secuencia(secuencia))
        cambios.setdefault(label, []).append(cambio_de_configuracion(secuencia))
    movimiento = {c: float(np.mean(v)) for c, v in recorridos.items()}
    configuracion = {c: float(np.mean(v)) for c, v in cambios.items()}

    with open(args.comparacion, encoding="utf-8") as handle:
        comp = json.load(handle)
    nombre_a, nombre_b = comp["modelo_a"], comp["modelo_b"]

    # La diferencia por clase no esta en el reporte; se recalcula desde las
    # predicciones agrupadas de cada modelo, que si estan.
    with open(Path(comp["reporte_a"]), encoding="utf-8") as handle:
        rep_a = json.load(handle)
    with open(Path(comp["reporte_b"]), encoding="utf-8") as handle:
        rep_b = json.load(handle)
    por_clase_a = rep_a["accuracy_por_clase"]
    por_clase_b = rep_b["accuracy_por_clase"]

    glosas = {}
    mapa_path = Path(args.labels_map)
    if mapa_path.exists():
        with open(mapa_path, encoding="utf-8") as handle:
            glosas = {f"clase_{int(k):02d}": v for k, v in json.load(handle).items()}

    clases = sorted(set(movimiento) & set(por_clase_a) & set(por_clase_b))
    if len(clases) < 10:
        raise SystemExit(f"Solo {len(clases)} clases en comun: revisa que los caches coincidan")

    x = np.array([movimiento[c] for c in clases])
    z = np.array([configuracion[c] for c in clases])
    y = np.array([por_clase_a[c] - por_clase_b[c] for c in clases])

    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    pearson_cfg = stats.pearsonr(z, y)
    spearman_cfg = stats.spearmanr(z, y)

    # Comparacion de los extremos: tercio mas quieto contra tercio con mas recorrido.
    orden = np.argsort(x)
    n_tercio = max(len(clases) // 3, 1)
    quietas = orden[:n_tercio]
    moviles = orden[-n_tercio:]
    mann = stats.mannwhitneyu(y[moviles], y[quietas], alternative="greater")

    print()
    print("=" * 70)
    print(f"MOVIMIENTO DE LA MANO contra VENTAJA DE {nombre_a} SOBRE {nombre_b}")
    print("=" * 70)
    print(f"  clases analizadas : {len(clases)}")
    print()
    print("  A) recorrido de la muneca (cuanto se desplaza la mano)")
    print(f"     Pearson  r = {pearson.statistic:+.3f}   (p = {pearson.pvalue:.4f})")
    print(f"     Spearman r = {spearman.statistic:+.3f}   (p = {spearman.pvalue:.4f})")
    print()
    print("  B) cambio de configuracion (cuanto cambia la forma de la mano)")
    print(f"     Pearson  r = {pearson_cfg.statistic:+.3f}   (p = {pearson_cfg.pvalue:.4f})")
    print(f"     Spearman r = {spearman_cfg.statistic:+.3f}   (p = {spearman_cfg.pvalue:.4f})")
    print()
    print(f"  tercio con MAS recorrido : ventaja media {y[moviles].mean() * 100:+.2f} pp")
    print(f"  tercio con MENOS recorrido: ventaja media {y[quietas].mean() * 100:+.2f} pp")
    print(f"  Mann-Whitney (mas > menos): p = {mann.pvalue:.4f}")
    print("-" * 70)

    def fila(indices, titulo):
        print(f"  {titulo}")
        for i in indices:
            c = clases[i]
            print(f"    {glosas.get(c, c):<16} recorrido {x[i]:7.2f}   ventaja {y[i] * 100:+6.1f} pp")

    fila(orden[-8:][::-1], "Las 8 senas con mas recorrido:")
    print()
    fila(orden[:8], "Las 8 senas con menos recorrido:")
    print("=" * 70)

    reporte = {
        "hipotesis": "la ventaja de la BiLSTM sobre el bosque se concentra en las senas con mas movimiento",
        "modelo_a": nombre_a,
        "modelo_b": nombre_b,
        "cache_movimiento": str(args.cache_v2),
        "n_clases": len(clases),
        "recorrido_muneca": {
            "pearson": {"r": float(pearson.statistic), "p": float(pearson.pvalue)},
            "spearman": {"rho": float(spearman.statistic), "p": float(spearman.pvalue)},
        },
        "cambio_configuracion": {
            "pearson": {"r": float(pearson_cfg.statistic), "p": float(pearson_cfg.pvalue)},
            "spearman": {"rho": float(spearman_cfg.statistic), "p": float(spearman_cfg.pvalue)},
        },
        "tercios": {
            "ventaja_media_mas_movimiento": float(y[moviles].mean()),
            "ventaja_media_menos_movimiento": float(y[quietas].mean()),
            "mann_whitney_p": float(mann.pvalue),
            "n_por_tercio": int(n_tercio),
        },
        "por_clase": {
            clases[i]: {
                "glosa": glosas.get(clases[i], clases[i]),
                "recorrido_medio": float(x[i]),
                "cambio_configuracion_medio": float(z[i]),
                "ventaja_bilstm": float(y[i]),
            }
            for i in range(len(clases))
        },
    }
    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8") as handle:
        json.dump(reporte, handle, indent=2, ensure_ascii=False)
    print()
    print(f"[OK] Reporte guardado en: {salida}")


if __name__ == "__main__":
    main()
