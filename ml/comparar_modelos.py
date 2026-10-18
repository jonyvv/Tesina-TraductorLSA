# -*- coding: utf-8 -*-
"""
Compara dos reportes LOSO entre si (por defecto, BiLSTM contra Random Forest).

    python ml/comparar_modelos.py

Por que no alcanza con restar las medias
----------------------------------------
La BiLSTM da 79,8 % y el Random Forest 78,2 %: 1,6 puntos de diferencia, con
desvios de 8,9 y 7,5. Mirando solo esos numeros, la diferencia parece ruido.
Pero los dos modelos se evaluaron sobre los MISMOS folds y las MISMAS muestras,
asi que la comparacion correcta es pareada, no entre distribuciones
independientes: lo que importa es si la BiLSTM le gana al bosque *en cada
sujeto*, no si los rangos se solapan.

Dos pruebas, que responden preguntas distintas:

- **Wilcoxon pareado sobre los 10 sujetos**: ¿la diferencia por sujeto es
  consistente? Es la prueba a nivel de fold, con n=10.
- **McNemar sobre las 2862 predicciones agrupadas**: de las muestras en que los
  dos modelos difieren, ¿se reparten parejo o uno acierta sistematicamente mas?
  Es la prueba estandar para comparar dos clasificadores sobre el mismo
  conjunto, y tiene mucho mas poder que la anterior porque usa cada muestra.

Ambas se reportan: si dan lo mismo, la conclusion es solida; si difieren, la de
McNemar es la que corresponde citar, aclarando el n.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lsa64.config import REPORTS_DIR


def cargar(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"No existe el reporte {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def accuracy_por_sujeto(reporte: dict) -> dict[str, float]:
    return {f["test_subject"]: f["test_accuracy"] for f in reporte["folds"]}


def comparar(rep_a: dict, rep_b: dict, nombre_a: str, nombre_b: str) -> dict:
    from scipy import stats

    por_sujeto_a = accuracy_por_sujeto(rep_a)
    por_sujeto_b = accuracy_por_sujeto(rep_b)
    sujetos = sorted(set(por_sujeto_a) & set(por_sujeto_b))
    if len(sujetos) < len(por_sujeto_a):
        raise SystemExit(
            f"Los reportes no cubren los mismos sujetos: "
            f"{sorted(por_sujeto_a)} vs {sorted(por_sujeto_b)}"
        )

    diferencias = [por_sujeto_a[s] - por_sujeto_b[s] for s in sujetos]
    gana_a = sum(1 for d in diferencias if d > 0)
    gana_b = sum(1 for d in diferencias if d < 0)

    wilcoxon = stats.wilcoxon(
        [por_sujeto_a[s] for s in sujetos],
        [por_sujeto_b[s] for s in sujetos],
    )

    # --- McNemar sobre las predicciones agrupadas.
    y_true_a = rep_a["predicciones"]["y_true"]
    y_pred_a = rep_a["predicciones"]["y_pred"]
    y_true_b = rep_b["predicciones"]["y_true"]
    y_pred_b = rep_b["predicciones"]["y_pred"]

    mcnemar = None
    if y_true_a == y_true_b:
        aciertos_a = [t == p for t, p in zip(y_true_a, y_pred_a)]
        aciertos_b = [t == p for t, p in zip(y_true_b, y_pred_b)]
        solo_a = sum(1 for a, b in zip(aciertos_a, aciertos_b) if a and not b)
        solo_b = sum(1 for a, b in zip(aciertos_a, aciertos_b) if b and not a)
        ambos = sum(1 for a, b in zip(aciertos_a, aciertos_b) if a and b)
        ninguno = sum(1 for a, b in zip(aciertos_a, aciertos_b) if not a and not b)
        # Prueba binomial exacta sobre los discordantes: bajo H0 se reparten 50/50.
        p_mcnemar = float(stats.binomtest(solo_a, solo_a + solo_b, 0.5).pvalue) if (solo_a + solo_b) else 1.0
        coinciden = sum(1 for pa, pb in zip(y_pred_a, y_pred_b) if pa == pb)
        mcnemar = {
            "aciertan_ambos": ambos,
            "falla_ambos": ninguno,
            f"acierta_solo_{nombre_a}": solo_a,
            f"acierta_solo_{nombre_b}": solo_b,
            "n_discordantes": solo_a + solo_b,
            "p_valor": p_mcnemar,
            "predicciones_identicas": coinciden,
            "proporcion_predicciones_identicas": round(coinciden / max(len(y_pred_a), 1), 4),
        }

    # --- Por clase: en cuantas gana cada uno.
    por_clase_a = rep_a.get("accuracy_por_clase", {})
    por_clase_b = rep_b.get("accuracy_por_clase", {})
    clases = sorted(set(por_clase_a) & set(por_clase_b))
    dif_clase = {c: round(por_clase_a[c] - por_clase_b[c], 4) for c in clases}
    mejores_a = sorted(dif_clase.items(), key=lambda kv: kv[1], reverse=True)[:8]
    mejores_b = sorted(dif_clase.items(), key=lambda kv: kv[1])[:8]

    return {
        "modelo_a": nombre_a,
        "modelo_b": nombre_b,
        "media_a": rep_a["accuracy_media"],
        "media_b": rep_b["accuracy_media"],
        "desvio_a": rep_a["accuracy_desvio"],
        "desvio_b": rep_b["accuracy_desvio"],
        "diferencia_media": rep_a["accuracy_media"] - rep_b["accuracy_media"],
        "por_sujeto": {
            s: {
                nombre_a: round(por_sujeto_a[s], 4),
                nombre_b: round(por_sujeto_b[s], 4),
                "diferencia": round(por_sujeto_a[s] - por_sujeto_b[s], 4),
            }
            for s in sujetos
        },
        "sujetos_gana_a": gana_a,
        "sujetos_gana_b": gana_b,
        "diferencia_media_pareada": statistics.mean(diferencias),
        "diferencia_desvio_pareada": statistics.stdev(diferencias) if len(diferencias) > 1 else 0.0,
        "wilcoxon": {"estadistico": float(wilcoxon.statistic), "p_valor": float(wilcoxon.pvalue), "n": len(sujetos)},
        "mcnemar": mcnemar,
        "clases_donde_gana_a": dict(mejores_a),
        "clases_donde_gana_b": dict(mejores_b),
        "n_clases_gana_a": sum(1 for v in dif_clase.values() if v > 0),
        "n_clases_gana_b": sum(1 for v in dif_clase.values() if v < 0),
        "n_clases_empate": sum(1 for v in dif_clase.values() if v == 0),
    }


def imprimir(comp: dict) -> None:
    a, b = comp["modelo_a"], comp["modelo_b"]
    print()
    print("=" * 68)
    print(f"COMPARACION PAREADA — {a} contra {b}")
    print("=" * 68)
    print(f"  {a:<16} {comp['media_a'] * 100:.1f}% +- {comp['desvio_a'] * 100:.1f}%")
    print(f"  {b:<16} {comp['media_b'] * 100:.1f}% +- {comp['desvio_b'] * 100:.1f}%")
    print(f"  diferencia      {comp['diferencia_media'] * 100:+.1f} puntos")
    print("-" * 68)
    print(f"  {'sujeto':<12} {a:>10} {b:>10} {'dif':>9}")
    for sujeto, valores in comp["por_sujeto"].items():
        print(f"  {sujeto:<12} {valores[a] * 100:>9.1f}% {valores[b] * 100:>9.1f}% "
              f"{valores['diferencia'] * 100:>+8.1f}")
    print("-" * 68)
    print(f"  gana {a} en {comp['sujetos_gana_a']} de {len(comp['por_sujeto'])} sujetos, "
          f"gana {b} en {comp['sujetos_gana_b']}")
    print(f"  diferencia pareada: {comp['diferencia_media_pareada'] * 100:+.2f} "
          f"+- {comp['diferencia_desvio_pareada'] * 100:.2f} puntos")
    w = comp["wilcoxon"]
    print(f"  Wilcoxon (n={w['n']} sujetos): p = {w['p_valor']:.4f}")
    if comp["mcnemar"]:
        m = comp["mcnemar"]
        print(f"  McNemar sobre {m['aciertan_ambos'] + m['falla_ambos'] + m['n_discordantes']} predicciones:")
        print(f"    aciertan ambos      : {m['aciertan_ambos']}")
        print(f"    falla ambos         : {m['falla_ambos']}")
        print(f"    acierta solo {a:<10}: {m[f'acierta_solo_{a}']}")
        print(f"    acierta solo {b:<10}: {m[f'acierta_solo_{b}']}")
        print(f"    p = {m['p_valor']:.4f}")
        print(f"    predicciones identicas: {m['proporcion_predicciones_identicas'] * 100:.1f}%")
    print("-" * 68)
    print(f"  por clase: gana {a} en {comp['n_clases_gana_a']}, gana {b} en "
          f"{comp['n_clases_gana_b']}, empatan {comp['n_clases_empate']}")
    print(f"  donde mas gana {a}: " + ", ".join(
        f"{c} ({v * 100:+.0f})" for c, v in list(comp["clases_donde_gana_a"].items())[:5]))
    print(f"  donde mas gana {b}: " + ", ".join(
        f"{c} ({v * 100:+.0f})" for c, v in list(comp["clases_donde_gana_b"].items())[:5]))
    print("=" * 68)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara dos reportes LOSO de forma pareada")
    parser.add_argument("--a", type=str, default=None, help="reporte del modelo A (default: BiLSTM)")
    parser.add_argument("--b", type=str, default=None, help="reporte del modelo B (default: Random Forest)")
    parser.add_argument("--nombre-a", type=str, default="BiLSTM")
    parser.add_argument("--nombre-b", type=str, default="RandomForest")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reportes = REPORTS_DIR
    path_a = Path(args.a) if args.a else reportes / "loso_lsa64.json"
    path_b = Path(args.b) if args.b else reportes / "baseline_rf_loso.json"
    salida = Path(args.output) if args.output else reportes / "comparacion_bilstm_rf.json"

    comp = comparar(cargar(path_a), cargar(path_b), args.nombre_a, args.nombre_b)
    comp["reporte_a"] = str(path_a)
    comp["reporte_b"] = str(path_b)
    imprimir(comp)

    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8") as handle:
        json.dump(comp, handle, indent=2, ensure_ascii=False)
    print()
    print(f"[OK] Comparacion guardada en: {salida}")


if __name__ == "__main__":
    main()
