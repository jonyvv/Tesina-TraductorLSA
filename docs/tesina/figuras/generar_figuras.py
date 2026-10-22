# -*- coding: utf-8 -*-
"""
docs/tesina/figuras/generar_figuras.py

Genera las figuras de la tesina a partir de los reportes versionados del
repositorio (ml/reports/*.json y backend/models/modelo_lsa64_lstm.json). Los
diagramas esquemáticos (arquitectura, secuencia, vector de features, etc.) se
dibujan con matplotlib para no depender de herramientas externas.

Correr desde la raíz del repo, con el entorno de ML:
    .venv-ml\\Scripts\\python docs/tesina/figuras/generar_figuras.py

Si se vuelve a correr el LOSO o se reentrena el modelo, alcanza con volver a
ejecutar este script y reconstruir el documento: ningún número de las figuras
está escrito a mano, salvo la retención por sujeto (ver RETENCION_POR_SUJETO).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

RAIZ = Path(__file__).resolve().parents[3]
SALIDA = Path(__file__).resolve().parent
REPORTES = RAIZ / "ml" / "reports"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

AZUL = "#2B5C8A"
AZUL_CLARO = "#9DB9D5"
NARANJA = "#D9822B"
GRIS = "#6B6B6B"
VERDE = "#3C8D5A"
ROJO = "#B5463C"

# Retención de frames con mano detectada por sujeto. No está en el reporte LOSO
# (sale del caché, que no se versiona): copiada de docs/PROXIMOS_PASOS.md §1.
RETENCION_POR_SUJETO = {
    "sujeto_01": 75.7, "sujeto_02": 73.7, "sujeto_03": 63.5, "sujeto_04": 68.3,
    "sujeto_05": 65.5, "sujeto_06": 73.7, "sujeto_07": 73.2, "sujeto_08": 58.5,
    "sujeto_09": 74.6, "sujeto_10": 69.7,
}


def cargar(nombre: str) -> dict:
    return json.loads((REPORTES / nombre).read_text(encoding="utf-8"))


def glosas() -> dict[str, str]:
    mapa = json.loads((RAIZ / "ml" / "lsa64_labels.json").read_text(encoding="utf-8"))
    return {f"clase_{int(k):02d}": v for k, v in mapa.items()}


def guardar(fig, nombre: str) -> None:
    fig.savefig(SALIDA / nombre)
    plt.close(fig)
    print("  ", nombre)


def acc_por_sujeto(reporte: dict) -> dict[str, float]:
    return {f["test_subject"]: f["test_accuracy"] * 100 for f in reporte["folds"]}


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------

def fig_loso_por_sujeto() -> None:
    rep = cargar("loso_lsa64.json")
    acc = acc_por_sujeto(rep)
    sujetos = sorted(acc)
    valores = [acc[s] for s in sujetos]
    media = rep["accuracy_media"] * 100
    desvio = rep["accuracy_desvio"] * 100

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colores = [ROJO if v < media - desvio else AZUL for v in valores]
    barras = ax.bar([s.replace("sujeto_", "S") for s in sujetos], valores, color=colores, zorder=2)
    ax.axhline(media, color=NARANJA, lw=1.5, label=f"Media {media:.1f} %")
    ax.axhspan(media - desvio, media + desvio, color=NARANJA, alpha=0.12, zorder=0,
               label=f"± 1 desvío ({desvio:.1f} pp)")
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylim(50, 100)
    ax.set_ylabel("Exactitud en la persona de prueba (%)")
    ax.set_xlabel("Persona dejada afuera (partición LOSO)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False, fontsize=8)
    guardar(fig, "fig_loso_por_sujeto.png")


def fig_retencion_vs_accuracy() -> None:
    acc = acc_por_sujeto(cargar("loso_lsa64.json"))
    sujetos = sorted(acc)
    x = np.array([RETENCION_POR_SUJETO[s] for s in sujetos])
    y = np.array([acc[s] for s in sujetos])
    r = np.corrcoef(x, y)[0, 1]
    m, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.scatter(x, y, color=AZUL, s=40, zorder=3)
    for s, xi, yi in zip(sujetos, x, y):
        ax.annotate(s.replace("sujeto_", "S"), (xi, yi), textcoords="offset points",
                    xytext=(5, 4), fontsize=8, color=GRIS)
    xs = np.linspace(x.min() - 1, x.max() + 1, 10)
    ax.plot(xs, m * xs + b, color=NARANJA, lw=1.2, ls="--", label=f"Tendencia lineal (r = {r:.2f})")
    ax.set_xlabel("Retención: frames con mano detectada (%)")
    ax.set_ylabel("Exactitud LOSO (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    guardar(fig, "fig_retencion_vs_accuracy.png")


def fig_filtrado_vs_grilla() -> None:
    a = acc_por_sujeto(cargar("loso_lsa64.json"))
    b = acc_por_sujeto(cargar("loso_lsa64_completo.json"))
    sujetos = sorted(a)
    idx = np.arange(len(sujetos))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(idx - 0.2, [a[s] for s in sujetos], 0.4, color=AZUL, label="Frames sin mano filtrados (oficial)")
    ax.bar(idx + 0.2, [b[s] for s in sujetos], 0.4, color=AZUL_CLARO, label="Grilla completa, relleno con ceros")
    ax.set_xticks(idx, [s.replace("sujeto_", "S") for s in sujetos])
    ax.set_ylim(50, 100)
    ax.set_ylabel("Exactitud (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=2)
    guardar(fig, "fig_filtrado_vs_grilla.png")


def fig_curva_entrenamiento() -> None:
    meta = json.loads((RAIZ / "backend" / "models" / "modelo_lsa64_lstm.json").read_text(encoding="utf-8"))
    hist = meta["history"]
    ep = [h["epoch"] for h in hist]
    loss = [h["loss"] for h in hist]
    val = [h["val_accuracy"] * 100 for h in hist]
    fig, ax1 = plt.subplots(figsize=(7.0, 3.6))
    ax1.plot(ep, loss, color=AZUL, marker="o", ms=3, label="Pérdida de entrenamiento")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida (entropía cruzada)", color=AZUL)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(ep, val, color=NARANJA, marker="s", ms=3, label="Exactitud de validación")
    ax2.set_ylabel("Exactitud de validación (%)", color=NARANJA)
    best = meta["best_epoch"]
    ax1.axvline(best, color=GRIS, ls=":", lw=1)
    ax1.text(best + 0.3, max(loss) * 0.9, f"mejor época ({best})", fontsize=8, color=GRIS)
    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, frameon=False, fontsize=8, loc="center right")
    guardar(fig, "fig_curva_entrenamiento.png")


def fig_accuracy_por_clase() -> None:
    rep = cargar("loso_lsa64.json")
    nombres = glosas()
    items = sorted(rep["accuracy_por_clase"].items(), key=lambda kv: kv[1])
    etiquetas = [nombres.get(k, k) for k, _ in items]
    valores = [v * 100 for _, v in items]
    fig, ax = plt.subplots(figsize=(7.2, 9.6))
    colores = [ROJO if v < 60 else (NARANJA if v < 80 else AZUL) for v in valores]
    ax.barh(range(len(valores)), valores, color=colores, height=0.75)
    ax.set_yticks(range(len(valores)), etiquetas, fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exactitud LOSO agrupada (%)")
    ax.axvline(rep["accuracy_global_agrupada"] * 100, color=GRIS, ls="--", lw=1)
    ax.margins(y=0.005)
    guardar(fig, "fig_accuracy_por_clase.png")


def matriz_confusion(rep: dict) -> np.ndarray:
    n = len(rep["labels"])
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(rep["predicciones"]["y_true"], rep["predicciones"]["y_pred"]):
        m[t, p] += 1
    return m


def fig_matriz_confusion() -> list[tuple[str, str, int]]:
    rep = cargar("loso_lsa64.json")
    m = matriz_confusion(rep)
    filas = m.sum(axis=1, keepdims=True)
    norm = m / np.maximum(filas, 1)
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ticks = list(range(0, 64, 4))
    ax.set_xticks(ticks, [f"{i + 1:02d}" for i in ticks], fontsize=7)
    ax.set_yticks(ticks, [f"{i + 1:02d}" for i in ticks], fontsize=7)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proporción de la fila")
    guardar(fig, "fig_matriz_confusion.png")

    # Pares más confundidos (fuera de la diagonal), para la tabla del capítulo.
    nombres = glosas()
    pares = []
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i != j and m[i, j] > 0:
                pares.append((nombres[rep["labels"][i]], nombres[rep["labels"][j]], int(m[i, j]), int(filas[i, 0])))
    pares.sort(key=lambda t: -t[2])
    return pares[:12]


def fig_screening() -> None:
    nombres = {
        "_screen_base.json": "Baseline",
        "_screen_do40.json": "Dropout 0,4",
        "_screen_fd30.json": "Frame-drop 0,30",
        "_screen_v2.json": "Features v2\n(posición)",
    }
    vals = [cargar(k)["accuracy_media"] * 100 for k in nombres]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colores = [GRIS, VERDE, VERDE, ROJO]
    barras = ax.bar(list(nombres.values()), vals, color=colores)
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f} %", ha="center", fontsize=8)
    ax.set_ylim(55, 70)
    ax.set_ylabel("Exactitud media S03 y S08 (%)")
    guardar(fig, "fig_screening.png")


# ---------------------------------------------------------------------------
# Diagramas esquemáticos
# ---------------------------------------------------------------------------

def caja(ax, x, y, w, h, texto, color=AZUL_CLARO, borde=AZUL, fs=9, negrita=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=borde, lw=1.2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs,
            fontweight="bold" if negrita else "normal", wrap=True)


def flecha(ax, x1, y1, x2, y2, texto=None, color=GRIS, fs=8, curva=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.2, connectionstyle=f"arc3,rad={curva}"))
    if texto:
        ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, texto, fontsize=fs, color=GRIS, va="center")


def lienzo(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def fig_arquitectura() -> None:
    fig, ax = lienzo(7.4, 6.2, (0, 10), (0, 10))
    # Cliente
    ax.add_patch(Rectangle((0.2, 7.3), 9.6, 2.5, fc="#F3F6FA", ec=AZUL, lw=1))
    ax.text(0.4, 9.55, "CLIENTE — navegador web (HTML + CSS + JavaScript)", fontsize=9, fontweight="bold", color=AZUL)
    caja(ax, 0.5, 7.6, 2.6, 1.5, "Cámara\n(getUserMedia)")
    caja(ax, 3.7, 7.6, 2.6, 1.5, "Canvas espejado\n→ JPEG 480 px")
    caja(ax, 6.9, 7.6, 2.6, 1.5, "UI: overlay,\ntexto traducido,\nmétricas")
    flecha(ax, 3.1, 8.35, 3.7, 8.35)
    # Backend
    ax.add_patch(Rectangle((0.2, 2.6), 9.6, 4.2, fc="#F7F3EE", ec=NARANJA, lw=1))
    ax.text(0.5, 3.6, "BACKEND\nFastAPI / Uvicorn\n(Python)", fontsize=9, fontweight="bold", color=NARANJA, va="center")
    caja(ax, 0.5, 5.0, 2.8, 1.2, "WebSocketHandler\n(una sesión por conexión)", "#F2DCC4", NARANJA)
    caja(ax, 3.8, 5.0, 2.6, 1.2, "SesionTraduccion\nsuavizado + ventana", "#F2DCC4", NARANJA)
    caja(ax, 6.9, 5.0, 2.6, 1.2, "TraductorService\n(sin estado)", "#F2DCC4", NARANJA)
    caja(ax, 3.8, 2.9, 2.6, 1.3, "Preprocesador\nMediaPipe Hands", "#F2DCC4", NARANJA)
    caja(ax, 6.9, 2.9, 2.6, 1.3, "ModeloLSE\n+ adaptador .pt/.joblib", "#F2DCC4", NARANJA)
    flecha(ax, 3.3, 5.6, 3.8, 5.6)
    flecha(ax, 6.4, 5.6, 6.9, 5.6)
    flecha(ax, 7.6, 5.0, 5.4, 4.2)
    flecha(ax, 8.2, 5.0, 8.2, 4.2)
    # Comunicación
    flecha(ax, 5.0, 7.6, 1.9, 6.2, "frames JPEG\n(binario)", curva=0.1)
    flecha(ax, 2.4, 6.2, 8.0, 7.6, "JSON: seña, confianza,\nestabilidad, landmarks", curva=0.15)
    # Común y ML
    caja(ax, 0.5, 0.4, 4.2, 1.6, "common/\nfeatures.py · models/lsa64.py\n(código compartido)", "#E3EFE6", VERDE)
    caja(ax, 5.3, 0.4, 4.2, 1.6, "ml/ (offline)\nextracción → caché → BiLSTM\n→ modelo_lsa64_lstm.pt", "#E3EFE6", VERDE)
    flecha(ax, 2.6, 2.0, 4.6, 2.9)
    flecha(ax, 7.4, 2.0, 8.0, 2.9, "carga al\narrancar")
    flecha(ax, 4.7, 1.2, 5.3, 1.2)
    guardar(fig, "fig_arquitectura.png")


def fig_secuencia() -> None:
    actores = ["Usuario", "Navegador", "WebSocket\nHandler", "Sesion\nTraduccion", "Preproce-\nsador", "ModeloLSE"]
    xs = np.linspace(0.8, 9.2, len(actores))
    fig, ax = lienzo(7.6, 7.0, (0, 10), (0, 13))
    for x, a in zip(xs, actores):
        caja(ax, x - 0.65, 11.9, 1.3, 0.9, a, fs=7.5)
        ax.plot([x, x], [0.3, 11.9], color=GRIS, lw=0.8, ls="--")
    mensajes = [
        (0, 1, "inicia cámara", 11.3),
        (1, 2, "conectar() — /ws/translate", 10.6),
        (2, 3, "nueva_sesion()", 10.0),
        (1, 2, "frame JPEG (cada 1/fps s)", 9.0),
        (2, 3, "traducir(bytes) [threadpool]", 8.4),
        (3, 4, "extraer_mano(frame)", 7.8),
        (4, 3, "vector 138 + landmarks", 7.2),
        (3, 5, "predecir_secuencia(ventana)", 6.4),
        (5, 3, "Prediccion{etiqueta, p}", 5.8),
        (3, 3, "voto mayoritario", 5.2),
        (3, 2, "dict resultado + ms_servidor", 4.4),
        (2, 1, "JSON", 3.8),
        (1, 0, "texto + overlay", 3.2),
        (0, 1, "detener", 2.0),
        (1, 2, "cerrar()", 1.4),
    ]
    for o, d, t, y in mensajes:
        if o == d:
            ax.add_patch(FancyArrowPatch((xs[o], y + 0.2), (xs[o], y - 0.25), arrowstyle="-|>",
                                         mutation_scale=10, color=AZUL, connectionstyle="arc3,rad=-1.2"))
            ax.text(xs[o] + 0.35, y, t, fontsize=7.5, va="center")
            continue
        ax.add_patch(FancyArrowPatch((xs[o], y), (xs[d], y), arrowstyle="-|>", mutation_scale=10,
                                     color=AZUL if d > o else NARANJA, lw=1.1,
                                     linestyle="-" if d > o else "--"))
        ax.text((xs[o] + xs[d]) / 2, y + 0.15, t, ha="center", fontsize=7.5)
    ax.add_patch(Rectangle((0.2, 3.0), 9.6, 6.4, fill=False, ec=GRIS, lw=0.8))
    ax.text(0.3, 9.15, "loop", fontsize=8, fontweight="bold", color=GRIS)
    guardar(fig, "fig_secuencia.png")


def fig_vector_features() -> None:
    fig, ax = lienzo(7.6, 2.6, (0, 138), (0, 4))
    segmentos = [
        (0, 1, "P", NARANJA), (1, 64, "63 coordenadas normalizadas\n(21 puntos × x, y, z)", AZUL_CLARO),
        (64, 69, "5 ángulos", VERDE),
        (69, 70, "P", NARANJA), (70, 133, "63 coordenadas normalizadas\n(21 puntos × x, y, z)", AZUL_CLARO),
        (133, 138, "5 ángulos", VERDE),
    ]
    for a, b, t, c in segmentos:
        ax.add_patch(Rectangle((a, 1.2), b - a, 1.4, fc=c, ec="white", lw=1))
        ax.text((a + b) / 2, 1.9, t, ha="center", va="center", fontsize=7 if b - a > 3 else 6)
    ax.annotate("", xy=(0, 3.1), xytext=(69, 3.1), arrowprops=dict(arrowstyle="<->", color=GRIS))
    ax.text(34.5, 3.4, "Mano izquierda — 69 valores", ha="center", fontsize=8)
    ax.annotate("", xy=(69, 3.1), xytext=(138, 3.1), arrowprops=dict(arrowstyle="<->", color=GRIS))
    ax.text(103.5, 3.4, "Mano derecha — 69 valores", ha="center", fontsize=8)
    for pos in (0, 1, 64, 69, 70, 133, 138):
        ax.text(pos, 0.8, str(pos), ha="center", fontsize=6.5, color=GRIS)
    ax.text(0, 0.1, "P = indicador de presencia de la mano (0 / 1)", fontsize=7.5, color=GRIS)
    guardar(fig, "fig_vector_features.png")


# Coordenadas aproximadas de una mano abierta (palma hacia la cámara), solo
# para ilustrar la numeración de los 21 puntos de MediaPipe.
MANO = np.array([
    [0.50, 0.05], [0.36, 0.14], [0.26, 0.26], [0.18, 0.37], [0.11, 0.46],
    [0.38, 0.45], [0.36, 0.60], [0.35, 0.70], [0.34, 0.79],
    [0.49, 0.47], [0.49, 0.64], [0.49, 0.75], [0.49, 0.85],
    [0.60, 0.45], [0.62, 0.60], [0.63, 0.70], [0.64, 0.78],
    [0.70, 0.40], [0.75, 0.51], [0.78, 0.59], [0.81, 0.66],
])


def fig_landmarks_mano() -> None:
    import sys
    sys.path.insert(0, str(RAIZ))
    from common.features import HAND_CONNECTIONS
    nombres = {0: "Muñeca", 4: "Punta\npulgar", 8: "Punta\níndice", 12: "Punta\nmedio",
               16: "Punta\nanular", 20: "Punta\nmeñique"}
    fig, ax = lienzo(4.6, 4.8, (0, 1), (-0.05, 0.95))
    for a, b in HAND_CONNECTIONS:
        ax.plot(MANO[[a, b], 0], MANO[[a, b], 1], color=AZUL, lw=2)
    ax.scatter(MANO[:, 0], MANO[:, 1], s=60, color=NARANJA, zorder=3, ec="white")
    for i, (x, y) in enumerate(MANO):
        ax.text(x + 0.018, y - 0.012, str(i), fontsize=7.5)
        if i in nombres:
            ax.text(x, y + 0.045, nombres[i], fontsize=6.5, ha="center", color=GRIS)
    guardar(fig, "fig_landmarks_mano.png")


def fig_pipeline_dos_etapas() -> None:
    fig, ax = lienzo(7.6, 2.8, (0, 12), (0, 4))
    pasos = [
        (0.1, "3200 videos\nLSA64\n(1080p, 60 fps)", "#EDEDED", GRIS),
        (2.5, "MediaPipe Hands\nen paralelo\n(8 procesos)", AZUL_CLARO, AZUL),
        (4.9, "Caché .npz\nsecuencias 138-d\n+ metadatos", "#E3EFE6", VERDE),
        (7.3, "Filtrado de frames\ny secuencias cortas", AZUL_CLARO, AZUL),
        (9.7, "BiLSTM\nentrenamiento /\nLOSO", "#F2DCC4", NARANJA),
    ]
    for x, t, c, b in pasos:
        caja(ax, x, 1.2, 2.1, 1.7, t, c, b, fs=8)
    for i in range(len(pasos) - 1):
        flecha(ax, pasos[i][0] + 2.1, 2.05, pasos[i + 1][0], 2.05)
    ax.annotate("", xy=(0.1, 0.7), xytext=(7.0, 0.7), arrowprops=dict(arrowstyle="<->", color=GRIS))
    ax.text(3.5, 0.3, "Etapa 1 — se paga una vez (~47 min)", ha="center", fontsize=8, color=GRIS)
    ax.annotate("", xy=(7.3, 0.7), xytext=(11.8, 0.7), arrowprops=dict(arrowstyle="<->", color=GRIS))
    ax.text(9.55, 0.3, "Etapa 2 — se repite (segundos a minutos)", ha="center", fontsize=8, color=GRIS)
    ax.text(5.95, 3.3, "invalida si cambian versión, espejado,\nframe_step o max_frames", fontsize=7, ha="center", color=VERDE)
    guardar(fig, "fig_pipeline_dos_etapas.png")


def fig_bilstm() -> None:
    fig, ax = lienzo(7.4, 4.4, (0, 12), (0, 7.2))
    T = ["x₁", "x₂", "x₃", "…", "x_T"]
    xs = [1.5, 3.5, 5.5, 7.5, 9.5]
    for x, t in zip(xs, T):
        caja(ax, x - 0.6, 0.3, 1.2, 0.8, t, "#EDEDED", GRIS, fs=9)
        if t != "…":
            caja(ax, x - 0.75, 1.8, 1.5, 0.8, "LSTM →", AZUL_CLARO, AZUL, fs=8)
            caja(ax, x - 0.75, 3.0, 1.5, 0.8, "← LSTM", "#F2DCC4", NARANJA, fs=8)
            flecha(ax, x, 1.1, x, 1.8)
    for i in range(len(xs) - 1):
        flecha(ax, xs[i] + 0.75, 2.2, xs[i + 1] - 0.75, 2.2, color=AZUL)
        flecha(ax, xs[i + 1] - 0.75, 3.4, xs[i] + 0.75, 3.4, color=NARANJA)
    caja(ax, 10.6, 2.6, 1.3, 1.6, "concat\n[h→T ; h←1]\n256", "#E3EFE6", VERDE, fs=7.5)
    flecha(ax, 10.25, 2.2, 10.6, 3.0, color=AZUL)
    flecha(ax, 2.0, 3.8, 10.8, 4.2, color=NARANJA, curva=-0.25)
    caja(ax, 10.6, 4.8, 1.3, 0.8, "Dropout", "#EDEDED", GRIS, fs=8)
    caja(ax, 10.6, 6.1, 1.3, 0.8, "Lineal\n→ 64", "#EDEDED", GRIS, fs=8)
    flecha(ax, 11.25, 4.2, 11.25, 4.8)
    flecha(ax, 11.25, 5.6, 11.25, 6.1)
    ax.text(0.2, 6.6, "Entrada: secuencia de T vectores de 138 features (T variable, 8 ≤ T ≤ 120)", fontsize=8, color=GRIS)
    guardar(fig, "fig_bilstm.png")


def fig_celda_lstm() -> None:
    fig, ax = lienzo(7.0, 3.6, (0, 10), (0, 5))
    ax.add_patch(Rectangle((1.0, 0.6), 8.0, 3.6, fc="#F3F6FA", ec=AZUL, lw=1.2))
    ax.plot([0.2, 9.8], [3.7, 3.7], color=GRIS, lw=2)
    ax.text(0.2, 3.85, "c_{t-1}", fontsize=8)
    ax.text(9.3, 3.85, "c_t", fontsize=8)
    ax.plot([0.2, 1.0], [1.0, 1.0], color=GRIS, lw=1.5)
    ax.text(0.1, 0.65, "h_{t-1}", fontsize=8)
    puertas = [(2.0, "f_t\nolvido\nσ"), (3.8, "i_t\nentrada\nσ"), (5.4, "ĉ_t\ncandidato\ntanh"), (7.2, "o_t\nsalida\nσ")]
    for x, t in puertas:
        caja(ax, x - 0.6, 1.2, 1.2, 1.2, t, "#F2DCC4", NARANJA, fs=7)
    # f_t multiplica la memoria anterior
    flecha(ax, 2.0, 2.4, 2.0, 3.5)
    ax.text(2.0, 3.62, "⊗", ha="center", va="center", fontsize=13)
    # i_t ⊗ ĉ_t se suma a la memoria
    flecha(ax, 3.8, 2.4, 4.5, 2.95)
    flecha(ax, 5.4, 2.4, 4.7, 2.95)
    ax.text(4.6, 3.05, "⊗", ha="center", va="center", fontsize=13)
    flecha(ax, 4.6, 3.2, 4.6, 3.5)
    ax.text(4.6, 3.62, "⊕", ha="center", va="center", fontsize=13)
    # salida: o_t ⊗ tanh(c_t)
    flecha(ax, 7.9, 3.7, 7.9, 2.95)
    ax.text(7.9, 2.8, "tanh", ha="center", fontsize=7)
    flecha(ax, 7.2, 2.4, 7.75, 2.65)
    ax.text(8.15, 2.2, "⊗ → h_t", fontsize=9)
    ax.plot([1.0, 7.2], [1.0, 1.0], color=GRIS, lw=1, ls=":")
    ax.text(5.0, 0.2, "x_t (vector de 138 features del frame t)", fontsize=8, ha="center", color=GRIS)
    flecha(ax, 5.0, 0.4, 5.0, 1.0)
    guardar(fig, "fig_celda_lstm.png")


def fig_protocolos() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    rng = np.random.RandomState(3)
    for ax, titulo, modo in zip(axes, ["Signer-dependent (k-fold aleatorio)", "Signer-independent (LOSO)"], ["dep", "ind"]):
        grilla = np.zeros((10, 32))
        if modo == "dep":
            grilla[rng.rand(10, 32) < 0.2] = 1
        else:
            grilla[2, :] = 1
        ax.imshow(grilla, cmap=matplotlib.colors.ListedColormap([AZUL_CLARO, NARANJA]), aspect="auto")
        ax.set_yticks(range(10), [f"S{i + 1:02d}" for i in range(10)], fontsize=7)
        ax.set_xticks([])
        ax.set_xlabel("Videos de cada sujeto", fontsize=8)
        ax.set_title(titulo, fontsize=9)
    fig.text(0.5, -0.02, "Celeste: entrenamiento · Naranja: test", ha="center", fontsize=8, color=GRIS)
    guardar(fig, "fig_protocolos.png")


def fig_gantt() -> None:
    tareas = [
        ("Anteproyecto y relevamiento", "2026-03-01", "2026-05-15", GRIS),
        ("Prototipo de escritorio (análisis)", "2026-05-01", "2026-06-15", GRIS),
        ("Arquitectura base y common/", "2026-06-15", "2026-07-10", AZUL),
        ("Backend FastAPI + WebSocket", "2026-07-01", "2026-08-10", AZUL),
        ("Frontend web y accesibilidad", "2026-07-10", "2026-08-14", AZUL),
        ("Pipeline LSA64 y correcciones", "2026-08-01", "2026-08-14", NARANJA),
        ("Validación LOSO y experimentos", "2026-08-10", "2026-09-01", NARANJA),
        ("Glosas, integración y modelo servido", "2026-08-25", "2026-09-05", NARANJA),
        ("Captura y evaluación del abecedario", "2026-11-01", "2026-11-30", ROJO),
        ("Evaluación en tiempo real y baseline", "2026-11-01", "2026-11-25", ROJO),
        ("Deploy en capa gratuita", "2026-11-05", "2026-11-25", ROJO),
        ("Pruebas con usuarios", "2026-11-20", "2026-12-05", ROJO),
        ("Redacción y revisión del documento", "2026-09-28", "2026-12-10", ROJO),
    ]
    import datetime as dt
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, (t, a, b, c) in enumerate(reversed(tareas)):
        ia = dt.date.fromisoformat(a)
        ib = dt.date.fromisoformat(b)
        ax.barh(i, (ib - ia).days, left=matplotlib.dates.date2num(ia), color=c, height=0.6)
    ax.set_yticks(range(len(tareas)), [t for t, *_ in reversed(tareas)], fontsize=7.5)
    ax.xaxis_date()
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: meses[matplotlib.dates.num2date(v).month - 1]))
    hoy = matplotlib.dates.date2num(dt.date(2026, 10, 21))
    ax.axvline(hoy, color=GRIS, ls="--", lw=1)
    ax.text(hoy + 2, len(tareas) - 0.6, "21/10", fontsize=7, color=GRIS)
    guardar(fig, "fig_gantt.png")


def main() -> None:
    print("Generando figuras en", SALIDA)
    fig_loso_por_sujeto()
    fig_retencion_vs_accuracy()
    fig_filtrado_vs_grilla()
    fig_curva_entrenamiento()
    fig_accuracy_por_clase()
    pares = fig_matriz_confusion()
    fig_screening()
    fig_arquitectura()
    fig_secuencia()
    fig_vector_features()
    fig_landmarks_mano()
    fig_pipeline_dos_etapas()
    fig_bilstm()
    fig_celda_lstm()
    fig_protocolos()
    fig_gantt()
    print("\nPares más confundidos (real → predicha: n / total de la clase):")
    for a, b, n, tot in pares:
        print(f"  {a} → {b}: {n} / {tot}")


if __name__ == "__main__":
    main()
