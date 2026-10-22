# -*- coding: utf-8 -*-
"""
docs/anteproyecto/figuras/generar_figuras.py

Figuras del anteproyecto corregido: el diagrama de Gantt (las mismas tareas y
semanas que el aprobado, redibujado para que se lean los nombres) y el boceto
de pantallas del Anexo C.

Correr desde la raíz del repo, con el entorno de ML:
    .venv-ml\\Scripts\\python docs/anteproyecto/figuras/generar_figuras.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

SALIDA = Path(__file__).resolve().parent

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "savefig.dpi": 220})

LEANDRO = "#185FA5"
JONATAN = "#0F6E56"
AMBOS = "#8A80D8"
HITO = "#333333"

# Semanas contadas desde la primera de mayo de 2026 (0) hasta la última de
# octubre (23). Inicio inclusive, fin exclusivo. Son las del Gantt aprobado.
FASES = [
    ("Fase 1 — Investigación y diseño", [
        ("Relevamiento bibliográfico y estado del arte", "L+J", 0, 2),
        ("Definición de la arquitectura del sistema", "L+J", 1, 4),
        ("Diseño de la arquitectura de la red neuronal", "Leandro", 2, 6),
        ("Diseño de la arquitectura de la aplicación web", "Jonatan", 2, 6),
    ]),
    ("Fase 2 — Datos y entrenamiento", [
        ("Recopilación y curación del dataset LSA64", "L+J", 5, 7),
        ("Preprocesamiento y aumentación de datos", "Leandro", 6, 10),
        ("Entrenamiento de la red neuronal (v1)", "Leandro", 9, 11),
        ("Evaluación y ajuste del modelo", "Leandro", 12, 15),
    ]),
    ("Fase 3 — Desarrollo de la aplicación web", [
        ("Configuración del entorno y estructura del proyecto", "Jonatan", 5, 7),
        ("Captura de video y detección de manos (MediaPipe)", "Jonatan", 7, 11),
        ("Integración del modelo con el frontend (API)", "L+J", 12, 16),
        ("Interfaz de usuario y experiencia de uso", "Jonatan", 14, 18),
    ]),
    ("Fase 4 — Integración y pruebas", [
        ("Pruebas funcionales de extremo a extremo", "L+J", 16, 20),
        ("Ajustes de rendimiento y precisión", "Leandro", 18, 22),
        ("Despliegue en capa gratuita (Render / Hugging Face)", "Jonatan", 18, 22),
    ]),
    ("Fase 5 — Documentación y cierre", [
        ("Documentación técnica", "L+J", 20, 23),
        ("Redacción del informe final de tesina", "L+J", 21, 24),
        ("Revisión y entrega final", "L+J", 23, 24),
    ]),
]

# Actividades de evaluación académica (la guía pide incluirlas en el cronograma).
HITOS = [
    ("Revisión de avance con el profesor tutor", [4, 8, 12, 16, 20]),
    ("Entrega del informe final de tesina", [24]),
]

COLOR = {"Leandro": LEANDRO, "Jonatan": JONATAN, "L+J": AMBOS}
MESES = ["May", "Jun", "Jul", "Ago", "Sep", "Oct"]


def fig_gantt() -> None:
    filas = []  # (texto, tipo, datos)
    for fase, tareas in FASES:
        filas.append((fase, "fase", None))
        filas.extend((t, "tarea", (r, a, b)) for t, r, a, b in tareas)
    filas.append(("Fase 6 — Seguimiento académico", "fase", None))
    filas.extend((t, "hito", s) for t, s in HITOS)

    alto = 0.27 * len(filas) + 0.9
    fig, ax = plt.subplots(figsize=(7.6, alto))
    n = len(filas)
    for i, (texto, tipo, datos) in enumerate(filas):
        y = n - 1 - i
        if tipo == "fase":
            ax.add_patch(Rectangle((-15.5, y - 0.45), 39.5, 0.9, color="#2B2B2B", clip_on=False, zorder=0))
            ax.text(-15.3, y, texto.upper(), va="center", ha="left", color="white", fontsize=7.5, fontweight="bold")
            continue
        ax.text(-15.3, y, texto, va="center", ha="left", fontsize=7.6, color="#1A1A1A")
        if tipo == "tarea":
            resp, a, b = datos
            ax.barh(y, b - a, left=a, height=0.55, color=COLOR[resp], zorder=3)
        else:
            ax.scatter([s - 0.5 for s in datos], [y] * len(datos), marker="D", s=26, color=HITO, zorder=3)
    for m in range(7):
        ax.axvline(m * 4, color="#D0D0D0", lw=0.7, zorder=1)
    for m, nombre in enumerate(MESES):
        ax.text(m * 4 + 2, n - 0.2 + 0.6, nombre, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xlim(-15.5, 24)
    ax.set_ylim(-0.7, n + 0.6)
    ax.axis("off")
    leyenda = [Rectangle((0, 0), 1, 1, color=c) for c in (LEANDRO, JONATAN, AMBOS)]
    leyenda.append(plt.Line2D([], [], marker="D", ls="", color=HITO))
    ax.legend(leyenda, ["Leandro", "Jonatan", "Ambos", "Hito académico"], loc="upper center",
              bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False, fontsize=7.5)
    fig.savefig(SALIDA / "gantt.png", bbox_inches="tight")
    plt.close(fig)


def _mano(ax, cx, cy, escala):
    """Esqueleto esquemático de una mano: muñeca + 5 dedos de 4 puntos."""
    dedos = [(-0.9, 0.3), (-0.45, 1.0), (0.0, 1.15), (0.45, 1.0), (0.85, 0.7)]
    for dx, dy in dedos:
        xs = [cx + dx * escala * k / 3 for k in range(4)]
        ys = [cy + dy * escala * k / 3 + 0.25 * escala * (k > 0) for k in range(4)]
        ax.plot([cx] + xs, [cy] + ys, color="#4C8BD6", lw=1.2)
        ax.scatter(xs, ys, s=7, color="#D64C4C", zorder=4)
    ax.scatter([cx], [cy], s=9, color="#D64C4C", zorder=4)


def _caja(ax, x, y, w, h, texto="", relleno="#F2F2F2", borde="#7A7A7A", **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=relleno, ec=borde, lw=0.9))
    if texto:
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", **kw)


def fig_wireframe() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 4.1), gridspec_kw={"width_ratios": [2.35, 1]})

    # Computadora
    ax = axs[0]
    _caja(ax, 0, 0, 16, 10, relleno="white", borde="#555555")
    ax.add_patch(Rectangle((0, 9.3), 16, 0.7, fc="#E4E4E4", ec="#555555", lw=0.9))
    for k, c in enumerate(("#E06C5F", "#E5B94E", "#6DBA6A")):
        ax.add_patch(Circle((0.45 + 0.4 * k, 9.65), 0.12, color=c))
    ax.text(8, 9.65, "Traductor LSA", ha="center", va="center", fontsize=7.5)
    _caja(ax, 0.5, 2.0, 9.5, 6.9, relleno="#EDEDED")
    ax.text(5.25, 8.5, "Video de la cámara", ha="center", fontsize=7, color="#555555")
    _mano(ax, 5.2, 3.7, 2.2)
    _caja(ax, 10.5, 6.3, 5.0, 2.6, relleno="white")
    ax.text(13, 8.4, "Seña reconocida", ha="center", fontsize=7, color="#555555")
    ax.text(13, 7.3, "AGUA", ha="center", fontsize=13, fontweight="bold")
    ax.text(13, 6.6, "confianza 0,87", ha="center", fontsize=6.5, color="#555555")
    _caja(ax, 10.5, 2.0, 5.0, 3.9, relleno="white")
    ax.text(13, 5.4, "Texto traducido", ha="center", fontsize=7, color="#555555")
    for k, linea in enumerate(("hola · agua ·", "gracias", "")):
        ax.text(10.9, 4.7 - 0.55 * k, linea, fontsize=7.5, va="center")
    _caja(ax, 0.5, 0.5, 3.0, 1.0, "Iniciar cámara", relleno="#185FA5", borde="#185FA5", color="white", fontsize=7)
    _caja(ax, 3.8, 0.5, 2.6, 1.0, "Detener", fontsize=7)
    _caja(ax, 10.5, 0.5, 2.4, 1.0, "Borrar", fontsize=7)
    _caja(ax, 13.1, 0.5, 2.4, 1.0, "Copiar", fontsize=7)
    ax.set_xlim(-0.2, 16.2)
    ax.set_ylim(-0.2, 10.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("a) Computadora", fontsize=8.5, y=-0.07)

    # Teléfono
    ax = axs[1]
    _caja(ax, 0, 0, 6, 11, relleno="white", borde="#555555")
    ax.text(3, 10.4, "Traductor LSA", ha="center", fontsize=7.5)
    _caja(ax, 0.4, 5.3, 5.2, 4.6, relleno="#EDEDED")
    _mano(ax, 3.0, 6.3, 1.5)
    _caja(ax, 0.4, 3.6, 5.2, 1.4, relleno="white")
    ax.text(3, 4.3, "AGUA", ha="center", va="center", fontsize=11, fontweight="bold")
    _caja(ax, 0.4, 1.6, 5.2, 1.7, relleno="white")
    ax.text(0.7, 2.45, "hola · agua · gracias", fontsize=7, va="center")
    _caja(ax, 0.4, 0.4, 2.5, 0.9, "Iniciar", relleno="#185FA5", borde="#185FA5", color="white", fontsize=7)
    _caja(ax, 3.1, 0.4, 2.5, 0.9, "Borrar", fontsize=7)
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-0.2, 11.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("b) Teléfono", fontsize=8.5, y=-0.07)

    fig.savefig(SALIDA / "wireframe.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_gantt()
    fig_wireframe()
    print("Figuras generadas en", SALIDA)
