# -*- coding: utf-8 -*-
"""
docs/anteproyecto/figuras/generar_figuras_anteproyecto.py

Figuras propias del anteproyecto v3. Reutiliza los helpers de estilo del script
de la tesina (docs/tesina/figuras/generar_figuras.py) para que los dos
documentos se vean iguales, y agrega tres figuras que la tesina todavía no
tiene:

  fig_loso_modelo_servido.png  LOSO del modelo que HOY sirve el backend
                               (regularizado: dropout 0,4 + frame-drop 0,30).
                               La figura de la tesina (fig_loso_por_sujeto.png)
                               sigue mostrando el baseline de 79,8 %; esta sale
                               de ml/reports/loso_do40_fd30.json.
  fig_clases_implementado.png  Diagrama de clases de lo que está construido,
                               no de lo que se propuso (ver
                               docs/CONFORMIDAD_ARQUITECTURA.md §2).
  fig_despliegue.png           Esquema de despliegue: backend local expuesto
                               con ngrok + frontend estático en Vercel.

Correr desde la raíz del repo:
    .venv-ml\\Scripts\\python docs/anteproyecto/figuras/generar_figuras_anteproyecto.py

Ningún número está escrito a mano: todos salen de ml/reports/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
SALIDA = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "docs" / "tesina" / "figuras"))

import generar_figuras as gf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

AZUL, AZUL_CLARO = gf.AZUL, gf.AZUL_CLARO
NARANJA, GRIS, VERDE, ROJO = gf.NARANJA, gf.GRIS, gf.VERDE, gf.ROJO
caja, flecha, lienzo = gf.caja, gf.flecha, gf.lienzo


def guardar(fig, nombre: str) -> None:
    fig.savefig(SALIDA / nombre)
    plt.close(fig)
    print("  ", nombre)


# ---------------------------------------------------------------------------
# 1. Resultado del modelo servido
# ---------------------------------------------------------------------------

def fig_loso_modelo_servido() -> None:
    rep = json.loads((RAIZ / "ml" / "reports" / "loso_do40_fd30.json").read_text(encoding="utf-8"))
    acc = {f["test_subject"]: f["test_accuracy"] * 100 for f in rep["folds"]}
    sujetos = sorted(acc)
    valores = [acc[s] for s in sujetos]
    media = rep["accuracy_media"] * 100
    desvio = rep["accuracy_desvio"] * 100

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colores = [ROJO if v < media - desvio else AZUL for v in valores]
    barras = ax.bar([s.replace("sujeto_", "S") for s in sujetos], valores, color=colores, zorder=2)
    ax.axhline(media, color=NARANJA, lw=1.5, label=f"Media {media:.1f} %".replace(".", ","))
    ax.axhspan(media - desvio, media + desvio, color=NARANJA, alpha=0.12, zorder=0,
               label=f"± 1 desvío ({desvio:.1f} pp)".replace(".", ","))
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}".replace(".", ","),
                ha="center", fontsize=8)
    ax.set_ylim(50, 100)
    ax.set_ylabel("Exactitud en la persona de prueba (%)")
    ax.set_xlabel("Persona dejada afuera (partición leave-one-subject-out)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False, fontsize=8)
    guardar(fig, "fig_loso_modelo_servido.png")


def fig_comparacion_modelos() -> None:
    """BiLSTM regularizada vs. BiLSTM sin regularizar vs. Random Forest."""
    reg = json.loads((RAIZ / "ml" / "reports" / "comparacion_regularizacion.json").read_text(encoding="utf-8"))
    rf = json.loads((RAIZ / "ml" / "reports" / "comparacion_bilstm_rf.json").read_text(encoding="utf-8"))

    etiquetas = ["Random Forest\n(línea base\nno neuronal)", "BiLSTM\nsin regularizar", "BiLSTM regularizada\n(modelo servido)"]
    medias = [rf["media_b"] * 100, reg["media_b"] * 100, reg["media_a"] * 100]
    desvios = [rf["desvio_b"] * 100, reg["desvio_b"] * 100, reg["desvio_a"] * 100]
    colores = [GRIS, AZUL_CLARO, AZUL]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    barras = ax.bar(etiquetas, medias, yerr=desvios, color=colores, capsize=6,
                    error_kw={"ecolor": GRIS, "lw": 1.1}, zorder=2)
    for b, m, d in zip(barras, medias, desvios):
        ax.text(b.get_x() + b.get_width() / 2, m + d + 1.2,
                f"{m:.1f} % ± {d:.1f}".replace(".", ","), ha="center", fontsize=8.5)
    ax.set_ylim(60, 100)
    ax.set_ylabel("Exactitud media LOSO (%)")
    ax.tick_params(axis="x", labelsize=8.5)
    guardar(fig, "fig_comparacion_modelos.png")


# ---------------------------------------------------------------------------
# 2. Diagrama de clases de lo implementado
# ---------------------------------------------------------------------------

def _clase(ax, x, y, w, nombre, miembros, color=AZUL_CLARO, borde=AZUL, nuevo=False):
    """Caja UML: nombre en negrita arriba, miembros debajo."""
    alto_cab = 0.52
    alto_cuerpo = 0.30 * len(miembros) + 0.18
    h = alto_cab + alto_cuerpo
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc="white", ec=borde, lw=1.6 if nuevo else 1.1,
                                ls="--" if nuevo else "-", zorder=3))
    ax.add_patch(Rectangle((x + 0.02, y - alto_cab), w - 0.04, alto_cab - 0.02,
                           fc=color, ec="none", zorder=4))
    ax.text(x + w / 2, y - alto_cab / 2, nombre, ha="center", va="center",
            fontsize=8.5, fontweight="bold", zorder=5)
    for i, m in enumerate(miembros):
        ax.text(x + 0.12, y - alto_cab - 0.26 - i * 0.30, m, ha="left", va="center",
                fontsize=7.2, color="#333333", zorder=5)
    return h


def fig_clases_implementado() -> None:
    fig, ax = lienzo(7.8, 9.6, (0, 10), (0, 14))

    # --- Zonas ---
    ax.add_patch(Rectangle((0.15, 12.05), 9.7, 1.9, fc="#F3F6FA", ec=AZUL, lw=1, zorder=1))
    ax.text(0.32, 13.70, "CLIENTE — navegador (frontend/app.js)", fontsize=8.5,
            fontweight="bold", color=AZUL, zorder=2)

    ax.add_patch(Rectangle((0.15, 3.70), 9.7, 8.10, fc="#F7F3EE", ec=NARANJA, lw=1, zorder=1))
    ax.text(0.32, 11.60, "BACKEND — FastAPI (backend/app/)", fontsize=8.5,
            fontweight="bold", color=NARANJA, zorder=2)

    ax.add_patch(Rectangle((0.15, 0.85), 9.7, 2.45, fc="#EFF6F1", ec=VERDE, lw=1, zorder=1))
    ax.text(0.32, 3.08, "CÓDIGO COMPARTIDO con el pipeline de entrenamiento ml/  (common/)",
            fontsize=8.5, fontweight="bold", color=VERDE, zorder=2)

    # --- Cliente ---
    _clase(ax, 0.45, 13.40, 4.3, "CamaraCaptura  «navegador»",
           ["stream : MediaStream", "iniciar() · capturarFrame() · detener()"])
    _clase(ax, 5.25, 13.40, 4.3, "ClienteWS  «navegador»",
           ["enviarFrame() · dibujarOverlay()", "registrarMetricas()"])

    # --- Backend: fila 1 ---
    _clase(ax, 0.45, 11.30, 4.3, "WebSocketHandler",
           ["conexiones : dict[WS, SesionTraduccion]",
            "conectar() · recibir_frame()", "enviar_resultado()"])
    _clase(ax, 5.25, 11.30, 4.3, "SesionTraduccion   (NUEVA)",
           ["_buffer · _secuencia · _historial",
            "traducir(bytes) : dict", "_suavizar() · historial()"],
           color="#F7DCC0", borde=ROJO, nuevo=True)

    # --- Backend: fila 2 ---
    _clase(ax, 2.85, 9.40, 4.3, "TraductorService  (sin estado)",
           ["preprocesador · modelo", "procesar(frame, secuencia)", "nueva_sesion()"],
           color="#F2DCC4", borde=NARANJA)

    # --- Backend: fila 3 ---
    _clase(ax, 0.35, 7.50, 2.9, "Frame",
           ["datos : bytes", "timestamp : float", "a_array()", "dimensiones() · serializar()"],
           color="#F2DCC4", borde=NARANJA)
    _clase(ax, 3.55, 7.50, 2.9, "Preprocesador",
           ["detector · tam_entrada", "extraer_mano() · normalizar()", "obtener_landmarks()"],
           color="#F2DCC4", borde=NARANJA)
    _clase(ax, 6.75, 7.50, 2.9, "ModeloLSE",
           ["ruta_modelo · clases", "cargar() · predecir()", "predecir_secuencia()", "top_n()"],
           color="#F2DCC4", borde=NARANJA)

    # --- Backend: fila 4 ---
    _clase(ax, 0.35, 5.30, 3.1, "Prediccion",
           ["etiqueta · confianza", "estabilidad   (NUEVO)", "es_valida() · a_dict()"],
           color="#F2DCC4", borde=NARANJA)
    _clase(ax, 3.90, 5.30, 5.75, "ModeloAdaptador  «interfaz»   (NUEVA)",
           ["TorchCheckpointAdapter (.pt)   ·   SklearnJoblibAdapter (.joblib)",
            "cargar() · predict_proba() · requiere_secuencia"],
           color="#F7DCC0", borde=ROJO, nuevo=True)

    # --- Común ---
    _clase(ax, 0.45, 2.85, 4.3, "common.features",
           ["FEATURE_VERSION · ESPEJADO_CANONICO", "build_feature_vector() → 138 valores"],
           color="#D9EBDF", borde=VERDE)
    _clase(ax, 5.25, 2.85, 4.3, "common.models.lsa64",
           ["BiLSTMClassifier", "(misma definición en ml/ y en backend/)"],
           color="#D9EBDF", borde=VERDE)

    # --- Relaciones ---
    flecha(ax, 4.75, 10.50, 5.25, 10.50)               # Handler -> Sesion
    flecha(ax, 7.40, 9.70, 6.20, 9.40)                 # Sesion -> Service
    flecha(ax, 3.60, 7.80, 1.80, 7.50)                 # Service -> Frame
    flecha(ax, 5.00, 7.80, 5.00, 7.50)                 # Service -> Preprocesador
    flecha(ax, 6.40, 7.80, 8.20, 7.50)                 # Service -> ModeloLSE
    flecha(ax, 7.60, 5.60, 7.00, 5.30)                 # ModeloLSE -> Adaptador
    flecha(ax, 7.00, 5.60, 2.20, 5.30, curva=0.12)     # ModeloLSE -> Prediccion
    flecha(ax, 2.60, 3.70, 2.60, 2.85, color=VERDE)    # backend -> common.features
    flecha(ax, 7.40, 3.70, 7.40, 2.85, color=VERDE)    # backend -> common.models

    ax.text(0.32, 0.35,
            "Borde punteado: clase o módulo que no existía en el diagrama de clases original.",
            fontsize=7.5, color=GRIS)
    guardar(fig, "fig_clases_implementado.png")


# ---------------------------------------------------------------------------
# 3. Diagrama de despliegue (ngrok + Vercel)
# ---------------------------------------------------------------------------

def fig_despliegue() -> None:
    fig, ax = lienzo(7.8, 6.4, (0, 10), (0, 8.2))

    # --- Usuario (arriba a la izquierda) ---
    ax.add_patch(Rectangle((0.15, 4.95), 2.80, 3.05, fc="#F3F6FA", ec=AZUL, lw=1, zorder=1))
    ax.text(0.33, 7.73, "USUARIO", fontsize=8.5, fontweight="bold", color=AZUL, zorder=2)
    caja(ax, 0.40, 5.25, 2.3, 2.15,
         "Navegador\n(PC o celular)\n\ncámara web\nvía getUserMedia", fs=8)

    # --- Vercel (arriba a la derecha) ---
    ax.add_patch(Rectangle((5.30, 6.60), 4.55, 1.40, fc="#F1F1F4", ec=GRIS, lw=1, zorder=1))
    ax.text(5.48, 7.73, "VERCEL · plan gratuito", fontsize=8.5, fontweight="bold",
            color=GRIS, zorder=2)
    caja(ax, 5.52, 6.78, 4.10, 0.80,
         "frontend estático\nindex.html · app.js · styles.css   (CDN, HTTPS)",
         "#E6E6EC", GRIS, fs=7.5)

    # --- ngrok ---
    ax.add_patch(Rectangle((5.30, 4.55), 4.55, 1.65, fc="#F7F3EE", ec=NARANJA, lw=1, zorder=1))
    ax.text(5.48, 5.93, "NGROK · plan gratuito", fontsize=8.5, fontweight="bold",
            color=NARANJA, zorder=2)
    caja(ax, 5.52, 4.73, 4.10, 1.03,
         "túnel TLS → https://xxxx.ngrok-free.app\nla URL cambia en cada arranque",
         "#F2DCC4", NARANJA, fs=7.5)

    # --- PC local ---
    ax.add_patch(Rectangle((0.15, 0.40), 9.70, 3.55, fc="#EFF6F1", ec=VERDE, lw=1, zorder=1))
    ax.text(0.33, 3.68, "PC DE DESARROLLO (Windows · Ryzen 7 · 32 GB RAM · RTX 3050)",
            fontsize=8.5, fontweight="bold", color=VERDE, zorder=2)

    caja(ax, 0.45, 2.15, 2.9, 1.25, "Uvicorn / FastAPI\nlocalhost:8000\n/ws/translate",
         "#D9EBDF", VERDE, fs=8)
    caja(ax, 3.60, 2.15, 2.9, 1.25, "Preprocesador\nMediaPipe\nHandLandmarker",
         "#D9EBDF", VERDE, fs=8)
    caja(ax, 6.75, 2.15, 2.9, 1.25, "ModeloLSE\nBiLSTM\n(PyTorch, CPU)",
         "#D9EBDF", VERDE, fs=8)
    caja(ax, 1.70, 0.62, 6.6, 1.10,
         "modelo_lsa64_lstm.pt (1,4 MB)  ·  common/features.py  ·  hand_landmarker.task\n"
         "LSA_CORS_ORIGINS = dominio de Vercel",
         "white", VERDE, fs=7.5)

    flecha(ax, 3.35, 2.78, 3.60, 2.78)
    flecha(ax, 6.50, 2.78, 6.75, 2.78)
    flecha(ax, 5.00, 1.72, 5.00, 2.15, color=VERDE)

    # --- Flujos entre nodos ---
    flecha(ax, 2.95, 7.25, 5.30, 7.25)
    ax.text(3.05, 7.47, "①  HTTPS", fontsize=7.5, color=GRIS)
    ax.text(3.05, 6.98, "descarga la interfaz", fontsize=7, color=GRIS)

    flecha(ax, 2.95, 5.80, 5.30, 5.40)
    ax.text(3.05, 5.82, "②  WSS", fontsize=7.5, color=GRIS)
    ax.text(3.05, 5.03, "frames JPEG / JSON", fontsize=7, color=GRIS)

    flecha(ax, 7.00, 4.55, 7.00, 3.95)
    ax.text(7.15, 4.22, "③  túnel → localhost:8000", fontsize=7.5, color=GRIS)

    ax.text(0.33, 0.12,
            "El navegador descarga la interfaz de Vercel y abre el WebSocket contra la URL de ngrok, "
            "que lo reenvía al backend local.",
            fontsize=7.5, color=GRIS)
    guardar(fig, "fig_despliegue.png")


def main() -> None:
    print("Generando figuras del anteproyecto en", SALIDA)
    fig_loso_modelo_servido()
    fig_comparacion_modelos()
    fig_clases_implementado()
    fig_despliegue()


if __name__ == "__main__":
    main()
