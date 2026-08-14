# -*- coding: utf-8 -*-
"""
backend/app/main.py

Punto de entrada de FastAPI. Arma el grafo de dependencias:
Preprocesador + ModeloLSE -> TraductorService -> WebSocketHandler

Correr con:
    uvicorn app.main:app --reload --port 8000
(desde la carpeta backend/, con el entorno virtual activado)
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .modelo_lse import ModeloLSE
from .preprocesador import Preprocesador
from .traductor_service import TraductorService
from .websocket_handler import WebSocketHandler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.features import HAND_CONNECTIONS  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lsa.main")

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "backend" / "models"

# Qué modelo se sirve, en orden de prioridad:
#   1. lo que diga LSA_MODEL_PATH (permite probar un checkpoint sin tocar código)
#   2. la BiLSTM de LSA64, si el .pt está presente (señas dinámicas, 64 clases)
#   3. el .joblib de scikit-learn (señas estáticas del dataset propio)
# El formato lo resuelve `modelos/factory.py` por la extensión del archivo.
_MODELO_LSTM = MODELS_DIR / "modelo_lsa64_lstm.pt"
RUTA_MODELO = Path(
    os.getenv("LSA_MODEL_PATH")
    or (_MODELO_LSTM if _MODELO_LSTM.exists() else MODELS_DIR / "modelo_lse.joblib")
)
UMBRAL_CONFIANZA = 0.6

# --- Construcción de dependencias (equivalente al "carga modelo" del diagrama) ---
# Nota: tanto `Preprocesador` (requiere el archivo hand_landmarker.task, ver
# common/download_model.py) como `ModeloLSE` (requiere el .joblib entrenado)
# pueden no estar disponibles todavía en un checkout nuevo del proyecto. El
# arranque del servidor NO debe fallar por eso -> se resuelven de forma
# perezosa al iniciar, y /health refleja el estado real.
preprocesador: Preprocesador | None = None
modelo = ModeloLSE(ruta_modelo=str(RUTA_MODELO), umbral_confianza=UMBRAL_CONFIANZA)
traductor_service: TraductorService | None = None
ws_handler: WebSocketHandler | None = None


def inicializar() -> None:
    global preprocesador, traductor_service, ws_handler

    try:
        preprocesador = Preprocesador(max_num_hands=2)
        logger.info("Preprocesador (MediaPipe HandLandmarker) inicializado.")
    except FileNotFoundError as exc:
        logger.warning(
            "%s\nEl backend arrancará igual, pero /ws/translate va a fallar "
            "hasta que corras: python common/download_model.py",
            exc,
        )

    try:
        modelo.cargar()
        logger.info("Modelo cargado. Clases: %s", modelo.clases)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning(
            "%s\nEl backend arrancará igual, pero /ws/translate va a fallar "
            "hasta que exista un modelo entrenado válido "
            "(ver ml/train.py o ml/train_lsa64.py).",
            exc,
        )

    if preprocesador is not None:
        traductor_service = TraductorService(preprocesador=preprocesador, modelo=modelo)
        ws_handler = WebSocketHandler(servicio=traductor_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # `@app.on_event("startup")` está deprecado en FastAPI; el reemplazo es este
    # context manager, que además da un lugar explícito para liberar recursos al
    # apagar (MediaPipe mantiene handles nativos abiertos).
    inicializar()
    yield
    if preprocesador is not None:
        preprocesador.close()
        logger.info("Preprocesador liberado.")


app = FastAPI(title="Traductor LSA API", version="0.1.0", lifespan=lifespan)

# En desarrollo, permitir cualquier origen. En producción, restringir al dominio
# real del frontend (ver docs/ARQUITECTURA.md, sección Deploy).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "preprocesador_listo": preprocesador is not None,
        "modelo_cargado": modelo.model is not None,
        "modelo": RUTA_MODELO.name,
        "clases": modelo.clases,
        "listo_para_traducir": ws_handler is not None and modelo.model is not None,
    }


@app.get("/model/info")
def model_info() -> dict:
    return {
        "clases": modelo.clases,
        "umbral_confianza": modelo.umbral_confianza,
        "feature_version": modelo.feature_version_entrenamiento,
        # Con un modelo de secuencias (la BiLSTM de LSA64) el backend no puede
        # predecir hasta juntar `ventana_inferencia` frames CON mano detectada.
        # El cliente lo necesita para no interpretar ese arranque como un error.
        "requiere_secuencia": modelo.requiere_secuencia,
        "ventana_inferencia": modelo.ventana_inferencia,
        # El frontend dibuja el esqueleto de la mano con estas conexiones. Se
        # sirven desde acá en vez de hardcodearlas en el JS para que haya una
        # sola definición del esqueleto en todo el proyecto (misma razón por la
        # que existe common/: si cambia, no puede quedar desincronizado).
        "conexiones_mano": HAND_CONNECTIONS,
    }


@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket) -> None:
    if ws_handler is None:
        # El preprocesador (MediaPipe) no pudo inicializarse -> avisar y cerrar
        # en vez de crashear. Ver /health para diagnosticar qué falta.
        await ws.accept()
        await ws.send_text(
            '{"error": "Backend no inicializado del todo. '
            'Revisa /health: falta el modelo de MediaPipe o el modelo entrenado."}'
        )
        await ws.close()
        return

    await ws_handler.conectar(ws)
    await ws_handler.recibir_frame(ws)


# Sirve el frontend estático (HTML/JS/CSS) directamente desde FastAPI para no
# necesitar un segundo servidor durante el desarrollo/demo de la tesina.
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
