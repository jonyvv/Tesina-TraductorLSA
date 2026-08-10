# -*- coding: utf-8 -*-
"""
backend/app/websocket_handler.py

Maneja las conexiones WebSocket, tal como aparece en el diagrama de clases
(`WebSocketHandler: conexiones, servicio, conectar(), recibirFrame(), enviarResultado()`)
y en el diagrama de secuencia (loop cada ~100ms: enviarFrame -> traducir -> resultado dict).
"""
from __future__ import annotations

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from .traductor_service import SesionTraduccion, TraductorService

logger = logging.getLogger("lsa.websocket")


class WebSocketHandler:
    def __init__(self, servicio: TraductorService):
        self.servicio = servicio
        # Una sesión (buffer de suavizado + historial) POR conexión. Antes el
        # buffer vivía en el TraductorService compartido, así que dos clientes
        # simultáneos se pisaban las predicciones entre sí.
        self.conexiones: dict[WebSocket, SesionTraduccion] = {}

    async def conectar(self, ws: WebSocket) -> None:
        await ws.accept()
        self.conexiones[ws] = self.servicio.nueva_sesion()
        logger.info("Cliente conectado (%d activos)", len(self.conexiones))

    def desconectar(self, ws: WebSocket) -> None:
        self.conexiones.pop(ws, None)
        logger.info("Cliente desconectado (%d activos)", len(self.conexiones))

    async def recibir_frame(self, ws: WebSocket) -> None:
        """Loop principal de la conexión: recibe frames binarios, los traduce,
        y devuelve el resultado como JSON. Corresponde al `loop: cada ~100ms`
        del diagrama de secuencia (el ritmo real lo marca el cliente)."""
        sesion = self.conexiones.get(ws)
        if sesion is None:  # no debería pasar: conectar() siempre corre antes
            await ws.close()
            return

        try:
            while True:
                datos = await ws.receive_bytes()
                try:
                    # MediaPipe + OpenCV + scikit-learn son sincrónicos y pesados:
                    # llamarlos directo acá bloquea el event loop entero mientras
                    # corren, así que con más de un cliente conectado las
                    # traducciones se serializan y la latencia se dispara.
                    # `run_in_threadpool` los saca del loop. (El acceso a
                    # MediaPipe está serializado con un lock dentro de
                    # Preprocesador, que no es thread-safe.)
                    resultado = await run_in_threadpool(sesion.traducir, datos)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Error procesando frame")
                    resultado = {"error": str(exc)}
                await self.enviar_resultado(ws, resultado)
        except WebSocketDisconnect:
            pass
        finally:
            self.desconectar(ws)

    async def enviar_resultado(self, ws: WebSocket, resultado: dict) -> None:
        await ws.send_text(json.dumps(resultado, ensure_ascii=False))
