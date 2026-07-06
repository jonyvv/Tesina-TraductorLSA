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

from .traductor_service import TraductorService

logger = logging.getLogger("lsa.websocket")


class WebSocketHandler:
    def __init__(self, servicio: TraductorService):
        self.servicio = servicio
        self.conexiones: list[WebSocket] = []

    async def conectar(self, ws: WebSocket) -> None:
        await ws.accept()
        self.conexiones.append(ws)
        self.servicio.reiniciar_buffer()
        logger.info("Cliente conectado (%d activos)", len(self.conexiones))

    def desconectar(self, ws: WebSocket) -> None:
        if ws in self.conexiones:
            self.conexiones.remove(ws)
        logger.info("Cliente desconectado (%d activos)", len(self.conexiones))

    async def recibir_frame(self, ws: WebSocket) -> None:
        """Loop principal de la conexión: recibe frames binarios, los traduce,
        y devuelve el resultado como JSON. Corresponde al `loop: cada ~100ms`
        del diagrama de secuencia (el ritmo real lo marca el cliente)."""
        try:
            while True:
                datos = await ws.receive_bytes()
                try:
                    resultado = self.servicio.traducir(datos)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Error procesando frame")
                    resultado = {"error": str(exc)}
                await self.enviar_resultado(ws, resultado)
        except WebSocketDisconnect:
            self.desconectar(ws)

    async def enviar_resultado(self, ws: WebSocket, resultado: dict) -> None:
        await ws.send_text(json.dumps(resultado, ensure_ascii=False))
