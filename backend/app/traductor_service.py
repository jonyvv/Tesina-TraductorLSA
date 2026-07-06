# -*- coding: utf-8 -*-
"""
backend/app/traductor_service.py

Orquesta preprocesado + modelo, tal como aparece en el diagrama de clases
(`TraductorService: preprocesador, modelo, procesar(), traducir(), historial()`).
"""
from __future__ import annotations

from collections import deque

from .frame import Frame
from .modelo_lse import ModeloLSE
from .prediccion import Prediccion
from .preprocesador import Preprocesador


class TraductorService:
    def __init__(self, preprocesador: Preprocesador, modelo: ModeloLSE,
                 tam_buffer_suavizado: int = 10):
        self.preprocesador = preprocesador
        self.modelo = modelo
        # Buffer de suavizado por voto mayoritario: estabiliza la predicción
        # entre frames consecutivos (mismo patrón que el prototipo de escritorio,
        # que sí acertó en esta parte de la UX).
        self._buffer = deque(maxlen=tam_buffer_suavizado)
        self._historial: list[Prediccion] = []

    def procesar(self, frame: Frame) -> Prediccion:
        """Pipeline completo para UN frame: decodificar -> extraer mano ->
        predecir. No aplica suavizado (eso lo hace `traducir`, que es el punto
        de entrada real desde el WebSocketHandler)."""
        arr = frame.a_array()
        resultado = self.preprocesador.extraer_mano(arr)

        if not resultado.any_hand_present():
            return Prediccion.vacia(umbral=self.modelo.umbral_confianza)

        return self.modelo.predecir(resultado.vector)

    def traducir(self, datos: bytes) -> dict:
        """Punto de entrada usado por el WebSocketHandler (ver diagrama de
        secuencia: `traducir(frame)` -> ... -> `resultado dict`)."""
        frame = Frame(datos=datos)
        prediccion_cruda = self.procesar(frame)

        etiqueta_suavizada, confianza_suavizada = self._suavizar(prediccion_cruda)
        prediccion_final = Prediccion(
            etiqueta=etiqueta_suavizada,
            confianza=confianza_suavizada,
            umbral=self.modelo.umbral_confianza,
        )

        if prediccion_final.es_valida():
            self._historial.append(prediccion_final)

        return prediccion_final.a_dict()

    def _suavizar(self, prediccion: Prediccion) -> tuple[str | None, float]:
        if prediccion.es_valida():
            self._buffer.append(prediccion.etiqueta)

        if len(self._buffer) < max(3, self._buffer.maxlen // 2):
            return None, 0.0

        valores = list(self._buffer)
        mas_comun = max(set(valores), key=valores.count)
        confianza = valores.count(mas_comun) / len(valores)
        return mas_comun, confianza

    def historial(self) -> list[dict]:
        return [p.a_dict() for p in self._historial]

    def reiniciar_buffer(self) -> None:
        """Se llama al reconectar un cliente nuevo, para no arrastrar el
        suavizado de una sesión anterior."""
        self._buffer.clear()
