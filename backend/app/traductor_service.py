# -*- coding: utf-8 -*-
"""
backend/app/traductor_service.py

Orquesta preprocesado + modelo, tal como aparece en el diagrama de clases
(`TraductorService: preprocesador, modelo, procesar(), traducir(), historial()`).

REFINAMIENTO respecto al diagrama original (ver docs/ARQUITECTURA.md §2):
el suavizado y el historial se movieron a una clase aparte, `SesionTraduccion`,
que se instancia UNA POR CONEXIÓN WebSocket. En el diagrama, `TraductorService`
guardaba el buffer como atributo propio; como el servicio se construye una sola
vez al arrancar la app y se comparte entre todos los clientes, ese buffer era
estado global: dos usuarios simultáneos se mezclaban las predicciones, y la
llamada a `reiniciar_buffer()` de uno le borraba el suavizado al otro.

`TraductorService` queda entonces sin estado mutable (solo preprocesador +
modelo, ambos compartibles), y todo lo que es "conversación con un cliente"
vive en `SesionTraduccion`. Eso incluye la ventana de frames de los modelos
dinámicos (la BiLSTM de LSA64): igual que el buffer de suavizado, es estado de
UNA conversación, y compartirla entre conexiones mezclaría las señas de dos
personas dentro de la misma secuencia.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .frame import Frame
from .modelo_lse import ModeloLSE
from .prediccion import Prediccion
from .preprocesador import Preprocesador

# Se importa desde el módulo compartido para no duplicar el criterio de
# serialización de landmarks entre captura y backend.
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.features import landmarks_para_overlay  # noqa: E402


@dataclass
class ResultadoFrame:
    """Todo lo que produce el pipeline para UN frame.

    En el diagrama de clases `procesar()` devolvía solo una `Prediccion`. Se
    amplió porque el cliente también necesita los landmarks (para dibujar el
    overlay sobre el video) y el tiempo de cómputo (para medir latencia, ver
    docs/ARQUITECTURA.md §9). Volver a correr MediaPipe para obtenerlos sería
    duplicar la parte más cara del pipeline, así que se devuelven de una.
    """
    prediccion: Prediccion
    landmarks: list[dict] = field(default_factory=list)
    manos_detectadas: int = 0


class TraductorService:
    """Sin estado por cliente: se puede compartir entre todas las conexiones."""

    def __init__(self, preprocesador: Preprocesador, modelo: ModeloLSE,
                 tam_buffer_suavizado: int = 10):
        self.preprocesador = preprocesador
        self.modelo = modelo
        self.tam_buffer_suavizado = tam_buffer_suavizado

    def procesar(self, frame: Frame, secuencia: deque | None = None) -> ResultadoFrame:
        """Pipeline completo para UN frame: decodificar -> extraer mano ->
        predecir. No aplica suavizado (eso lo hace `SesionTraduccion.traducir`,
        que es el punto de entrada real desde el WebSocketHandler).

        `secuencia` es la ventana de frames de la sesión que llama. La recibe
        por parámetro en vez de guardarla: así el servicio sigue sin estado y
        dos conexiones no se pisan la secuencia (mismo motivo que el buffer de
        suavizado). Solo la usan los modelos dinámicos.
        """
        arr = frame.a_array()
        resultado = self.preprocesador.extraer_mano(arr)

        landmarks = landmarks_para_overlay(resultado.raw_result)
        manos = int(resultado.left_present) + int(resultado.right_present)

        def _resultado(prediccion: Prediccion) -> ResultadoFrame:
            return ResultadoFrame(
                prediccion=prediccion,
                landmarks=landmarks,
                manos_detectadas=manos,
            )

        if not resultado.any_hand_present():
            # Sin manos se corta la seña: la ventana acumulada dejó de ser una
            # secuencia continua, así que arrancar de cero es lo correcto.
            if secuencia is not None:
                secuencia.clear()
            return _resultado(Prediccion.vacia(umbral=self.modelo.umbral_confianza))

        if self.modelo.requiere_secuencia:
            if secuencia is None:
                raise RuntimeError(
                    "El modelo cargado necesita una ventana de frames. Usá "
                    "`servicio.nueva_sesion().traducir(...)` en vez de llamar a "
                    "`procesar()` suelto."
                )
            secuencia.append(resultado.vector)
            # Hasta no llenar la ventana no hay con qué predecir. Se devuelve
            # una predicción vacía, pero CON los landmarks: el overlay tiene que
            # dibujarse desde el primer frame igual.
            if len(secuencia) < secuencia.maxlen:
                return _resultado(Prediccion.vacia(umbral=self.modelo.umbral_confianza))
            ventana = np.asarray(list(secuencia), dtype=np.float32)
            return _resultado(self.modelo.predecir_secuencia(ventana))

        return _resultado(self.modelo.predecir(resultado.vector))

    def nueva_sesion(self) -> "SesionTraduccion":
        """Crea el estado de suavizado/historial para una conexión nueva."""
        return SesionTraduccion(servicio=self, tam_buffer=self.tam_buffer_suavizado)


class SesionTraduccion:
    """Estado de UNA conexión WebSocket: buffer de suavizado + ventana + historial."""

    def __init__(self, servicio: TraductorService, tam_buffer: int = 10):
        self.servicio = servicio
        # Buffer de suavizado por voto mayoritario: estabiliza la predicción
        # entre frames consecutivos (mismo patrón que el prototipo de escritorio,
        # que sí acertó en esta parte de la UX).
        self._buffer: deque[tuple[str | None, float]] = deque(maxlen=tam_buffer)
        # Ventana de features para los modelos dinámicos. Con un modelo estático
        # `ventana_inferencia` es 1 y la ventana no se usa.
        self._secuencia: deque = deque(maxlen=max(1, servicio.modelo.ventana_inferencia))
        self._historial: list[Prediccion] = []

    def traducir(self, datos: bytes) -> dict:
        """Punto de entrada usado por el WebSocketHandler (ver diagrama de
        secuencia: `traducir(frame)` -> ... -> `resultado dict`)."""
        inicio = time.perf_counter()

        frame = Frame(datos=datos)
        resultado_frame = self.servicio.procesar(frame, secuencia=self._secuencia)

        etiqueta, confianza, estabilidad = self._suavizar(resultado_frame.prediccion)

        prediccion_final = Prediccion(
            etiqueta=etiqueta,
            confianza=confianza,
            umbral=self.servicio.modelo.umbral_confianza,
            estabilidad=estabilidad,
        )

        if prediccion_final.es_valida():
            self._historial.append(prediccion_final)

        respuesta = prediccion_final.a_dict()
        respuesta["landmarks"] = resultado_frame.landmarks
        respuesta["manos"] = resultado_frame.manos_detectadas
        # Tiempo de cómputo del servidor. Restándoselo al round-trip que mide el
        # cliente se obtiene el costo de red + codificación, que son dos cosas
        # distintas y conviene poder separarlas al medir (ver ARQUITECTURA.md §9).
        respuesta["ms_servidor"] = round((time.perf_counter() - inicio) * 1000, 2)
        return respuesta

    def _suavizar(self, prediccion: Prediccion) -> tuple[str | None, float, float]:
        """Voto mayoritario sobre las últimas N predicciones.

        Devuelve (etiqueta, confianza_del_modelo, estabilidad).

        Dos cambios respecto a la versión anterior:

        1. Ahora se registra TAMBIÉN el frame sin seña válida (como `None`). Antes
           solo se agregaban las predicciones válidas, así que el buffer nunca se
           vaciaba: bajabas las manos y el backend seguía reportando la última seña
           con `valida: true` para siempre. Metiendo los `None` al voto, el
           resultado decae solo cuando dejás de hacer la seña.

        2. Se separan dos números que antes viajaban con el mismo nombre y se
           comparaban contra el mismo umbral, aunque miden cosas distintas:
             - `confianza`   -> qué tan seguro está el MODELO (probabilidad media
                                de los frames que votaron por la etiqueta ganadora)
             - `estabilidad` -> qué tan consistente fue la detección en el TIEMPO
                                (proporción de votos de la ganadora en el buffer)
           Sirven además como dos métricas distintas para la sección de resultados.
        """
        if prediccion.es_valida():
            self._buffer.append((prediccion.etiqueta, prediccion.confianza))
        else:
            self._buffer.append((None, 0.0))

        # Hasta no tener suficientes muestras, no arriesgamos una etiqueta.
        minimo = max(3, (self._buffer.maxlen or 0) // 2)
        if len(self._buffer) < minimo:
            return None, 0.0, 0.0

        etiquetas = [etq for etq, _ in self._buffer]
        ganadora = max(set(etiquetas), key=etiquetas.count)
        votos = etiquetas.count(ganadora)
        estabilidad = votos / len(etiquetas)

        if ganadora is None:
            return None, 0.0, estabilidad

        confianzas = [c for etq, c in self._buffer if etq == ganadora]
        confianza_media = sum(confianzas) / len(confianzas)
        return ganadora, confianza_media, estabilidad

    def historial(self) -> list[dict]:
        return [p.a_dict() for p in self._historial]

    def reiniciar_buffer(self) -> None:
        self._buffer.clear()
        self._secuencia.clear()
