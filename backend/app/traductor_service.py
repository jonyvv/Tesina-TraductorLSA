from __future__ import annotations

from collections import deque

import numpy as np

from .frame import Frame
from .modelo_lse import ModeloLSE
from .prediccion import Prediccion
from .preprocesador import Preprocesador


class TraductorService:
    def __init__(self, preprocesador: Preprocesador, modelo: ModeloLSE, tam_buffer_suavizado: int = 10):
        self.preprocesador = preprocesador
        self.modelo = modelo
        self._buffer = deque(maxlen=tam_buffer_suavizado)
        self._buffer_secuencia = deque(maxlen=max(1, modelo.ventana_inferencia))
        self._historial: list[Prediccion] = []

    def procesar(self, frame: Frame) -> Prediccion:
        arr = frame.a_array()
        resultado = self.preprocesador.extraer_mano(arr)

        if not resultado.any_hand_present():
            self._buffer_secuencia.clear()
            return Prediccion.vacia(umbral=self.modelo.umbral_confianza)

        if self.modelo.requiere_secuencia:
            self._buffer_secuencia.append(resultado.vector)
            if len(self._buffer_secuencia) < self._buffer_secuencia.maxlen:
                return Prediccion.vacia(umbral=self.modelo.umbral_confianza)
            secuencia = np.asarray(list(self._buffer_secuencia), dtype=np.float32)
            return self.modelo.predecir_secuencia(secuencia)

        return self.modelo.predecir(resultado.vector)

    def traducir(self, datos: bytes) -> dict:
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
        self._buffer.clear()
        self._buffer_secuencia.clear()
