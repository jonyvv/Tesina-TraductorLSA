# -*- coding: utf-8 -*-
"""
backend/app/prediccion.py

Value object de una predicción, tal como aparece en el diagrama de clases
(`Prediccion: etiqueta, confianza, es_valida(), a_dict()`).

Se agregó el campo `estabilidad` respecto al diagrama original: antes el
suavizado temporal reescribía `confianza` con la proporción de votos del buffer,
así que un mismo nombre significaba dos cosas distintas según en qué punto del
pipeline se lo mirara (probabilidad del modelo vs. consistencia en el tiempo), y
las dos se comparaban contra el mismo umbral. Ahora son campos separados con su
propio umbral. Ver `SesionTraduccion._suavizar` en traductor_service.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prediccion:
    etiqueta: str | None
    confianza: float                  # probabilidad que le asigna el MODELO
    umbral: float = 0.6               # umbral mínimo de confianza del modelo
    estabilidad: float = 1.0          # proporción de votos en el buffer temporal
    umbral_estabilidad: float = 0.6   # umbral mínimo de consistencia temporal

    # Para una predicción cruda de un solo frame (la que devuelve
    # `ModeloLSE.predecir`) el suavizado todavía no se aplicó, así que
    # `estabilidad` arranca en 1.0 y el chequeo no la penaliza.

    def es_valida(self) -> bool:
        """Una predicción se considera válida si hay etiqueta, el modelo está
        suficientemente seguro, Y la detección se mantuvo estable en el tiempo."""
        return (
            self.etiqueta is not None
            and self.confianza >= self.umbral
            and self.estabilidad >= self.umbral_estabilidad
        )

    def a_dict(self) -> dict:
        return {
            "seña": self.etiqueta if self.es_valida() else None,
            "confianza": round(float(self.confianza), 4),
            "estabilidad": round(float(self.estabilidad), 4),
            "valida": self.es_valida(),
        }

    @classmethod
    def vacia(cls, umbral: float = 0.6) -> "Prediccion":
        return cls(etiqueta=None, confianza=0.0, umbral=umbral, estabilidad=0.0)
