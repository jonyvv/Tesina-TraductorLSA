# -*- coding: utf-8 -*-
"""
backend/app/prediccion.py

Value object de una predicción, tal como aparece en el diagrama de clases
(`Prediccion: etiqueta, confianza, es_valida(), a_dict()`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prediccion:
    etiqueta: str | None
    confianza: float
    umbral: float = 0.6

    def es_valida(self) -> bool:
        """Una predicción solo se considera válida si supera el umbral de confianza
        Y hay una etiqueta (evita reportar `None` con confianza alta por error)."""
        return self.etiqueta is not None and self.confianza >= self.umbral

    def a_dict(self) -> dict:
        return {
            "seña": self.etiqueta if self.es_valida() else None,
            "confianza": round(float(self.confianza), 4),
            "valida": self.es_valida(),
        }

    @classmethod
    def vacia(cls, umbral: float = 0.6) -> "Prediccion":
        return cls(etiqueta=None, confianza=0.0, umbral=umbral)
