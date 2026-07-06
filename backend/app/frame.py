# -*- coding: utf-8 -*-
"""
backend/app/frame.py

Representa un frame recibido por WebSocket, tal como aparece en el diagrama
de clases (`Frame: datos, timestamp, serializar(), dimensiones()`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Frame:
    datos: bytes            # bytes crudos recibidos por WS (JPEG/PNG codificado)
    timestamp: float = field(default_factory=time.time)

    def a_array(self) -> np.ndarray:
        """Decodifica los bytes JPEG/PNG a un array BGR (formato OpenCV)."""
        buffer = np.frombuffer(self.datos, dtype=np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("No se pudo decodificar el frame recibido (bytes inválidos)")
        return img

    def dimensiones(self) -> tuple[int, int]:
        arr = self.a_array()
        h, w = arr.shape[:2]
        return (w, h)

    def serializar(self) -> bytes:
        """Devuelve los bytes crudos (ya vienen serializados desde el cliente)."""
        return self.datos
