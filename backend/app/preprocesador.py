# -*- coding: utf-8 -*-
"""
backend/app/preprocesador.py

Detecta la mano con MediaPipe y produce el vector de features normalizado,
tal como aparece en el diagrama de clases
(`Preprocesador: detector, tam_entrada, extraerMano(), normalizar(), obtenerLandmarks()`).

Usa el módulo compartido `common.features` para garantizar que el preprocesado
en producción sea IDÉNTICO al usado durante el entrenamiento (ver
common/features.py para la justificación de por qué esto es crítico, y para
la nota sobre por qué se usa la API MediaPipe Tasks en vez de la legacy
`mediapipe.solutions`).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

# Permite importar `common/` sin necesidad de instalar el paquete (ver README).
sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.features import (  # noqa: E402
    FEATURE_VECTOR_LENGTH,
    FEATURE_VERSION,
    FeatureExtractionResult,
    build_feature_vector,
    new_hands_detector,
)


class Preprocesador:
    def __init__(self, max_num_hands: int = 2):
        self.detector = new_hands_detector(max_num_hands=max_num_hands)
        self.tam_entrada = FEATURE_VECTOR_LENGTH
        self.version_features = FEATURE_VERSION
        # Una única instancia de HandLandmarker se comparte entre todas las
        # conexiones WebSocket, y el backend ahora procesa los frames en un
        # threadpool (ver websocket_handler.py). MediaPipe NO es thread-safe:
        # dos hilos llamando a detect() sobre el mismo landmarker pueden
        # corromper su estado interno. El lock serializa el acceso.
        self._lock = threading.Lock()

    def obtener_landmarks(self, frame_bgr: np.ndarray):
        """Corre MediaPipe sobre el frame y devuelve el HandLandmarkerResult crudo."""
        with self._lock:
            return self.detector.process(frame_bgr)

    def extraer_mano(self, frame_bgr: np.ndarray) -> FeatureExtractionResult:
        """Pipeline completo: detecta manos y devuelve el vector de features listo
        para pasarle al modelo. Ya viene normalizado (ver `normalizar`)."""
        result = self.obtener_landmarks(frame_bgr)
        return build_feature_vector(result)

    def normalizar(self, arr: np.ndarray) -> np.ndarray:
        """Punto de extensión explícito: si en el futuro se agrega normalización
        adicional a nivel de imagen (ej. ecualización de histograma, resize fijo
        antes de pasar por MediaPipe), va acá. Hoy la normalización relevante
        (traslación/escala de landmarks) ya se aplica dentro de
        `common.features.build_feature_vector`."""
        return arr

    def close(self):
        self.detector.close()
