# -*- coding: utf-8 -*-
"""
common/features.py
===================

Módulo COMPARTIDO de extracción y normalización de landmarks de mano.

IMPORTANTE — por qué existe este módulo separado:
Este código es usado tanto por el pipeline de captura/entrenamiento (carpeta `ml/`)
como por el backend de inferencia en tiempo real (carpeta `backend/`). Si esta lógica
se duplicara en ambos lados (como pasaba en el prototipo original de escritorio) y
un día se modifica en un solo lugar, el modelo entrenado y el modelo en producción
quedan desincronizados de forma silenciosa: no hay ningún error, simplemente
predicciones incorrectas. Por eso vive en un único lugar y ambos lados lo importan.

NOTA IMPORTANTE SOBRE LA API DE MEDIAPIPE (verificado al armar este proyecto):

El prototipo original usaba `mediapipe.solutions.hands`, la API "legacy". Esa API
YA NO viene incluida en los paquetes de `mediapipe` que se instalan hoy con pip
(a partir de aproximadamente la serie 0.10.21+, `import mediapipe as mp;
mp.solutions` directamente no existe — es un problema conocido y sin resolver
en el repositorio oficial de Google, ver github.com/google-ai-edge/mediapipe
issues #6192, #6200, #6204). Se comprobó al armar este proyecto: con
`mediapipe==0.10.33` instalado desde PyPI, `mp.solutions.hands` lanza
`AttributeError: module 'mediapipe' has no attribute 'solutions'`.

Por eso este módulo usa la API soportada actualmente: **MediaPipe Tasks**
(`mediapipe.tasks.vision.HandLandmarker`). Es la misma tecnología por debajo
(el mismo modelo de detección de manos de Google), pero con una interfaz
distinta y, a diferencia de la legacy, requiere descargar explícitamente el
archivo del modelo (`hand_landmarker.task`) — ver `common/download_model.py`.

Decisiones de diseño del esquema de features (corrigen problemas detectados
en el análisis del prototipo de escritorio):

1. Vector de longitud FIJA con slots estables para mano IZQUIERDA y mano DERECHA
   (según `handedness`), en vez de tomar "la mano [0]" según el orden de
   detección. Esto permite reconocer señas bimanuales y evita que el mismo
   gesto genere vectores distintos según el orden en que se detectaron las manos.

2. Normalización respecto a la muñeca (traslación) y al tamaño de la mano (escala).
   Sin esto, la misma seña genera vectores distintos según la distancia a la
   cámara o la posición dentro del encuadre.

3. Flag de presencia explícito por mano (0.0 / 1.0), para que el modelo pueda
   distinguir "mano ausente" de "mano en el origen de coordenadas".

4. FEATURE_VERSION explícito: si el esquema de features cambia, hay que subir
   la versión y re-entrenar. El backend valida esto al cargar el modelo (ver
   backend/app/modelo_lse.py) para no servir predicciones con features
   desalineadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constantes del esquema de features
# ---------------------------------------------------------------------------

FEATURE_VERSION = "v1"

NUM_LANDMARKS = 21          # puntos del esqueleto de la mano que da MediaPipe
COORDS_PER_LANDMARK = 3     # x, y, z
ANGLES_PER_HAND = 5         # un ángulo por dedo (pulgar, índice, medio, anular, meñique)
PRESENCE_FLAG_LEN = 1

FEATURES_PER_HAND = (
    PRESENCE_FLAG_LEN + (NUM_LANDMARKS * COORDS_PER_LANDMARK) + ANGLES_PER_HAND
)  # 1 + 63 + 5 = 69

NUM_HANDS = 2  # slots fijos: [izquierda, derecha]
FEATURE_VECTOR_LENGTH = FEATURES_PER_HAND * NUM_HANDS  # 138

# Cadenas de landmarks para calcular el ángulo de flexión de cada dedo
# (base de la mano -> nudillo -> punta), igual criterio que el prototipo original.
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],      # pulgar
    [0, 5, 6, 7, 8],      # índice
    [0, 9, 10, 11, 12],   # medio
    [0, 13, 14, 15, 16],  # anular
    [0, 17, 18, 19, 20],  # meñique
]

# Ruta por defecto del modelo descargado por common/download_model.py
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


@dataclass
class FeatureExtractionResult:
    """Resultado de procesar un frame: el vector de features + info de debug/UI."""
    vector: np.ndarray               # shape (FEATURE_VECTOR_LENGTH,)
    left_present: bool
    right_present: bool
    raw_result: object = None        # HandLandmarkerResult crudo, útil para dibujar overlay

    def any_hand_present(self) -> bool:
        return self.left_present or self.right_present


# ---------------------------------------------------------------------------
# Funciones internas de cálculo (no dependen de MediaPipe directamente,
# reciben arrays de numpy — así se pueden testear sin cámara ni modelo)
# ---------------------------------------------------------------------------

def normalize_landmarks(landmarks_xyz: np.ndarray) -> np.ndarray:
    """
    Normaliza landmarks para que sean invariantes a:
      - posición de la mano en el encuadre (se centra en la muñeca, landmark 0)
      - distancia de la mano a la cámara (se reescala por un tamaño de referencia
        de la propia mano: distancia muñeca -> base del dedo medio, landmark 9)
    """
    wrist = landmarks_xyz[0].copy()
    centered = landmarks_xyz - wrist
    scale = np.linalg.norm(centered[9]) + 1e-6
    return centered / scale


def calculate_angles(landmarks_xyz: np.ndarray) -> list[float]:
    """Ángulo de flexión de cada dedo, calculado sobre landmarks ya normalizados."""
    angles = []
    for chain in FINGER_CHAINS:
        p1 = landmarks_xyz[chain[0]]
        p2 = landmarks_xyz[chain[2]]
        p3 = landmarks_xyz[chain[4]]
        v1, v2 = p1 - p2, p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(float(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    return angles


def _hand_feature_vector(landmark_list) -> np.ndarray:
    """`landmark_list`: lista de 21 NormalizedLandmark (con atributos .x .y .z)."""
    raw = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)
    norm = normalize_landmarks(raw)
    angles = calculate_angles(norm)
    presence = np.array([1.0], dtype=np.float32)
    return np.concatenate(
        [presence, norm.flatten().astype(np.float32), np.array(angles, dtype=np.float32)]
    )


def _empty_hand_vector() -> np.ndarray:
    return np.zeros(FEATURES_PER_HAND, dtype=np.float32)


# ---------------------------------------------------------------------------
# API pública — extracción a partir del resultado de MediaPipe Tasks
# ---------------------------------------------------------------------------

def build_feature_vector(hand_landmarker_result) -> FeatureExtractionResult:
    """
    Construye el vector de features de un frame a partir de un
    `mediapipe.tasks.vision.HandLandmarkerResult` (el resultado de
    `HandLandmarker.detect(...)`).

    Devuelve siempre un vector de longitud FEATURE_VECTOR_LENGTH, aunque no se
    detecte ninguna mano (en ese caso, todo en ceros con presencia=False).
    """
    left = _empty_hand_vector()
    right = _empty_hand_vector()
    left_present = False
    right_present = False

    hand_landmarks_list = getattr(hand_landmarker_result, "hand_landmarks", None) or []
    handedness_list = getattr(hand_landmarker_result, "handedness", None) or []

    for landmark_list, handedness in zip(hand_landmarks_list, handedness_list):
        # handedness es una lista de Category; tomamos la de mayor score (la [0]).
        label = handedness[0].category_name  # "Left" o "Right"
        vec = _hand_feature_vector(landmark_list)
        if label == "Left":
            left = vec
            left_present = True
        else:
            right = vec
            right_present = True

    vector = np.concatenate([left, right])
    return FeatureExtractionResult(
        vector=vector,
        left_present=left_present,
        right_present=right_present,
        raw_result=hand_landmarker_result,
    )


class HandDetector:
    """
    Wrapper fino sobre `mediapipe.tasks.vision.HandLandmarker` en modo IMAGE
    (adecuado para procesar frames sueltos, uno a la vez, como llegan por
    WebSocket o desde una cámara leída con OpenCV — no requiere manejar
    timestamps de video como el modo VIDEO/LIVE_STREAM).

    Uso:
        detector = HandDetector()
        result = detector.process(frame_bgr)   # frame_bgr: array de OpenCV
        extraccion = build_feature_vector(result)
        detector.close()

    O como context manager:
        with HandDetector() as detector:
            result = detector.process(frame_bgr)
    """

    def __init__(self, model_path: Optional[str] = None, max_num_hands: int = 2,
                 min_hand_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5):
        import mediapipe as mp

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo de MediaPipe en '{path}'.\n"
                f"Corré primero:\n    python common/download_model.py\n"
                f"para descargar 'hand_landmarker.task' (una sola vez, ~8 MB)."
            )

        base_options = mp.tasks.BaseOptions(model_asset_path=str(path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray):
        """Recibe un frame BGR (formato OpenCV) y devuelve un HandLandmarkerResult."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect(mp_image)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def new_hands_detector(model_path: Optional[str] = None, max_num_hands: int = 2,
                        min_hand_detection_confidence: float = 0.6,
                        min_tracking_confidence: float = 0.5) -> HandDetector:
    """Factory con parámetros consistentes en todo el proyecto (captura,
    entrenamiento y backend). Ver `HandDetector` para el detalle de uso."""
    return HandDetector(
        model_path=model_path,
        max_num_hands=max_num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


# ---------------------------------------------------------------------------
# Utilidad de dibujo (reemplaza a mp.solutions.drawing_utils, que no existe
# más en la API nueva). Solo para visualización en capture_dataset.py — no
# se usa en el backend.
# ---------------------------------------------------------------------------

# Conexiones del esqueleto de la mano (21 puntos), equivalente a lo que
# exponía la API legacy como HAND_CONNECTIONS.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # índice
    (5, 9), (9, 10), (10, 11), (11, 12),     # medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anular
    (13, 17), (17, 18), (18, 19), (19, 20),  # meñique
    (0, 17),                                  # palma
]


def draw_landmarks(frame_bgr: np.ndarray, hand_landmarker_result) -> None:
    """Dibuja los landmarks y conexiones sobre `frame_bgr`, in-place.
    Reemplazo mínimo de `mp.solutions.drawing_utils.draw_landmarks`."""
    import cv2

    h, w = frame_bgr.shape[:2]
    hand_landmarks_list = getattr(hand_landmarker_result, "hand_landmarks", None) or []

    for landmark_list in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmark_list]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, points[a], points[b], (0, 255, 0), 2)
        for x, y in points:
            cv2.circle(frame_bgr, (x, y), 3, (255, 0, 0), -1)
