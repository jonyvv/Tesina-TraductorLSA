# -*- coding: utf-8 -*-
"""
common/download_model.py

Descarga el archivo de modelo `hand_landmarker.task` que requiere la API
MediaPipe Tasks (`mediapipe.tasks.vision.HandLandmarker`).

Por qué hace falta este paso (a diferencia del prototipo de escritorio
original, que no lo necesitaba): la API legacy `mediapipe.solutions.hands`
traía el modelo empaquetado dentro de la librería. La API nueva (Tasks), que
es la que hay que usar porque la legacy ya no está disponible en los
paquetes actuales de PyPI, requiere el archivo del modelo como un asset
externo. Se descarga una sola vez y se versiona localmente (NO se sube al
repositorio git por su tamaño — está en .gitignore).

Uso:
    python common/download_model.py

Corrobora la descarga con un chequeo de tamaño mínimo, y si `common/models/`
ya tiene el archivo, no vuelve a descargarlo.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Modelo oficial de Google, variante "float16" (liviana, ~7.5 MB), landmarker
# de una sola mano por instancia de detección pero soporta múltiples manos
# vía num_hands en las opciones. Ver:
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
MIN_EXPECTED_SIZE_BYTES = 1_000_000  # ~1 MB; el real ronda 7-8 MB, esto es solo un piso de sanidad


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > MIN_EXPECTED_SIZE_BYTES:
        print(f"[OK] El modelo ya existe en {MODEL_PATH} "
              f"({MODEL_PATH.stat().st_size / 1e6:.1f} MB). No se vuelve a descargar.")
        return

    print(f"Descargando modelo desde:\n  {MODEL_URL}\nhacia:\n  {MODEL_PATH}\n")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[ERROR] No se pudo descargar el modelo automáticamente: {exc}\n\n"
            f"Si tu red bloquea storage.googleapis.com, descargalo manualmente "
            f"desde un navegador y guardalo como:\n  {MODEL_PATH}\n\n"
            f"URL: {MODEL_URL}\n"
            f"(También hay enlaces alternativos en la documentación oficial: "
            f"https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)"
        )
        sys.exit(1)

    size_mb = MODEL_PATH.stat().st_size / 1e6
    if MODEL_PATH.stat().st_size < MIN_EXPECTED_SIZE_BYTES:
        print(f"[!] El archivo descargado pesa solo {size_mb:.2f} MB, "
              f"parece incompleto o corrupto. Borralo y volvé a intentar.")
        sys.exit(1)

    print(f"[OK] Modelo descargado correctamente ({size_mb:.1f} MB).")


if __name__ == "__main__":
    main()
