# -*- coding: utf-8 -*-
"""
ml/capture_dataset.py

Captura de muestras para el dataset de entrenamiento. Reemplaza al capturador
Tkinter del prototipo original con una versión más simple (ventana de OpenCV)
que corrige los problemas detectados en el análisis del repo base:

  1. Usa el mismo `common.features` que usará el backend en producción -> no
     hay drift entre captura y servido (incluye usar la misma API de
     MediaPipe Tasks, no la legacy `mediapipe.solutions`).
  2. Soporta AMBAS manos (no solo "la primera mano detectada").
  3. Cada muestra guarda metadatos de sesión (sujeto, condición de luz) para
     poder hacer un split train/test por sesión y evitar fuga de datos.
  4. Elimina el bug de "features" con forma distinta entre estático/dinámico:
     acá TODAS las muestras (estáticas o de una secuencia dinámica) usan
     vectores de longitud fija `FEATURE_VECTOR_LENGTH`. Las dinámicas se
     guardan como una LISTA de esos vectores (secuencia), nunca aplanadas ni
     mezcladas con las estáticas en el mismo esquema.

Requiere haber corrido antes (una sola vez):
    python common/download_model.py

Uso:
    python capture_dataset.py --sujeto leandro --sesion 1 --luz "natural"

Controles:
    [ESPACIO]  captura una muestra estática (una letra/pose)
    [R]        inicia/detiene grabación de una secuencia dinámica (una palabra)
    [N]        nueva palabra/letra a capturar
    [Q]        salir
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.features import (  # noqa: E402
    FEATURE_VERSION,
    build_feature_vector,
    draw_landmarks,
    new_hands_detector,
)

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "sign_language_dataset"
DYNAMIC_SEQUENCE_LEN = 30  # frames por muestra dinámica


def guardar_muestra(label: str, tipo: str, payload, sujeto: str, sesion: str, luz: str) -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    sample_id = uuid.uuid4().hex[:8]
    sample = {
        "label": label,
        "type": tipo,  # "static" | "dynamic"
        "feature_version": FEATURE_VERSION,
        "payload": payload,  # vector (static) o lista de vectores (dynamic)
        "sujeto": sujeto,
        "sesion": sesion,
        "condicion_luz": luz,
        "timestamp": datetime.now().isoformat(),
    }
    filename = DATASET_DIR / f"{label}_{sujeto}_{sesion}_{tipo}_{sample_id}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(sample, f)
    return filename


def pedir_palabra(actual: str) -> str:
    nueva = input(f"Palabra/letra a capturar (actual: '{actual}'): ").strip().lower()
    return nueva if nueva else actual


def main() -> None:
    parser = argparse.ArgumentParser(description="Captura de dataset LSA")
    parser.add_argument("--sujeto", required=True, help="Identificador de la persona que capturó")
    parser.add_argument("--sesion", required=True, help="Identificador de la sesión (ej. '1', '2025-06-30-tarde')")
    parser.add_argument("--luz", default="no especificada", help="Condición de iluminación")
    parser.add_argument("--camara", type=int, default=0)
    args = parser.parse_args()

    try:
        detector = new_hands_detector(max_num_hands=2)
    except FileNotFoundError as exc:
        print(exc)
        return

    cap = cv2.VideoCapture(args.camara)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {args.camara}")
        detector.close()
        return

    palabra_actual = pedir_palabra("")
    grabando_dinamica = False
    buffer_dinamico: list[list[float]] = []
    contador_muestras = 0

    print("\nControles: [ESPACIO]=captura estática  [R]=grabar dinámica  [N]=nueva palabra  [Q]=salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = detector.process(frame)
        draw_landmarks(frame, result)

        resultado = build_feature_vector(result)

        if grabando_dinamica and resultado.any_hand_present():
            buffer_dinamico.append(resultado.vector.tolist())
            if len(buffer_dinamico) >= DYNAMIC_SEQUENCE_LEN:
                guardar_muestra(palabra_actual, "dynamic", buffer_dinamico, args.sujeto, args.sesion, args.luz)
                contador_muestras += 1
                buffer_dinamico = []
                grabando_dinamica = False
                print(f"[OK] Muestra dinámica guardada. Total sesión: {contador_muestras}")

        cv2.putText(frame, f"Palabra: {palabra_actual}", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Muestras (sesion): {contador_muestras}", (10, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        if grabando_dinamica:
            cv2.putText(frame, f"GRABANDO... {len(buffer_dinamico)}/{DYNAMIC_SEQUENCE_LEN}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Captura de dataset LSA", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            palabra_actual = pedir_palabra(palabra_actual)
        elif key == ord("r"):
            grabando_dinamica = not grabando_dinamica
            buffer_dinamico = []
            print("Grabación dinámica:", "INICIADA" if grabando_dinamica else "CANCELADA")
        elif key == 32:  # espacio
            if resultado.any_hand_present():
                guardar_muestra(palabra_actual, "static", resultado.vector.tolist(),
                                 args.sujeto, args.sesion, args.luz)
                contador_muestras += 1
                print(f"[OK] Muestra estática guardada. Total sesión: {contador_muestras}")
            else:
                print("[!] No se detecta ninguna mano, no se guardó la muestra")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"\nSesión finalizada. {contador_muestras} muestras guardadas en {DATASET_DIR}")


if __name__ == "__main__":
    main()
