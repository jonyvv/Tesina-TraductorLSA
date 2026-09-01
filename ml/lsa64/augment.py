# -*- coding: utf-8 -*-
"""
Augmentation de secuencias de landmarks, para el split de entrenamiento.

Por que hace falta
------------------
El modelo sobreajusta fuerte: loss de train 0,064 contra val 0,79. Con 2862
secuencias y 64 clases hay ~45 muestras por clase, y de esas solo 7 sujetos
distintos aportan al train de cada fold. La red memoriza personas.

Las tres transformaciones de aca atacan cosas distintas:

- `frame_drop` es la mas importante para este dataset. MediaPipe detecta manos
  en entre el 58 % y el 76 % de los frames segun la persona, y la accuracy por
  sujeto correlaciona 0,68 con esa tasa de retencion. Tirar frames al azar en
  entrenamiento simula justamente al sujeto con mala deteccion, que es el caso
  que el modelo falla.
- `time_scale` simula que cada persona sena a distinta velocidad.
- `noise` simula la imprecision del propio estimador de landmarks.

Todas se aplican SOLO al split de entrenamiento y SOLO durante el fit. Nunca a
validacion ni a test: augmentar la evaluacion cambiaria lo que se esta midiendo.

Layout del vector de features (ver common/features.py)
------------------------------------------------------
Por mano (69): [presencia] + [21 landmarks x 3 coords] + [5 angulos]
Dos manos: offsets 0 y FEATURES_PER_HAND.

El flag de presencia NUNCA se toca: es lo que distingue "mano detectada" de
"mano ausente", y ensuciarlo convertiria un frame vacio en uno con datos falsos.
"""
from __future__ import annotations

import numpy as np

from common.features import coord_slices, presence_indices

MIN_FRAMES_TRAS_AUGMENTAR = 4


def _frames_con_mano(seq: np.ndarray) -> np.ndarray:
    """Mascara booleana: True donde al menos una mano fue detectada.

    Los offsets salen del ancho real del vector, no de una constante: v1 mide 138
    y v2 (con posicion de la muneca) mide 144, y el bloque por mano pasa de 69 a
    72. Hardcodear FEATURES_PER_HAND leeria la columna equivocada en v2.
    """
    return np.logical_or.reduce([seq[:, i] > 0 for i in presence_indices(seq.shape[1])])


def aplicar_noise(seq: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    """Ruido gaussiano sobre las coordenadas normalizadas.

    Solo en frames con mano detectada: sumarle ruido a un frame vacio lo
    convertiria en una mano fantasma en el origen.

    Los angulos se dejan como estan. Son derivados de las coordenadas, asi que
    en rigor deberian recalcularse, pero para sigma chico (<= 0,02 sobre
    coordenadas normalizadas) el desvio inducido en los angulos es despreciable
    frente al costo de rehacer toda la normalizacion por muestra.
    """
    if sigma <= 0:
        return seq
    seq = seq.copy()
    con_mano = _frames_con_mano(seq)
    if not con_mano.any():
        return seq
    for sl in coord_slices(seq.shape[1]):
        bloque = seq[con_mano, sl]
        seq[con_mano, sl] = bloque + rng.normal(0.0, sigma, size=bloque.shape).astype(seq.dtype)
    return seq


def aplicar_frame_drop(seq: np.ndarray, p: float, rng: np.random.RandomState) -> np.ndarray:
    """Descarta cada frame con probabilidad `p`, conservando el orden.

    Simula al sujeto con mala deteccion de manos. No toca la longitud minima:
    si el sorteo dejaria menos de MIN_FRAMES_TRAS_AUGMENTAR, devuelve la
    secuencia sin tocar antes que producir una muestra degenerada.
    """
    if p <= 0 or len(seq) <= MIN_FRAMES_TRAS_AUGMENTAR:
        return seq
    quedan = rng.random_sample(len(seq)) >= p
    if quedan.sum() < MIN_FRAMES_TRAS_AUGMENTAR:
        return seq
    return seq[quedan]


def aplicar_time_scale(seq: np.ndarray, s: float, rng: np.random.RandomState) -> np.ndarray:
    """Reescala la duracion por un factor uniforme en [1-s, 1+s].

    Remuestrea por vecino mas cercano sobre el eje temporal: no interpola entre
    frames, asi que no inventa poses intermedias ni rompe el flag de presencia.
    """
    if s <= 0 or len(seq) <= MIN_FRAMES_TRAS_AUGMENTAR:
        return seq
    factor = 1.0 + rng.uniform(-s, s)
    nuevo_largo = int(round(len(seq) * factor))
    nuevo_largo = max(MIN_FRAMES_TRAS_AUGMENTAR, nuevo_largo)
    if nuevo_largo == len(seq):
        return seq
    idx = np.clip(np.round(np.linspace(0, len(seq) - 1, nuevo_largo)).astype(int), 0, len(seq) - 1)
    return seq[idx]


def augmentar(
    seq: np.ndarray,
    rng: np.random.RandomState,
    noise: float = 0.0,
    frame_drop: float = 0.0,
    time_scale: float = 0.0,
) -> np.ndarray:
    """Aplica las tres transformaciones en orden. Con todo en 0 devuelve `seq` tal cual.

    El orden importa: primero se altera la grilla temporal (drop y scale) y
    despues se agrega ruido, para que el ruido sea independiente por frame
    sobreviviente y no se duplique al remuestrear.
    """
    if noise <= 0 and frame_drop <= 0 and time_scale <= 0:
        return seq
    seq = aplicar_frame_drop(seq, frame_drop, rng)
    seq = aplicar_time_scale(seq, time_scale, rng)
    seq = aplicar_noise(seq, noise, rng)
    return np.ascontiguousarray(seq, dtype=np.float32)


def hay_augmentation(config) -> bool:
    return (
        getattr(config, "aug_noise", 0.0) > 0
        or getattr(config, "aug_frame_drop", 0.0) > 0
        or getattr(config, "aug_time_scale", 0.0) > 0
    )
