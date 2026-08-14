from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

from common.features import build_feature_vector

try:
    from common.features import ESPEJADO_CANONICO
except ImportError:
    # La constante la define la rama `lean` en common/features.py. Hasta que se
    # haga el merge, se asume el mismo valor para no romper el contrato.
    ESPEJADO_CANONICO = True

from .config import DATASET_VIDEO_EXTENSIONS


@dataclass
class VideoSample:
    path: Path
    label: str
    subject: str | None = None
    split: str | None = None


def normalize_path_key(value: str) -> str:
    return Path(value).as_posix().lower().strip()


def load_annotations(csv_path: Path | None) -> dict[str, dict[str, str]]:
    if not csv_path:
        return {}
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo de annotations: {csv_path}")

    annotations: dict[str, dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_key = row.get("video") or row.get("path") or row.get("file")
            label = row.get("label")
            if not video_key or not label:
                continue
            annotations[normalize_path_key(video_key)] = {
                "label": label.strip(),
                "subject": (row.get("subject") or row.get("sujeto") or "").strip() or None,
                "split": (row.get("split") or "").strip().lower() or None,
            }
    return annotations


def find_video_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DATASET_VIDEO_EXTENSIONS
    )


# LSA64 nombra sus archivos como CCC_SSS_RRR.mp4, donde:
#   CCC = clase de la seña (001..064)
#   SSS = sujeto que la ejecuta (001..010)
#   RRR = repeticion (001..005)
# Sin este caso especial, el parseo generico de abajo tomaba tokens[0] como
# "indice" y armaba la etiqueta con SSS_RRR, es decir entrenaba a predecir
# sujeto+repeticion en vez de la sena. Ver tests/test_lsa64_data.py.
LSA64_FILENAME_RE = re.compile(r"^(?P<clase>\d{3})_(?P<sujeto>\d{3})_(?P<repeticion>\d{3})$")


def parse_lsa64_stem(stem: str) -> dict[str, str] | None:
    """Devuelve {'clase','sujeto','repeticion'} si el nombre sigue el patron
    de LSA64, o None si no aplica (para caer al parseo generico)."""
    match = LSA64_FILENAME_RE.match(stem.strip())
    if not match:
        return None
    return match.groupdict()


def lsa64_label(clase: str, label_map: dict[str, str] | None = None) -> str:
    """Etiqueta legible para una clase de LSA64.

    Por defecto usa 'clase_NN' en vez de inventar el nombre de la sena: los
    nombres reales se pasan con --labels-map (JSON o CSV clase,nombre) para
    que queden trazables a la fuente citada en la tesina.
    """
    if label_map and clase in label_map:
        return label_map[clase]
    return f"clase_{int(clase):02d}"


def infer_label_from_path(
    video_path: Path,
    dataset_dir: Path,
    label_map: dict[str, str] | None = None,
) -> str | None:
    parsed = parse_lsa64_stem(video_path.stem)
    if parsed:
        return lsa64_label(parsed["clase"], label_map)

    relative = video_path.relative_to(dataset_dir)
    if len(relative.parts) >= 2:
        return relative.parts[0]

    tokens = [token for token in re.split(r"[_\-\s]+", video_path.stem) if token]
    if not tokens:
        return None

    stop_tokens = {"r", "l", "b", "right", "left", "both"}
    if tokens[0].isdigit() and len(tokens) >= 2:
        label_tokens = []
        for token in tokens[1:]:
            if token.lower() in stop_tokens:
                break
            label_tokens.append(token)
        if label_tokens:
            return " ".join(label_tokens)

    label_tokens = []
    for token in tokens:
        if token.lower() in stop_tokens:
            break
        label_tokens.append(token)
    return " ".join(label_tokens) if label_tokens else None


def infer_subject_from_stem(video_path: Path) -> str | None:
    parsed = parse_lsa64_stem(video_path.stem)
    if parsed:
        # Sin esto el sujeto quedaba en None y train_val_test_split agrupaba
        # por nombre de archivo: un grupo por video, o sea la misma persona
        # apareciendo en train y en test (fuga de datos).
        return f"sujeto_{int(parsed['sujeto']):02d}"

    stem = video_path.stem.lower()
    tokens = stem.replace("-", "_").split("_")
    for token in tokens:
        if token.startswith("subj") and len(token) > 4:
            return token[4:]
        if token.startswith("subject") and len(token) > 7:
            return token[7:]
        if token.startswith("s") and token[1:].isdigit():
            return token[1:]
        if token.startswith("p") and token[1:].isdigit():
            return token[1:]
    return None


def load_label_map(path: Path | None) -> dict[str, str]:
    """Carga un mapa clase -> nombre de sena desde JSON ({"001": "Opaco"}) o
    CSV con columnas clase,nombre. Las claves se normalizan a 3 digitos."""
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de labels: {path}")

    raw: dict[str, str] = {}
    if path.suffix.lower() == ".json":
        import json

        with open(path, "r", encoding="utf-8") as handle:
            raw = {str(key): str(value) for key, value in json.load(handle).items()}
    else:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                clase = row.get("clase") or row.get("class") or row.get("id")
                nombre = row.get("nombre") or row.get("label") or row.get("name")
                if clase and nombre:
                    raw[str(clase).strip()] = nombre.strip()

    return {f"{int(key):03d}": value for key, value in raw.items() if str(key).strip().isdigit()}


def build_samples(
    dataset_dir: Path,
    annotations: dict[str, dict[str, str]],
    label_map: dict[str, str] | None = None,
) -> list[VideoSample]:
    samples: list[VideoSample] = []
    for video_path in find_video_files(dataset_dir):
        key_candidates = {
            normalize_path_key(str(video_path)),
            normalize_path_key(str(video_path.relative_to(dataset_dir))),
            normalize_path_key(video_path.name),
        }
        annotation = next((annotations[key] for key in key_candidates if key in annotations), None)

        if annotation:
            label = annotation["label"]
            subject = annotation.get("subject")
            split = annotation.get("split")
        else:
            label = infer_label_from_path(video_path, dataset_dir, label_map)
            subject = infer_subject_from_stem(video_path)
            split = None

        if not label:
            print(f"[!] Saltando sin etiqueta clara: {video_path}")
            continue

        samples.append(VideoSample(path=video_path, label=label, subject=subject, split=split))
    return samples


def train_val_test_split(samples: list[VideoSample], seed: int) -> tuple[list[int], list[int], list[int]]:
    explicit = {"train": [], "val": [], "test": []}
    grouped: dict[str, list[int]] = {}

    for idx, sample in enumerate(samples):
        if sample.split in explicit:
            explicit[sample.split].append(idx)
            continue
        grouped.setdefault(sample.subject or sample.path.stem, []).append(idx)

    if not any(explicit.values()):
        group_keys = list(grouped.keys())
        random.Random(seed).shuffle(group_keys)
        train_cut = max(1, int(len(group_keys) * 0.7))
        val_cut = max(train_cut + 1, int(len(group_keys) * 0.85)) if len(group_keys) >= 3 else train_cut
        train_keys = group_keys[:train_cut]
        val_keys = group_keys[train_cut:val_cut]
        test_keys = group_keys[val_cut:]
        return (
            [idx for key in train_keys for idx in grouped[key]],
            [idx for key in val_keys for idx in grouped[key]],
            [idx for key in test_keys for idx in grouped[key]],
        )

    train_idx = list(explicit["train"])
    val_idx = list(explicit["val"])
    test_idx = list(explicit["test"])
    if grouped:
        group_keys = list(grouped.keys())
        random.Random(seed).shuffle(group_keys)
        train_cut = max(1, int(len(group_keys) * 0.7))
        val_cut = max(train_cut + 1, int(len(group_keys) * 0.85)) if len(group_keys) >= 3 else train_cut
        train_idx.extend(idx for key in group_keys[:train_cut] for idx in grouped[key])
        val_idx.extend(idx for key in group_keys[train_cut:val_cut] for idx in grouped[key])
        test_idx.extend(idx for key in group_keys[val_cut:] for idx in grouped[key])
    return train_idx, val_idx, test_idx


def sequence_from_video(
    video_path: Path,
    detector,
    frame_step: int,
    max_frames: int,
    keep_empty_frames: bool,
):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] No se pudo abrir el video: {video_path}")
        return None

    frames: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            # Contrato ESPEJADO_CANONICO de common/features.py: TODO el proyecto
            # procesa la imagen espejada. Sin este flip, los videos de LSA64
            # entran crudos y MediaPipe asigna la handedness al revés que en
            # captura e inferencia: la seña cae en el slot de la otra mano.
            # Como 42 de las 64 señas de LSA64 son de una sola mano, el vector
            # queda con los 69 valores en el bloque equivocado y el modelo
            # entrenado así predice cualquier cosa al servirlo desde el frontend.
            if ESPEJADO_CANONICO:
                frame = cv2.flip(frame, 1)

            result = build_feature_vector(detector.process(frame))
            if keep_empty_frames or result.any_hand_present():
                frames.append(result.vector.astype(np.float32))

            frame_index += 1
            if max_frames > 0 and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    if not frames:
        return None
    return np.stack(frames, axis=0)

