# -*- coding: utf-8 -*-
"""
Etapa 2 del pipeline: entrenar la BiLSTM sobre features ya extraidas.

La extraccion de landmarks con MediaPipe (etapa 1, ml/extract_features.py) es
lo caro. Aca solo se lee el cache .npz y se entrena, que son segundos. Si no
hay cache disponible, se extrae al vuelo y se guarda para la proxima.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .cache import (
    FeatureCache,
    build_meta,
    load_cache_if_compatible,
    puede_servir_frames_completos,
    save_cache,
)
from .fitting import ajustar_modelo, evaluar as _evaluate, set_seed as _set_seed
from .config import LSA64TrainingConfig, default_cache_path
from .data import (
    ESPEJADO_CANONICO,
    VideoSample,
    build_samples,
    load_annotations,
    load_label_map,
    train_val_test_split,
)


def _resolve_cache_path(config: LSA64TrainingConfig) -> Path:
    return config.cache_path or default_cache_path()


def _expected_meta(config: LSA64TrainingConfig) -> dict:
    return build_meta(
        feature_version=config.feature_version,
        feature_vector_length=config.feature_vector_length,
        frame_step=config.frame_step,
        max_frames=config.max_frames,
        keep_empty_frames=config.keep_empty_frames,
        espejado=ESPEJADO_CANONICO,
    )


def _extract_and_cache(config: LSA64TrainingConfig, cache_path: Path) -> FeatureCache:
    """Fallback cuando no hay cache: extrae en paralelo y lo guarda."""
    from .extract import default_workers, extract_sequences

    if not config.dataset_dir.exists():
        raise FileNotFoundError(f"No existe el directorio del dataset: {config.dataset_dir}")

    annotations = load_annotations(config.annotations)
    label_map = load_label_map(config.labels_map)
    samples = build_samples(config.dataset_dir, annotations, label_map)
    if not samples:
        raise ValueError("No se encontraron videos validos para entrenar.")

    workers = config.workers or default_workers()
    print(f"Extrayendo landmarks de {len(samples)} videos con {workers} procesos...")
    print("(esto tarda; se hace una sola vez y queda cacheado)")
    sequences, descartes = extract_sequences(
        samples,
        frame_step=config.frame_step,
        max_frames=config.max_frames,
        min_seq_len=config.min_seq_len,
        keep_empty_frames=True,  # siempre completo; el filtro es del entrenamiento
        workers=workers,
    )
    if not sequences:
        raise ValueError("No se pudieron extraer secuencias validas.")
    for path, motivo in descartes[:10]:
        print(f"[!] Descartado {Path(path).name}: {motivo}")
    if len(descartes) > 10:
        print(f"[!] ... y {len(descartes) - 10} descartes mas")

    valid = sorted(sequences.keys())
    cache = FeatureCache(
        sequences=[sequences[i] for i in valid],
        labels=[samples[i].label for i in valid],
        subjects=[samples[i].subject for i in valid],
        paths=[str(samples[i].path) for i in valid],
        splits=[samples[i].split for i in valid],
        meta=build_meta(
            feature_version=config.feature_version,
            feature_vector_length=config.feature_vector_length,
            frame_step=config.frame_step,
            max_frames=config.max_frames,
            keep_empty_frames=True,
            espejado=ESPEJADO_CANONICO,
            min_seq_len=config.min_seq_len,
            dataset_dir=str(config.dataset_dir),
        ),
    )
    save_cache(cache_path, cache)
    print(f"[OK] Cache guardado en {cache_path}")
    return cache


def _frames_con_mano(secuencia: np.ndarray) -> np.ndarray:
    """Máscara de frames donde MediaPipe detecto al menos una mano.

    Los indices de los flags salen de presence_indices(): dependen del ancho
    real del vector, porque el bloque de cada mano mide 69 en v1 y 72 en v2
    (con la posicion de la muneca). Ver common/features.py.
    """
    from common.features import presence_indices

    import numpy as _np

    idx = presence_indices(secuencia.shape[1])
    return _np.logical_or.reduce([secuencia[:, i] > 0 for i in idx])


def _apply_frame_filter(cache: FeatureCache, keep_empty_frames: bool) -> FeatureCache:
    """Descarta los frames sin mano si el entrenamiento no los quiere.

    Se aplica en memoria, no al extraer: la extraccion guarda la secuencia
    completa y aca se decide. Eso permite comparar los dos modos sobre el mismo
    cache en vez de re-extraer 46 minutos por cada experimento.

    Ojo con lo que significa cada modo:
      - keep_empty_frames=True  -> se conserva la grilla temporal. Un frame en
        ceros dice "aca no habia mano", y los flags de presencia lo codifican.
      - keep_empty_frames=False -> los huecos se cierran y la secuencia se
        comprime de forma irregular: el gesto le llega acelerado a saltos a la
        LSTM, y tanto mas cuanto peor detecte MediaPipe a esa persona.
    """
    if keep_empty_frames:
        return cache

    nuevas, descartados_total, frames_antes = [], 0, 0
    for secuencia in cache.sequences:
        mascara = _frames_con_mano(secuencia)
        frames_antes += len(secuencia)
        descartados_total += int((~mascara).sum())
        nuevas.append(np.ascontiguousarray(secuencia[mascara]))

    if not descartados_total:
        return cache

    print(
        f"[i] {descartados_total} de {frames_antes} frames descartados por no tener mano "
        f"({100 * descartados_total / max(frames_antes, 1):.1f}%)"
    )
    return FeatureCache(
        sequences=nuevas,
        labels=list(cache.labels),
        subjects=list(cache.subjects),
        paths=list(cache.paths),
        splits=list(cache.splits),
        meta={**cache.meta, "keep_empty_frames_aplicado": False},
    )


def _apply_min_seq_len(cache: FeatureCache, min_seq_len: int) -> FeatureCache:
    """min_seq_len solo descarta secuencias cortas, no cambia las features.

    Por eso NO invalida el cache (re-extraer 3200 videos para cambiar un filtro
    seria absurdo): se aplica en memoria al cargar. El cache guarda todo lo que
    paso el umbral de extraccion, y aca se puede endurecer, nunca aflojar.
    """
    keep = [i for i, seq in enumerate(cache.sequences) if len(seq) >= min_seq_len]
    if len(keep) == len(cache.sequences):
        return cache

    descartadas = len(cache.sequences) - len(keep)
    print(f"[i] {descartadas} secuencias descartadas por min_seq_len={min_seq_len}")
    return FeatureCache(
        sequences=[cache.sequences[i] for i in keep],
        labels=[cache.labels[i] for i in keep],
        subjects=[cache.subjects[i] for i in keep],
        paths=[cache.paths[i] for i in keep],
        splits=[cache.splits[i] for i in keep],
        meta={**cache.meta, "min_seq_len_aplicado": min_seq_len},
    )


def nombres_de_clases(clases, labels_map: Path | None) -> list[str]:
    """Traduce clase_NN -> nombre real de la sena, para MOSTRAR.

    Se aplica al final, sobre las clases ya codificadas, y nunca antes: el
    LabelEncoder ordena alfabeticamente, asi que si los nombres entraran al
    entrenamiento cambiarian que indice le toca a cada sena y el modelo saldria
    distinto segun el --labels-map que se haya pasado. Con clase_01..clase_64 el
    orden alfabetico coincide con el numerico, y ese es el orden canonico.

    Medido: entrenar con los nombres adentro movia el resultado de 0,8125 a
    0,8159 y el best_epoch de 11 a 32 sobre datos identicos. El nombre es
    presentacion; la identidad de la clase es el id.

    Por lo mismo el cache guarda siempre el id canonico: si guardara el nombre,
    cambiar el mapa obligaria a re-extraer 47 minutos y dos caches que en
    realidad son iguales quedarian marcados como incompatibles.
    """
    canonicas = [str(c) for c in clases]
    if not labels_map:
        return canonicas

    mapa = load_label_map(labels_map)
    if not mapa:
        return canonicas

    def traducir(label: str) -> str:
        match = re.fullmatch(r"clase_(\d+)", label)
        if not match:
            return label  # ya tiene nombre real: idempotente
        return mapa.get(f"{int(match.group(1)):03d}", label)

    nombres = [traducir(label) for label in canonicas]

    # "Sin nombre" es la que quedo como clase_NN, no la que ya venia con nombre:
    # sobre una lista ya traducida esto no tiene que avisar nada.
    sin_nombre = [n for n in nombres if re.fullmatch(r"clase_\d+", n)]
    if sin_nombre:
        print(f"[!] {len(sin_nombre)} clases sin nombre en el mapa: {', '.join(sin_nombre[:5])}")

    repetidos = sorted({n for n in nombres if nombres.count(n) > 1})
    if repetidos:
        # Dos clases con el mismo nombre se pisan en los reportes por clase y en
        # lo que ve el usuario, aunque el modelo las distinga perfectamente.
        print(f"[!] Nombres repetidos en el mapa: {', '.join(repetidos)}")

    traducidas = len(canonicas) - len(sin_nombre)
    print(f"Nombres de senas tomados de {labels_map.name}: {traducidas}/{len(canonicas)} clases.")
    return nombres


def _postprocesar(cache: FeatureCache, config: LSA64TrainingConfig) -> FeatureCache:
    """Filtros que se aplican en memoria al cargar. El orden importa: primero
    se descartan los frames sin mano, y recien despues se mide el largo minimo
    de la secuencia resultante."""
    cache = _apply_frame_filter(cache, config.keep_empty_frames)
    return _apply_min_seq_len(cache, config.min_seq_len)


def load_features(config: LSA64TrainingConfig) -> FeatureCache:
    cache_path = _resolve_cache_path(config)
    if config.refresh_cache:
        return _postprocesar(_extract_and_cache(config, cache_path), config)

    cache, motivos = load_cache_if_compatible(cache_path, _expected_meta(config))
    if cache is not None:
        print(f"Cache de features: {cache_path}")
        print(f"  {cache.summary()}")
        print(f"  frames: {'completos' if config.keep_empty_frames else 'solo con mano'}")

        if config.keep_empty_frames and not puede_servir_frames_completos(cache.meta):
            raise ValueError(
                f"Pediste --keep-empty-frames pero {cache_path.name} se extrajo descartando "
                f"los frames sin mano: esos frames ya no existen.\n"
                f"Volve a extraer con: python ml/extract_features.py --dataset-dir <ruta>"
            )

        cache_min = cache.meta.get("min_seq_len")
        if cache_min is not None and config.min_seq_len < cache_min:
            print(
                f"[!] Pediste min_seq_len={config.min_seq_len} pero el cache se extrajo "
                f"con {cache_min}: las secuencias mas cortas ya no estan. "
                f"Usa --refresh-cache si las necesitas."
            )
        return _postprocesar(cache, config)

    print(f"No se usa el cache ({'; '.join(motivos)}).")
    return _postprocesar(_extract_and_cache(config, cache_path), config)


def _check_subject_leakage(
    metadata: list[VideoSample],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
) -> list[str]:
    """Un sujeto que aparece en train y en test infla la accuracy: el modelo
    puede memorizar a la persona en vez de aprender la sena."""
    def subjects(indices: list[int]) -> set[str]:
        return {metadata[i].subject for i in indices if metadata[i].subject}

    train_s, val_s, test_s = subjects(train_idx), subjects(val_idx), subjects(test_idx)
    warnings = []
    for name, other, overlap in (
        ("train/test", "test", train_s & test_s),
        ("train/val", "val", train_s & val_s),
        ("val/test", "test", val_s & test_s),
    ):
        if overlap:
            warnings.append(f"sujetos compartidos entre {name}: {sorted(overlap)}")
    return warnings


def train_lsa64_model(config: LSA64TrainingConfig) -> Path:
    import torch
    from sklearn.preprocessing import LabelEncoder

    if config.frame_step < 1 or config.min_seq_len < 1:
        raise ValueError("frame_step y min_seq_len deben ser >= 1")

    _set_seed(config.seed)
    cache = load_features(config)
    sequence_list = cache.sequences
    metadata = [
        VideoSample(path=Path(p), label=lbl, subject=subj, split=split)
        for p, lbl, subj, split in zip(cache.paths, cache.labels, cache.subjects, cache.splits)
    ]

    label_encoder = LabelEncoder()
    label_list = label_encoder.fit_transform(cache.labels).astype(np.int64)
    # El encoder se ajusta SIEMPRE sobre los ids canonicos; los nombres son
    # solo para mostrar y no entran al entrenamiento. Ver nombres_de_clases().
    clases_mostradas = nombres_de_clases(label_encoder.classes_, config.labels_map)
    train_idx, val_idx, test_idx = train_val_test_split(metadata, config.seed)
    if not train_idx or not test_idx:
        raise ValueError("El split quedo vacio: revisa los sujetos/labels del dataset.")

    leakage = _check_subject_leakage(metadata, train_idx, val_idx, test_idx)
    print()
    print(f"Clases : {len(label_encoder.classes_)}")
    print(f"Split  : train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"Sujetos: train={sorted({metadata[i].subject for i in train_idx if metadata[i].subject})}")
    print(f"         test ={sorted({metadata[i].subject for i in test_idx if metadata[i].subject})}")
    if leakage:
        for warning in leakage:
            print(f"[!] FUGA DE DATOS: {warning}")
        print("[!] La accuracy reportada no es valida como resultado de la tesina.")
    else:
        print("[OK] Ningun sujeto aparece en mas de un split.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device} | seed: {config.seed} (corrida reproducible)")
    print()

    resultado = ajustar_modelo(
        sequence_list,
        label_list,
        n_clases=len(label_encoder.classes_),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        config=config,
        verbose=True,
    )
    model = resultado.model
    test_acc = resultado.test_accuracy
    y_true, y_pred = resultado.y_true, resultado.y_pred
    best_val_acc = resultado.best_val_accuracy
    best_epoch = resultado.best_epoch
    history = resultado.history
    use_validation = best_val_acc is not None

    print()
    print(f"[RESULTADO] test accuracy: {test_acc:.4f}")

    per_class = _per_class_accuracy(y_true, y_pred, clases_mostradas)
    peores = sorted(per_class.items(), key=lambda kv: kv[1])[:5]
    if peores:
        print("Clases con peor accuracy:", ", ".join(f"{k}={v:.2f}" for k, v in peores))

    config.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_classes": list(clases_mostradas),
            "feature_version": config.feature_version,
            "feature_vector_length": config.feature_vector_length,
            "hidden_size": config.hidden_size,
            "frame_step": config.frame_step,
            "max_frames": config.max_frames,
            "min_seq_len": config.min_seq_len,
            "dataset": "LSA64",
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "samples": [
                {
                    "video": str(sample.path),
                    "label": sample.label,
                    "subject": sample.subject,
                    "split": sample.split,
                }
                for sample in metadata
            ],
        },
        config.output,
    )

    summary_path = config.output.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": "LSA64",
                "feature_version": config.feature_version,
                "feature_vector_length": config.feature_vector_length,
                "labels": list(clases_mostradas),
                "labels_canonicas": [str(c) for c in label_encoder.classes_],
                "labels_map": str(config.labels_map) if config.labels_map else None,
                "test_accuracy": test_acc,
                "best_val_accuracy": best_val_acc if use_validation else None,
                "best_epoch": best_epoch if use_validation else None,
                "epochs_corridas": len(history),
                "hiperparametros": {
                    "hidden_size": config.hidden_size,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "epochs_max": config.epochs,
                    "patience": config.patience,
                    "frame_step": config.frame_step,
                    "max_frames": config.max_frames,
                    "seed": config.seed,
                    "dropout": config.dropout,
                    "weight_decay": config.weight_decay,
                    "aug_noise": config.aug_noise,
                    "aug_frame_drop": config.aug_frame_drop,
                    "aug_time_scale": config.aug_time_scale,
                    "keep_empty_frames": config.keep_empty_frames,
                },
                "split": {
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "test_size": len(test_idx),
                    "train_subjects": sorted({metadata[i].subject for i in train_idx if metadata[i].subject}),
                    "val_subjects": sorted({metadata[i].subject for i in val_idx if metadata[i].subject}),
                    "test_subjects": sorted({metadata[i].subject for i in test_idx if metadata[i].subject}),
                    "fugas_detectadas": leakage,
                },
                "accuracy_por_clase": per_class,
                "history": history,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] Modelo guardado en: {config.output}")
    print(f"[OK] Resumen guardado en: {summary_path}")
    return config.output


def _per_class_accuracy(y_true: list[int], y_pred: list[int], classes) -> dict[str, float]:
    totales = Counter(y_true)
    aciertos = Counter(t for t, p in zip(y_true, y_pred) if t == p)
    return {
        str(classes[label]): round(aciertos[label] / total, 4)
        for label, total in sorted(totales.items())
    }
