# -*- coding: utf-8 -*-
"""
Etapa 2 del pipeline: entrenar la BiLSTM sobre features ya extraidas.

La extraccion de landmarks con MediaPipe (etapa 1, ml/extract_features.py) es
lo caro. Aca solo se lee el cache .npz y se entrena, que son segundos. Si no
hay cache disponible, se extrae al vuelo y se guarda para la proxima.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from common.features import FEATURE_VECTOR_LENGTH
from common.models.lsa64 import BiLSTMClassifier

from .cache import FeatureCache, build_meta, load_cache_if_compatible, save_cache
from .config import LSA64TrainingConfig, default_cache_path
from .data import (
    ESPEJADO_CANONICO,
    VideoSample,
    build_samples,
    load_annotations,
    load_label_map,
    train_val_test_split,
)


def _evaluate(model, loader, device, collect: bool = False):
    import torch

    model.eval()
    correct, total = 0, 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, lengths, y_batch in loader:
            x = x.to(device)
            y_batch = y_batch.to(device)
            out = model(x, lengths)
            pred = out.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
            if collect:
                y_true.extend(y_batch.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())
    accuracy = correct / max(total, 1)
    return (accuracy, y_true, y_pred) if collect else accuracy


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
        keep_empty_frames=config.keep_empty_frames,
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
            keep_empty_frames=config.keep_empty_frames,
            espejado=ESPEJADO_CANONICO,
            min_seq_len=config.min_seq_len,
            dataset_dir=str(config.dataset_dir),
        ),
    )
    save_cache(cache_path, cache)
    print(f"[OK] Cache guardado en {cache_path}")
    return cache


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


def load_features(config: LSA64TrainingConfig) -> FeatureCache:
    cache_path = _resolve_cache_path(config)
    if config.refresh_cache:
        return _apply_min_seq_len(_extract_and_cache(config, cache_path), config.min_seq_len)

    cache, motivos = load_cache_if_compatible(cache_path, _expected_meta(config))
    if cache is not None:
        print(f"Cache de features: {cache_path}")
        print(f"  {cache.summary()}")
        cache_min = cache.meta.get("min_seq_len")
        if cache_min is not None and config.min_seq_len < cache_min:
            print(
                f"[!] Pediste min_seq_len={config.min_seq_len} pero el cache se extrajo "
                f"con {cache_min}: las secuencias mas cortas ya no estan. "
                f"Usa --refresh-cache si las necesitas."
            )
        return _apply_min_seq_len(cache, config.min_seq_len)

    print(f"No se usa el cache ({'; '.join(motivos)}).")
    return _apply_min_seq_len(_extract_and_cache(config, cache_path), config.min_seq_len)


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
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset

    if config.frame_step < 1 or config.min_seq_len < 1:
        raise ValueError("frame_step y min_seq_len deben ser >= 1")

    cache = load_features(config)
    sequence_list = cache.sequences
    metadata = [
        VideoSample(path=Path(p), label=lbl, subject=subj, split=split)
        for p, lbl, subj, split in zip(cache.paths, cache.labels, cache.subjects, cache.splits)
    ]

    label_encoder = LabelEncoder()
    label_list = label_encoder.fit_transform(cache.labels).astype(np.int64)
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

    class SequenceDataset(Dataset):
        def __init__(self, indices: list[int]):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            seq_idx = self.indices[index]
            return torch.from_numpy(sequence_list[seq_idx]), int(label_list[seq_idx])

    def collate(batch):
        sequences_batch, labels_batch = zip(*batch)
        lengths = torch.tensor([len(sequence) for sequence in sequences_batch], dtype=torch.long)
        padded = pad_sequence(sequences_batch, batch_first=True)
        return padded, lengths, torch.tensor(labels_batch, dtype=torch.long)

    train_loader = DataLoader(
        SequenceDataset(train_idx), batch_size=config.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = (
        DataLoader(SequenceDataset(val_idx), batch_size=config.batch_size, shuffle=False, collate_fn=collate)
        if val_idx else None
    )
    test_loader = DataLoader(
        SequenceDataset(test_idx), batch_size=config.batch_size, shuffle=False, collate_fn=collate
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    model = BiLSTMClassifier(FEATURE_VECTOR_LENGTH, config.hidden_size, len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    epochs_sin_mejora = 0
    use_validation = val_loader is not None
    history: list[dict] = []

    print()
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for x, lengths, y_batch in train_loader:
            x = x.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            out = model(x, lengths)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / max(len(train_loader), 1)
        if use_validation:
            val_acc = _evaluate(model, val_loader, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                epochs_sin_mejora = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_sin_mejora += 1
            metric_text = f" - val acc: {val_acc:.4f}"
        else:
            val_acc = None
            metric_text = ""

        history.append({"epoch": epoch + 1, "loss": epoch_loss, "val_accuracy": val_acc})
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {epoch_loss:.4f}{metric_text}")

        # Early stopping: con 64 clases y ~2200 secuencias, seguir despues de
        # que la validacion deja de mejorar solo agrega overfitting.
        if use_validation and config.patience > 0 and epochs_sin_mejora >= config.patience:
            print(f"Early stopping en epoch {epoch + 1} (mejor: epoch {best_epoch}, val acc {best_val_acc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, y_true, y_pred = _evaluate(model, test_loader, device, collect=True)
    print()
    print(f"[RESULTADO] test accuracy: {test_acc:.4f}")

    per_class = _per_class_accuracy(y_true, y_pred, label_encoder.classes_)
    peores = sorted(per_class.items(), key=lambda kv: kv[1])[:5]
    if peores:
        print("Clases con peor accuracy:", ", ".join(f"{k}={v:.2f}" for k, v in peores))

    config.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_classes": list(label_encoder.classes_),
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
                "labels": list(label_encoder.classes_),
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
