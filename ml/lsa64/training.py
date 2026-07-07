from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.features import FEATURE_VECTOR_LENGTH
from common.models.lsa64 import BiLSTMClassifier

from .config import LSA64TrainingConfig
from .data import build_samples, load_annotations, sequence_from_video, train_val_test_split


def _evaluate(model, loader, device):
    import torch

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, lengths, y_batch in loader:
            x = x.to(device)
            y_batch = y_batch.to(device)
            out = model(x, lengths)
            pred = out.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / max(total, 1)


def train_lsa64_model(config: LSA64TrainingConfig) -> Path:
    import torch
    from sklearn.preprocessing import LabelEncoder
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset

    if not config.dataset_dir.exists():
        raise FileNotFoundError(f"No existe el directorio del dataset: {config.dataset_dir}")
    if config.frame_step < 1 or config.min_seq_len < 1:
        raise ValueError("frame_step y min_seq_len deben ser >= 1")

    annotations = load_annotations(config.annotations)
    print("Dataset:", config.dataset_dir)
    print("Existe:", config.dataset_dir.exists())

    for p in config.dataset_dir.iterdir():
        print(p)
    samples = build_samples(config.dataset_dir, annotations)
    if not samples:
        raise ValueError("No se encontraron videos validos para entrenar.")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform([sample.label for sample in samples])
    train_idx, val_idx, test_idx = train_val_test_split(samples, config.seed)

    detector = None
    sequences: dict[int, np.ndarray] = {}
    try:
        from common.features import new_hands_detector

        detector = new_hands_detector(max_num_hands=2)
        for idx, sample in enumerate(samples):
            seq = sequence_from_video(
                sample.path,
                detector=detector,
                frame_step=config.frame_step,
                max_frames=config.max_frames,
                keep_empty_frames=config.keep_empty_frames,
            )
            if seq is not None and len(seq) >= config.min_seq_len:
                sequences[idx] = seq
            else:
                print(f"[!] Video descartado por secuencia corta o vacia: {sample.path}")
    finally:
        if detector is not None:
            detector.close()

    valid_indices = sorted(sequences.keys())
    if not valid_indices:
        raise ValueError("No se pudieron extraer secuencias validas.")

    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
    sequence_list = [sequences[idx] for idx in valid_indices]
    label_list = np.array([labels[idx] for idx in valid_indices], dtype=np.int64)
    metadata = [samples[idx] for idx in valid_indices]

    def translate(indices: list[int]) -> list[int]:
        return [remap[idx] for idx in indices if idx in remap]

    train_idx = translate(train_idx)
    val_idx = translate(val_idx)
    test_idx = translate(test_idx)
    if not train_idx or not test_idx:
        raise ValueError("El split quedo vacio despues de filtrar secuencias invalidas.")

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

    train_loader = DataLoader(SequenceDataset(train_idx), batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = (
        DataLoader(SequenceDataset(val_idx), batch_size=config.batch_size, shuffle=False, collate_fn=collate)
        if val_idx else None
    )
    test_loader = DataLoader(SequenceDataset(test_idx), batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiLSTMClassifier(FEATURE_VECTOR_LENGTH, config.hidden_size, len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    use_validation = val_loader is not None

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

        if use_validation:
            val_acc = _evaluate(model, val_loader, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            metric_text = f" - val acc: {val_acc:.4f}"
        else:
            metric_text = ""
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {total_loss:.4f}{metric_text}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc = _evaluate(model, test_loader, device)
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
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "test_size": len(test_idx),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] Modelo guardado en: {config.output}")
    print(f"[OK] Resumen guardado en: {summary_path}")
    return config.output
