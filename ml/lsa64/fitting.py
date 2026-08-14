# -*- coding: utf-8 -*-
"""
Loop de entrenamiento de la BiLSTM, aislado del resto.

Vive aparte para que el entrenamiento normal (`training.py`) y la validacion
leave-one-subject-out (`evaluation.py`) usen exactamente el mismo codigo. Si
cada uno tuviera su propia copia, un cambio en uno haria que los numeros
dejaran de ser comparables sin que nadie se entere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.features import FEATURE_VECTOR_LENGTH
from common.models.lsa64 import BiLSTMClassifier

from .config import LSA64TrainingConfig


@dataclass
class FitResult:
    model: object
    test_accuracy: float
    y_true: list[int]
    y_pred: list[int]
    best_val_accuracy: float | None
    best_epoch: int | None
    epochs_corridas: int
    history: list[dict] = field(default_factory=list)


def set_seed(seed: int) -> None:
    """Fija TODAS las fuentes de aleatoriedad del entrenamiento.

    Sin esto, el seed solo determinaba el split: la inicializacion de los pesos,
    el dropout y el shuffle del DataLoader quedaban libres, y dos corridas sobre
    los mismos datos podian diferir varios puntos de accuracy. Un resultado que
    no se puede reproducir no sirve para la tesina.
    """
    import random as _random

    import torch

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluar(model, loader, device, collect: bool = False):
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


def _build_loaders(sequence_list, label_list, indices_por_split, config):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset

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
        lengths = torch.tensor([len(s) for s in sequences_batch], dtype=torch.long)
        padded = pad_sequence(sequences_batch, batch_first=True)
        return padded, lengths, torch.tensor(labels_batch, dtype=torch.long)

    train_idx, val_idx, test_idx = indices_por_split

    # El shuffle del DataLoader necesita su propio generador seedeado: no
    # alcanza con torch.manual_seed() global.
    generador = torch.Generator()
    generador.manual_seed(config.seed)

    train_loader = DataLoader(
        SequenceDataset(train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generador,
    )
    val_loader = (
        DataLoader(SequenceDataset(val_idx), batch_size=config.batch_size, shuffle=False, collate_fn=collate)
        if val_idx else None
    )
    test_loader = DataLoader(
        SequenceDataset(test_idx), batch_size=config.batch_size, shuffle=False, collate_fn=collate
    )
    return train_loader, val_loader, test_loader


def ajustar_modelo(
    sequence_list: list[np.ndarray],
    label_list: np.ndarray,
    n_clases: int,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    config: LSA64TrainingConfig,
    verbose: bool = True,
) -> FitResult:
    """Entrena la BiLSTM y devuelve el resultado sobre test.

    Reseedea al empezar, asi cada fold de LOSO arranca de las mismas condiciones
    iniciales y las diferencias entre folds se deben al sujeto, no al azar.
    """
    import torch

    set_seed(config.seed)
    train_loader, val_loader, test_loader = _build_loaders(
        sequence_list, label_list, (train_idx, val_idx, test_idx), config
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiLSTMClassifier(FEATURE_VECTOR_LENGTH, config.hidden_size, n_clases).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    epochs_sin_mejora = 0
    usa_validacion = val_loader is not None
    history: list[dict] = []
    epochs_corridas = 0

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

        epochs_corridas = epoch + 1
        epoch_loss = total_loss / max(len(train_loader), 1)

        if usa_validacion:
            val_acc = evaluar(model, val_loader, device)
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
        if verbose:
            print(f"Epoch {epoch + 1}/{config.epochs} - loss: {epoch_loss:.4f}{metric_text}")

        if usa_validacion and config.patience > 0 and epochs_sin_mejora >= config.patience:
            if verbose:
                print(
                    f"Early stopping en epoch {epoch + 1} "
                    f"(mejor: epoch {best_epoch}, val acc {best_val_acc:.4f})"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, y_true, y_pred = evaluar(model, test_loader, device, collect=True)
    return FitResult(
        model=model,
        test_accuracy=test_acc,
        y_true=y_true,
        y_pred=y_pred,
        best_val_accuracy=best_val_acc if usa_validacion else None,
        best_epoch=best_epoch if usa_validacion else None,
        epochs_corridas=epochs_corridas,
        history=history,
    )
