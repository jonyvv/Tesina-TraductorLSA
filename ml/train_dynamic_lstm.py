# -*- coding: utf-8 -*-
"""
ml/train_dynamic_lstm.py

FASE 2 — Entrenamiento de un modelo secuencial (LSTM) para señas DINÁMICAS
(palabras con movimiento, guardadas por capture_dataset.py con type="dynamic").

Por qué un modelo aparte y no reusar train.py:
Una muestra dinámica es una SECUENCIA de vectores de features (una por frame),
no un vector único. Un RandomForest o un MLP tabular no puede representar esa
estructura temporal sin aplanar la secuencia (lo cual pierde la noción de
"orden" y fue justamente el bug del prototipo original). Un LSTM está diseñado
para esto: procesa la secuencia frame a frame y mantiene un estado interno que
resume el movimiento completo.

Requiere PyTorch (no está en ml/requirements.txt por defecto porque es una
dependencia pesada; instalar con `pip install torch --index-url ...` según tu
plataforma, ver https://pytorch.org/get-started/locally/).

Uso:
    python train_dynamic_lstm.py --epochs 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.features import FEATURE_VECTOR_LENGTH, FEATURE_VERSION  # noqa: E402

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "sign_language_dataset"
MODELS_DIR = Path(__file__).resolve().parents[1] / "backend" / "models"


def cargar_muestras_dinamicas():
    secuencias, labels, groups = [], [], []
    for path in DATASET_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                sample = json.load(f)
        except Exception:
            continue

        if sample.get("type") != "dynamic":
            continue
        if sample.get("feature_version") != FEATURE_VERSION:
            continue

        seq = sample.get("payload")
        if not isinstance(seq, list) or len(seq) == 0:
            continue
        if any(len(frame) != FEATURE_VECTOR_LENGTH for frame in seq):
            continue

        secuencias.append(np.array(seq, dtype=np.float32))
        labels.append(sample["label"])
        groups.append(f"{sample.get('sujeto', 'na')}_{sample.get('sesion', 'na')}")

    return secuencias, labels, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.nn.utils.rnn import pad_sequence
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print(
            "PyTorch no está instalado. Instalalo con:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "(o la variante con CUDA si tenés GPU)."
        )
        return

    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder

    secuencias, labels_raw, groups = cargar_muestras_dinamicas()
    if len(secuencias) == 0:
        print(f"No hay muestras dinámicas en {DATASET_DIR}. "
              f"Capturá algunas con capture_dataset.py (tecla R).")
        return

    print(f"Muestras dinámicas cargadas: {len(secuencias)}")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_raw)
    clases = list(label_encoder.classes_)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(secuencias, labels, groups))

    class SecuenciasDataset(Dataset):
        def __init__(self, idxs):
            self.idxs = idxs

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            idx = self.idxs[i]
            return torch.from_numpy(secuencias[idx]), int(labels[idx])

    def collate(batch):
        seqs, labs = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs])
        padded = pad_sequence(seqs, batch_first=True)  # (B, T_max, F)
        return padded, lengths, torch.tensor(labs)

    train_loader = DataLoader(SecuenciasDataset(train_idx), batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate)
    test_loader = DataLoader(SecuenciasDataset(test_idx), batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate)

    class LSTMClasificador(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)

        def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            # concatenar el último estado oculto de ambas direcciones
            h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
            return self.fc(h_cat)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMClasificador(FEATURE_VECTOR_LENGTH, args.hidden_size, len(clases)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, lengths)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, lengths, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x, lengths)
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = correct / max(total, 1)
            print(f"Epoch {epoch + 1}/{args.epochs} — loss: {total_loss:.4f} — test acc: {acc:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "clases": clases,
            "hidden_size": args.hidden_size,
            "input_size": FEATURE_VECTOR_LENGTH,
            "feature_version": FEATURE_VERSION,
        },
        MODELS_DIR / "modelo_lse_dinamico.pt",
    )
    print(f"[OK] Modelo LSTM guardado en {MODELS_DIR / 'modelo_lse_dinamico.pt'}")
    print(
        "\nNota: para servir este modelo desde el backend hay que agregar una "
        "variante de ModeloLSE que cargue .pt con PyTorch (o exportarlo a ONNX "
        "y usar onnxruntime, más liviano para producción). Ver docs/ARQUITECTURA.md."
    )


if __name__ == "__main__":
    main()
