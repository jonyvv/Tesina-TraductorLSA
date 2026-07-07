from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    pack_padded_sequence = None


class BiLSTMClassifier(nn.Module if nn is not None else object):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        if nn is None:
            raise ImportError("PyTorch no esta instalado en este entorno.")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        if torch is None or pack_padded_sequence is None:
            raise ImportError("PyTorch no esta instalado en este entorno.")
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(features))

