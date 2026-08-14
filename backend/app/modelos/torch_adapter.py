from __future__ import annotations

from pathlib import Path

import numpy as np

from common.models.lsa64 import BiLSTMClassifier

from .base import ModeloAdaptador


class TorchCheckpointAdapter(ModeloAdaptador):
    def __init__(self, ruta_modelo: Path):
        self.ruta_modelo = ruta_modelo
        self.model = None
        self._clases: list[str] = []
        self._ventana_inferencia = 8

    def cargar(self) -> dict:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "El modelo .pt requiere PyTorch en el entorno del backend. "
                "Instalalo con: pip install torch --index-url "
                "https://download.pytorch.org/whl/cpu"
            ) from exc

        # `weights_only=False` explícito: el checkpoint no guarda solo tensores,
        # también las etiquetas (`label_classes`, que son objetos de numpy) y los
        # hiperparámetros. PyTorch va a invertir el default en una versión
        # próxima, y ahí esto rompería sin el parámetro puesto a mano. Es seguro
        # porque el .pt lo genera `ml/train_lsa64.py` y viaja versionado en el
        # repo — nunca se carga un checkpoint de origen desconocido.
        checkpoint = torch.load(self.ruta_modelo, map_location="cpu", weights_only=False)
        self._clases = list(checkpoint["label_classes"])
        self._ventana_inferencia = int(checkpoint.get("min_seq_len", 8))
        hidden_size = int(checkpoint["hidden_size"])
        input_size = int(checkpoint["feature_vector_length"])
        self.model = BiLSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=len(self._clases))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        return checkpoint

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        secuencia = np.asarray(features, dtype=np.float32)
        if secuencia.ndim == 1:
            secuencia = secuencia[np.newaxis, :]
        return self.predict_proba_secuencia(secuencia)

    def predict_proba_secuencia(self, secuencia: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("Llamá a cargar() antes de predecir()")

        secuencia = np.asarray(secuencia, dtype=np.float32)
        if secuencia.ndim != 2:
            raise ValueError("La secuencia debe tener forma (T, F).")

        tensor = torch.tensor(secuencia, dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([secuencia.shape[0]], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(tensor, lengths)
            probas = torch.softmax(logits, dim=1)[0]
        return probas.cpu().numpy().astype(np.float32)

    @property
    def requiere_secuencia(self) -> bool:
        return True

    @property
    def ventana_inferencia(self) -> int:
        return self._ventana_inferencia

    @property
    def clases(self) -> list[str]:
        return self._clases
