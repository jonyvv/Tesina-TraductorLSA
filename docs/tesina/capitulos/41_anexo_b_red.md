# Anexo B — Detalle de la red neuronal

Este anexo complementa la sección 10.1 con las dimensiones de cada capa de la red, su implementación y el contenido del archivo del modelo.

## B.1 Capas y parámetros

Tabla: Capas de la red BiLSTM, dimensiones y cantidad de parámetros. {#tab:capas}

| Capa | Entrada | Salida | Parámetros | Cálculo |
|---|---|---|---|---|
| LSTM hacia adelante | (T, 138) | 128 | 137 216 | 4 × (138 × 128 + 128 × 128 + 2 × 128) |
| LSTM hacia atrás | (T, 138) | 128 | 137 216 | Ídem |
| Concatenación | 128 + 128 | 256 | 0 | |
| Dropout (p = 0,2) | 256 | 256 | 0 | Sólo activo durante el entrenamiento |
| Lineal | 256 | 64 | 16 448 | 256 × 64 + 64 |
| **Total** | | | **290 880** | |

El factor 4 de las capas LSTM corresponde a las tres compuertas y a la memoria candidata, cada una con su propia matriz de pesos para la entrada, otra para el estado oculto y sus sesgos (en PyTorch, dos vectores de sesgo por compuerta).

## B.2 Implementación

La red se define en `common/models/lsa64.py` y se utiliza tanto en el entrenamiento como en el backend:

```
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(features))
```

El método `forward` recibe el lote de secuencias completadas con ceros y la longitud real de cada una. `pack_padded_sequence` hace que la recurrencia se detenga en el último cuadro válido de cada secuencia, de modo que el relleno no contamina el estado final. `hidden[-2]` y `hidden[-1]` son los estados finales de la dirección hacia adelante y hacia atrás, respectivamente.

## B.3 Contenido del archivo del modelo

El archivo `backend/models/modelo_lsa64_lstm.pt` es un diccionario de PyTorch con los siguientes campos:

Tabla: Campos del archivo del modelo entrenado. {#tab:checkpoint}

| Campo | Contenido |
|---|---|
| `model_state_dict` | Pesos de la red |
| `label_classes` | Nombres de las 64 clases, en el orden de las neuronas de salida |
| `feature_version` | Versión del esquema de características (`v1`) |
| `feature_vector_length` | Longitud del vector por cuadro (138) |
| `hidden_size` | Unidades LSTM por dirección (128) |
| `frame_step` | Muestreo temporal usado al extraer (2) |
| `max_frames` | Longitud máxima de secuencia (120) |
| `min_seq_len` | Longitud mínima de secuencia (8); el backend la usa como tamaño de la ventana |
| `dataset`, `train_size`, `val_size`, `test_size`, `samples` | Trazabilidad: conjunto de datos, tamaños del reparto y lista de videos usados |

Con estos campos, el backend reconstruye la red y valida su compatibilidad sin depender de ninguna configuración externa.
