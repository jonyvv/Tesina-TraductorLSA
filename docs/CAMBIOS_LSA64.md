# Cambios realizados hoy

Este resumen documenta la rama `Modulo-LSA64` y los cambios que hicimos para
mantener una arquitectura limpia, modular y fácil de extender.

## 1. Entrenamiento LSA64 separado por módulos

Se reemplazó un script monolítico por un paquete chico en `ml/lsa64/`:

- `config.py`: constantes y configuración de entrenamiento.
- `data.py`: carga de videos, inferencia de etiquetas y split.
- `source.py`: descarga/extracción de dataset.
- `training.py`: loop de entrenamiento.
- `model.py`: reutiliza la arquitectura del `BiLSTM`.

Ejemplo de la separación:

```python
# ml/train_lsa64.py
from lsa64.config import LSA64TrainingConfig
from lsa64.source import resolve_dataset_root
from lsa64.training import train_lsa64_model
```

Explicación:
- `train_lsa64.py` quedó como entrypoint liviano.
- Toda la lógica pesada se mueve a módulos pequeños, lo que facilita pruebas,
  mantenimiento y cambios futuros.

## 2. Entrada flexible del dataset

Agregamos tres formas de entrada para no obligarte a preparar todo manualmente:

```bash
python ml/train_lsa64.py --dataset-dir "C:\ruta\LSA64"
python ml/train_lsa64.py --dataset-archive "C:\ruta\LSA64.zip"
python ml/train_lsa64.py --download-url "https://tu-url-directa/LSA64.zip"
```

Fragmento clave:

```python
dataset_root = resolve_dataset_root(
    dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
    dataset_archive=Path(args.dataset_archive) if args.dataset_archive else None,
    download_url=args.download_url,
    work_dir=Path(args.work_dir),
)
```

Explicación:
- Podés entrenar desde carpeta, ZIP o URL directa.
- El script descarga/extráe y entrena en un solo paso.

## 3. Modelo compartido entre entrenamiento e inferencia

La arquitectura `BiLSTM` se movió a un módulo común:

```python
# common/models/lsa64.py
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
```

Explicación:
- La misma clase se usa para entrenar y para inferir.
- Evita duplicación y reduce el riesgo de desalineación entre ML y backend.

## 4. Backend con adaptadores limpios

`ModeloLSE` dejó de conocer detalles de serialización y delega en adaptadores:

```python
# backend/app/modelo_lse.py
self.adaptador = crear_adaptador_modelo(path)
data = self.adaptador.cargar()
```

Y la selección del formato se resuelve por extensión:

```python
# backend/app/modelos/factory.py
if path.suffix.lower() == ".joblib":
    return SklearnJoblibAdapter(path)
if path.suffix.lower() == ".pt":
    return TorchCheckpointAdapter(path)
```

Explicación:
- `.joblib` sigue sirviendo para el modelo clásico.
- `.pt` habilita el modelo LSA64 sin ensuciar la fachada principal.
- `TraductorService` sigue siendo el orquestador, no el dueño del formato.

## 5. Ventana temporal para LSA64

Cuando el backend carga un `.pt`, `TraductorService` acumula una secuencia de
frames antes de predecir:

```python
if self.modelo.requiere_secuencia:
    self._buffer_secuencia.append(resultado.vector)
    if len(self._buffer_secuencia) < self._buffer_secuencia.maxlen:
        return Prediccion.vacia(umbral=self.modelo.umbral_confianza)
    secuencia = np.asarray(list(self._buffer_secuencia), dtype=np.float32)
    return self.modelo.predecir_secuencia(secuencia)
```

Explicación:
- Esto permite usar LSA64 como seña/palabra dinámica.
- La lógica temporal no se mete en el frontend ni en la clase del modelo
  clásico; vive donde corresponde.

## 6. Documentación y defaults

También actualizamos la documentación para reflejar el flujo nuevo:

- `README.md`: comandos de entrenamiento y uso del dataset.
- `docs/ARQUITECTURA.md`: backend con soporte para `.joblib` y `.pt`.
- `backend/app/main.py`: detecta automáticamente `modelo_lsa64_lstm.pt` si existe.

Comportamiento actual:

```python
RUTA_MODELO = Path(
    os.getenv("LSA_MODEL_PATH")
    or (MODELS_DIR / "modelo_lsa64_lstm.pt"
        if (MODELS_DIR / "modelo_lsa64_lstm.pt").exists()
        else MODELS_DIR / "modelo_lse.joblib")
)
```

Explicación:
- Si existe el modelo LSA64, el backend lo toma solo.
- Si no, sigue funcionando con el modelo clásico sin romper nada.

## 7. Resumen corto

- Separamos el pipeline LSA64 en módulos pequeños.
- Agregamos carga desde carpeta, ZIP o URL directa.
- Compartimos la arquitectura del modelo entre ML y backend.
- Mantuvimos el backend limpio con adaptadores.
- Dejamos el sistema listo para usar LSA64 sin rearmar todo a mano.

