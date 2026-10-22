# Anexo C — Guía de instalación y ejecución

Este anexo resume los pasos para reproducir el sistema a partir del repositorio. El código fuente está disponible en el repositorio público https://github.com/jonyvv/Tesina-TraductorLSA, y la versión completa y actualizada de estas instrucciones, en su archivo `README.md`. [[PENDIENTE: agregar la dirección de la carpeta compartida "Fuentes del proyecto" que exige la facultad.]]

## C.1 Requisitos

- Python 3.12.
- Git.
- Una cámara web y un navegador moderno (Chrome, Edge o Firefox) para usar la aplicación.
- Opcionalmente, Docker.
- Para reentrenar el modelo: el conjunto de datos LSA64, que se descarga desde la página de sus autores.

## C.2 Descarga del modelo de MediaPipe

El archivo de pesos de MediaPipe no se versiona. Se descarga una única vez con:

```
pip install mediapipe
python common/download_model.py
```

El script guarda el archivo en `common/models/hand_landmarker.task`.

## C.3 Ejecución del backend y del cliente

```
python -m venv .venv-backend
.venv-backend\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --port 8000
```

Con el servidor en marcha, la aplicación queda disponible en `http://localhost:8000`. El servicio `http://localhost:8000/health` informa si el modelo de MediaPipe y el modelo entrenado se cargaron correctamente, y `http://localhost:8000/docs` muestra la documentación de la API.

## C.4 Ejecución con Docker

```
docker build -t traductor-lsa .
docker run --rm -p 8000:8000 traductor-lsa
```

La imagen descarga el modelo de MediaPipe durante la construcción e incluye el modelo entrenado, de modo que no requiere pasos adicionales.

## C.5 Reentrenamiento del modelo

```
python -m venv .venv-ml
.venv-ml\Scripts\activate
pip install -r ml/requirements.txt

# Etapa 1: extracción de características (una sola vez, ~47 minutos)
python ml/extract_features.py --dataset-dir ".lsa64_cache/extracted/all"

# Etapa 2: entrenamiento del modelo que se sirve (segundos)
python ml/train_lsa64.py --labels-map ml/lsa64_labels.json

# Validación leave-one-subject-out (~55 minutos)
python ml/evaluate_loso.py
```

El entrenamiento guarda el modelo en `backend/models/modelo_lsa64_lstm.pt` y su resumen en el archivo `.json` del mismo nombre. La validación guarda su reporte en `ml/reports/loso_lsa64.json`.

## C.6 Pruebas automatizadas

```
pip install -r requirements-dev.txt
pytest tests/ ml/tests/ -v
```

## C.7 Regeneración de las figuras y de este documento

```
python docs/tesina/figuras/generar_figuras.py
cd docs/tesina/build
npm install
node build.js
powershell -ExecutionPolicy Bypass -File postproceso.ps1
```

El último paso requiere Microsoft Word: actualiza los índices y exporta el documento a PDF.
