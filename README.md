# Traductor LSA — Proyecto base (tesina)

Implementación de la arquitectura cliente/backend/modelo definida en
`docs/ARQUITECTURA.md`, a partir del diagrama de arquitectura y del diagrama de
clases/secuencia provistos. Corrige los problemas detectados en el análisis
del prototipo de escritorio previo (ver ese informe): soporte de dos manos,
normalización de landmarks, separación real entre señas estáticas y
dinámicas, y comparación explícita Random Forest vs. red neuronal.

## Estructura

```
lsa-traductor/
├── common/                 # Extracción/normalización de landmarks (compartido)
│   └── features.py
├── backend/                 # FastAPI: WebSocket + servicio de inferencia
│   ├── app/
│   │   ├── main.py
│   │   ├── frame.py
│   │   ├── preprocesador.py
│   │   ├── modelo_lse.py
│   │   ├── prediccion.py
│   │   ├── traductor_service.py
│   │   └── websocket_handler.py
│   ├── models/               # acá se guarda el .joblib entrenado
│   └── requirements.txt
├── ml/                       # Captura de dataset + entrenamiento
│   ├── capture_dataset.py
│   ├── train.py                    # Random Forest vs. MLP (señas estáticas)
│   ├── train_dynamic_lstm.py       # LSTM (señas dinámicas, fase 2)
│   ├── reports/                    # matrices de confusión generadas
│   └── requirements.txt
├── frontend/
│   └── index.html            # cliente: cámara + WebSocket + UI
├── data/
│   └── sign_language_dataset/  # muestras capturadas (JSON)
└── docs/
    └── ARQUITECTURA.md       # documento de arquitectura formal
```

## Cómo correrlo (desarrollo local)

### 0. Descargar el modelo de MediaPipe (una sola vez)

Este proyecto usa la API **MediaPipe Tasks** (`HandLandmarker`), no la API legacy
`mediapipe.solutions` (esa API ya no viene incluida en los paquetes de
`mediapipe` que se instalan hoy desde PyPI — ver la nota al inicio de
`common/features.py` para el detalle). La API nueva requiere descargar
explícitamente el archivo del modelo:

```bash
pip install mediapipe --break-system-packages   # o dentro de un venv, sin la flag
python common/download_model.py
```

Esto descarga `common/models/hand_landmarker.task` (~7-8 MB) desde los
servidores de Google. Hace falta correrlo una sola vez; tanto `ml/capture_dataset.py`
como el backend (`backend/app/preprocesador.py`) lo usan automáticamente después.

Si tu red bloquea `storage.googleapis.com`, el script te va a mostrar la URL
para descargarlo manualmente desde el navegador.

### 1. Entorno para ML (captura + entrenamiento)

```bash
python -m venv .venv-ml
source .venv-ml/bin/activate   # Windows: .venv-ml\Scripts\activate
pip install -r ml/requirements.txt

# Capturar dataset (repetir por cada palabra/letra y sujeto/sesión)
python ml/capture_dataset.py --sujeto leandro --sesion 1 --luz "natural"

# Entrenar (Random Forest + MLP, comparados) sobre las muestras estáticas
python ml/train.py
```

Esto deja `backend/models/modelo_lse.joblib` listo para que lo cargue el backend.

### 2. Backend

```bash
python -m venv .venv-backend
source .venv-backend/bin/activate
pip install -r backend/requirements.txt

cd backend
uvicorn app.main:app --reload --port 8000
```

Con el backend corriendo, `http://localhost:8000` ya sirve también el
frontend (FastAPI monta `frontend/` como estático), así que no hace falta un
segundo servidor para probar de punta a punta.

### 3. Frontend (si se quiere servir aparte, ej. durante desarrollo del UI)

```bash
cd frontend
python -m http.server 5500
```

y ajustar `WS_URL` en `index.html` si el backend no corre en `localhost:8000`.

## Estado actual / qué falta

Validado con pruebas automáticas durante el armado de este scaffold (lógica de
features, entrenamiento de punta a punta con datos sintéticos, arranque del
backend, y la clase `TraductorService` con suavizado):

- [x] Captura con soporte de dos manos y normalización de landmarks (wrist-relative, invariante a escala — verificado).
- [x] Backend FastAPI con WebSocket, siguiendo el diagrama de clases y de secuencia; arranca de forma robusta aunque falten el modelo de MediaPipe o el modelo entrenado (ver `/health`).
- [x] Entrenamiento comparado Random Forest vs. MLP para señas estáticas (abecedario), con split agrupado por sesión (`GroupShuffleSplit`/`GroupKFold`) para evitar fuga de datos.
- [x] Frontend funcional (cámara + overlay + texto traducido).
- [ ] Modelo dinámico (LSTM) servido en producción — hoy solo se entrena (`train_dynamic_lstm.py`), falta integrarlo a `ModeloLSE`/`TraductorService`.
- [ ] Dataset propio real (este scaffold no incluye datos, hay que capturarlos).
- [ ] Deploy en Render/Railway/GCP con Docker.
- [ ] Validación de usabilidad con usuarios reales de LSA.

**Nota sobre MediaPipe:** el prototipo de escritorio original usaba
`mediapipe.solutions.hands` (API legacy). Esa API ya no está disponible en los
paquetes de `mediapipe` actuales de PyPI, así que este proyecto usa la API
soportada (`mediapipe.tasks.vision.HandLandmarker`), que requiere el paso 0
de descarga del modelo descripto arriba. Ver el comentario al inicio de
`common/features.py` para el detalle completo.

Ver `docs/ARQUITECTURA.md` para el detalle de cada decisión de diseño y los
experimentos sugeridos para la sección de resultados de la tesina.
