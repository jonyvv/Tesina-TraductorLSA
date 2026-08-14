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
│   ├── models/               # pesos entrenados (.joblib / .pt) — NO se versionan
│   └── requirements.txt
├── ml/                       # Captura de dataset + entrenamiento
│   ├── capture_dataset.py
│   ├── train.py                    # Random Forest vs. MLP (señas estáticas)
│   ├── train_dynamic_lstm.py       # LSTM (señas dinámicas, fase 2)
│   ├── extract_features.py         # LSA64 etapa 1: landmarks -> cache .npz
│   ├── train_lsa64.py              # LSA64 etapa 2: BiLSTM sobre el cache
│   ├── evaluate_loso.py            # validación leave-one-subject-out
│   ├── lsa64/                      # implementación de LSA64 (cache, folds, fit)
│   ├── tests/                      # tests de splits, cache y entrenamiento
│   ├── reports/                    # matrices de confusión y reportes .json
│   └── requirements.txt
├── frontend/                 # cliente: cámara + WebSocket + UI
│   ├── index.html              # estructura (semántica y accesible)
│   ├── styles.css              # tokens de diseño + temas claro/oscuro
│   └── app.js                  # captura, WebSocket, overlay y métricas
├── tests/                    # tests automáticos (pytest)
│   ├── test_features.py
│   ├── test_modelo_lse.py
│   ├── test_traductor_service.py
│   └── test_backend_e2e.py
├── data/
│   └── sign_language_dataset/  # muestras capturadas (JSON)
├── Dockerfile                # imagen del backend (sirve también el frontend)
├── requirements-dev.txt      # dependencias para correr los tests
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

#### LSA64: pipeline en dos etapas

Extraer los landmarks con MediaPipe es la parte cara (~130.000 inferencias sobre
frames de 1080p para los 3200 videos); entrenar la BiLSTM sobre esos landmarks
son segundos. Por eso las dos etapas están separadas: la extracción se paga una
sola vez y queda cacheada en un `.npz`.

```bash
# Etapa 1 — una sola vez (~46 min con 8 procesos en un Ryzen 7 5700)
python ml/extract_features.py --dataset-dir ".lsa64_cache/extracted/all"

# Etapa 2 — todas las veces que quieras, en segundos
python ml/train_lsa64.py --epochs 60 --hidden-size 256
python ml/train_lsa64.py --lr 5e-4 --batch-size 32
```

Opciones útiles:

- `--workers N` — procesos en paralelo para la extracción (por defecto, la mitad de los hilos).
- `--refresh-cache` — fuerza volver a extraer (necesario si cambiás `--frame-step` o `--max-frames`; el cache se invalida solo, igual).
- `--labels-map labels.json` — nombres legibles de las señas (`{"001": "Opaco", ...}`). Por defecto usa `clase_01`..`clase_64`.
- `--limit N` en `extract_features.py` — prueba rápida sobre N videos.

También podés seguir pasando `--dataset-archive` o `--download-url`: si no hay
cache, `train_lsa64.py` extrae y lo genera solo.

**Split por sujeto.** `train_val_test_split` agrupa por persona (7 sujetos a
train, 1 a val, 2 a test), y el entrenamiento verifica explícitamente que
ningún sujeto aparezca en más de un split. Si detecta fuga, lo avisa por
consola y lo deja registrado en el `.json` de resultados: una accuracy medida
sobre personas ya vistas en entrenamiento no sirve como resultado de la tesina.

#### Validación leave-one-subject-out

Un solo split con 2 sujetos de test da un número inestable: tres corridas sobre
datos prácticamente idénticos dieron 86,05 %, 78,55 % y 81,25 %. Ese spread no es
ruido de medición, es la varianza real de estimar con dos personas. Por eso el
resultado que se reporta sale de LOSO:

```bash
# Entrena 10 modelos (uno por sujeto dejado afuera) y reporta media ± desvío.
# Reusa el mismo cache y el mismo loop de entrenamiento que train_lsa64.py,
# así los números son comparables. No re-extrae nada.
python ml/evaluate_loso.py
```

Deja el reporte completo en `ml/reports/loso_lsa64.json`: accuracy por fold, por
clase y por sujeto, más las predicciones de los 10 folds agrupadas — cada muestra
del dataset queda predicha exactamente una vez por un modelo que nunca vio a esa
persona, así que sirven para armar la matriz de confusión del dataset entero.

**LOSO no guarda modelos**: es un protocolo de medición, no produce un `.pt` para
servir. El modelo que se sirve sale de `ml/train_lsa64.py`.

Última medición registrada (11/8/2026): **79,8 % ± 8,9 %** signer-independent.
El análisis de ese número está en `docs/PROXIMOS_PASOS.md`.

Esto deja `backend/models/modelo_lse.joblib` listo para que lo cargue el backend.

Si vas a usar LSA64, el flujo recomendado es organizar los videos por clase o
pasar un CSV de anotaciones con columnas `video,label,subject` y, de ser
posible, `split`. Ese script genera un modelo secuencial `.pt` pensado para
señas/palabras, no para el abecedario. Para letras sigue usando `ml/train.py`
con tus propias capturas.

La implementación de LSA64 quedó separada en `ml/lsa64/` para mantener el
entrypoint chico y dejar cada responsabilidad en su módulo. Si usás
`--dataset-archive` o `--download-url`, el script descarga/extráe y entrena en
un solo paso.

El backend ahora detecta automáticamente `backend/models/modelo_lsa64_lstm.pt`
si existe; si no, sigue usando `modelo_lse.joblib`. Para servir el `.pt`
necesitás instalar `torch` en el entorno del backend.

**Los pesos entrenados no se versionan.** `.gitignore` excluye `backend/models/*.pt`
y `.joblib`, y también el cache `.lsa64_cache/` (~33 MB). Quien clone el repo tiene
el código pero no los pesos: hay que reentrenar (etapa 1 + etapa 2) o pasarse el
`.pt` por fuera de git.

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

### 4. Docker (opcional, y lo que se usa para deployar)

Un solo contenedor sirve la API, el WebSocket y el frontend:

```bash
docker build -t traductor-lsa .
```

```bash
docker run --rm -p 8000:8000 traductor-lsa
```

El modelo de MediaPipe se descarga durante el build, así que la imagen queda
autocontenida. El modelo entrenado (`backend/models/modelo_lse.joblib`) se copia
si existe: se versiona junto al backend, nunca se entrena en producción.

En Render/Railway el puerto llega por la variable `$PORT` y el contenedor ya la
respeta. Antes de la entrega final hay que restringir `allow_origins` en
`backend/app/main.py` al dominio real (hoy está en `"*"` para desarrollo).

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Cubren el módulo de features (invarianza a traslación/escala, slots estables por
mano), la carga y validación de modelos, y el suavizado temporal con aislamiento
entre conexiones. Buena parte son **regresiones** de bugs reales encontrados en
el proyecto — cada test de ese tipo está marcado con `REGRESIÓN` en su docstring
y explica qué fallaba antes.

No requieren cámara, ni MediaPipe, ni un modelo entrenado: usan dobles de prueba
y datos sintéticos.

## Estado actual / qué falta

- [x] Captura con soporte de dos manos y normalización de landmarks (wrist-relative, invariante a escala — verificado por tests).
- [x] Backend FastAPI con WebSocket, siguiendo el diagrama de clases y de secuencia; arranca de forma robusta aunque falten el modelo de MediaPipe o el modelo entrenado (ver `/health`).
- [x] Entrenamiento comparado Random Forest vs. MLP para señas estáticas (abecedario), con split agrupado por sesión (`GroupShuffleSplit`/`GroupKFold`) para evitar fuga de datos.
- [x] Estado de suavizado por conexión, procesamiento fuera del event loop, y validación estricta del modelo al cargarlo.
- [x] Frontend con overlay del esqueleto de la mano, métricas en vivo, temas claro/oscuro, diseño responsive y reconexión automática del WebSocket.
- [x] Accesibilidad: estructura semántica, `aria-live` en el texto traducido, barras con `role="progressbar"`, foco visible, `prefers-reduced-motion`, áreas táctiles de 24 px y contraste WCAG AA (mínimo medido 4.93:1 en ambos temas).
- [x] Medición de latencia end-to-end en el cliente, separando cómputo del servidor de red + codificación.
- [x] Suite de tests automáticos (`tests/` para el backend, `ml/tests/` para el pipeline LSA64).
- [x] `Dockerfile` para deploy en un solo servicio.
- [x] Pipeline LSA64 separado en extracción cacheada + entrenamiento, con extracción paralela.
- [x] Etiquetado correcto de LSA64 (`CCC_SSS_RRR`) y split agrupado por sujeto, con chequeo de fuga.
- [x] Entrenamiento completo sobre LSA64, validado con leave-one-subject-out (10 folds, accuracy por sujeto y por clase). Ver `docs/PROXIMOS_PASOS.md`.
- [x] **Modelo dinámico (BiLSTM) servido en producción**: `ModeloLSE` delega en un adaptador según la extensión (`.joblib`/`.pt`) y `SesionTraduccion` acumula la ventana temporal por conexión.
- [ ] **Dataset propio real.** Sigue pendiente para las señas estáticas: hace falta capturar cada seña en **al menos 2 sesiones distintas** (idealmente con más de un sujeto). Con una sola sesión por seña, el split agrupado deja clases enteras fuera del entrenamiento y las métricas no significan nada — `ml/train.py` ahora lo detecta y aborta con un mensaje explicando el problema.
- [ ] Deploy efectivo en Render/Railway/GCP (la imagen está lista; falta publicarla y restringir CORS).
- [ ] Validación de usabilidad con usuarios reales de LSA.

**Nota sobre MediaPipe:** el prototipo de escritorio original usaba
`mediapipe.solutions.hands` (API legacy). Esa API ya no está disponible en los
paquetes de `mediapipe` actuales de PyPI, así que este proyecto usa la API
soportada (`mediapipe.tasks.vision.HandLandmarker`), que requiere el paso 0
de descarga del modelo descripto arriba. Ver el comentario al inicio de
`common/features.py` para el detalle completo.

Ver `docs/ARQUITECTURA.md` para el detalle de cada decisión de diseño y los
experimentos sugeridos para la sección de resultados de la tesina.
