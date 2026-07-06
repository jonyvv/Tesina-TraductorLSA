# Arquitectura del Traductor LSA — Documento base

Este documento formaliza la arquitectura propuesta (ver diagramas provistos) y
documenta las decisiones tomadas al implementarla, incluyendo las correcciones
aplicadas sobre los problemas detectados en el análisis del repositorio de
escritorio previo.

---

## 1. Vista de capas

```
┌──────────────────────────────────────────────────────────────┐
│                    CLIENTE · navegador web                     │
│  Cámara web (MediaDevices API) → captura JPEG cada ~120ms      │
│  → WebSocket → backend                                          │
│  ← WebSocket ← {seña, confianza, valida}                        │
│  UI: overlay sobre el video + texto traducido acumulado         │
└──────────────────────────────────────────────────────────────┘
                              │  binario (bytes JPEG) / JSON
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    BACKEND · FastAPI (Python)                  │
│                                                                   │
│  WebSocketHandler                                                │
│    conectar() / recibir_frame() / enviar_resultado()             │
│         │                                                        │
│         ▼                                                        │
│  TraductorService                                                 │
│    procesar(frame) → Prediccion cruda                             │
│    traducir(bytes) → suaviza (buffer voto mayoritario) → dict     │
│         │                              │                          │
│         ▼                              ▼                          │
│  Preprocesador                    ModeloLSE                       │
│    extraer_mano() (MediaPipe)      cargar() / predecir() / top_n()│
│    normalizar() (común)                                           │
└──────────────────────────────────────────────────────────────┘
                              │  carga al arrancar
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                 MODELO · entrenado offline (ml/)                │
│  Dataset propio (JSON por muestra, estático o secuencia)         │
│  → train.py (RandomForest + MLP, comparados)                     │
│  → train_dynamic_lstm.py (LSTM, fase 2, señas dinámicas)         │
│  → backend/models/modelo_lse.joblib (o .pt)                      │
└──────────────────────────────────────────────────────────────┘
```

Esta es exactamente la arquitectura de 3 capas del diagrama que compartiste
(cliente / backend / modelo), con una diferencia deliberada respecto al
esquema original: el **módulo `common/`**, que no aparecía en el diagrama pero
es necesario para que la extracción de landmarks sea *idéntica* entre el
pipeline de entrenamiento (`ml/`) y el backend de inferencia (`backend/`). Sin
esto, cualquier cambio futuro en cómo se calculan los landmarks tiene que
replicarse a mano en dos lugares, y si se olvida uno, el modelo entrenado deja
de coincidir con lo que el backend le manda a predecir — sin ningún error
visible, solo predicciones incorrectas. Es el bug más importante que
encontramos en el prototipo de escritorio original.

## 2. Vista de clases (implementación de tu diagrama UML)

El diagrama de clases que compartiste se implementó tal cual, con la única
adición del acoplamiento explícito a `common.features` en `Preprocesador` y
`ModeloLSE`:

| Clase (tu diagrama) | Archivo | Responsabilidad |
|---|---|---|
| `CamaraCaptura` | *(vive en el frontend, no en Python)* | Captura de video vía `MediaDevices`, en `frontend/index.html` |
| `Frame` | `backend/app/frame.py` | Wrapper de los bytes recibidos por WS + decodificación a array |
| `Preprocesador` | `backend/app/preprocesador.py` | MediaPipe + extracción/normalización de landmarks |
| `ModeloLSE` | `backend/app/modelo_lse.py` | Carga y sirve el modelo entrenado |
| `Prediccion` | `backend/app/prediccion.py` | Value object etiqueta/confianza |
| `TraductorService` | `backend/app/traductor_service.py` | Orquesta preprocesado + modelo + suavizado + historial |
| `WebSocketHandler` | `backend/app/websocket_handler.py` | Maneja conexiones y el loop de recepción/envío |

Nota sobre `CamaraCaptura`: en tu diagrama aparece como clase del lado
servidor, pero en el diagrama de secuencia el `Usuario` interactúa con el
`Browser`, que es quien tiene la cámara. La implementación sigue el diagrama de
secuencia: la captura de video vive en el navegador (JavaScript, en
`frontend/index.html`), y lo único que cruza la red son los frames ya
capturados. Esto es consistente con cómo funciona `MediaDevices` (es una API
del navegador, no se puede invocar desde Python).

## 3. Vista de secuencia (implementada)

El flujo implementado sigue exactamente tu diagrama de secuencia:

```
Usuario → Browser: hace seña
Browser → WS Handler: conectar()
WS Handler → Browser: ws_aceptada
loop cada ~120ms:
  Browser → WS Handler: enviarFrame(bytes)
  WS Handler → Traductor: traducir(frame)
  Traductor → Preprocesador: extraerMano(f)
  Preprocesador → Traductor: landmarks[] (vector normalizado)
  Traductor → Modelo: predecir(vector)
  Modelo → Traductor: Prediccion{etiqueta, confianza}
  Traductor → WS Handler: resultado dict (con suavizado aplicado)
  WS Handler → Browser: {"seña": "A", "confianza": 0.97, "valida": true}
  Browser → Usuario: mostrar "A"
Usuario → Browser: detener cámara
Browser → WS Handler: cerrar()
```

Diferencia respecto al diagrama original: agregamos un paso de **suavizado por
voto mayoritario** dentro de `TraductorService.traducir()` (buffer de las
últimas N predicciones) antes de devolver el resultado. Es el mismo patrón que
ya usaba el prototipo de escritorio para estabilizar la predicción entre
frames — ahí sí estaba bien resuelto, y lo trasladamos tal cual.

## 4. Esquema de features (`common/features.py`)

Vector de longitud fija = **138 valores**:

```
[ mano_izquierda (69) | mano_derecha (69) ]

cada bloque de 69 =
  1  presencia (0.0 / 1.0)
  63 landmarks normalizados (21 puntos × x,y,z, wrist-relative + escala invariante)
  5  ángulos de flexión por dedo
```

Decisiones que corrigen problemas del prototipo original:
- **Dos manos, slots estables por `handedness`** (no por orden de detección) → soporta señas bimanuales.
- **Normalización wrist-relative + invariante a escala** → la misma seña genera vectores similares sin importar distancia/posición respecto a cámara.
- **`FEATURE_VERSION`** validado al cargar el modelo en el backend, para que un desajuste de esquema falle explícitamente en el arranque, no en producción de forma silenciosa.

**Nota técnica adicional (detectada al implementar este scaffold, no en el
análisis original del repo):** el prototipo de escritorio usaba
`mediapipe.solutions.hands`, la API "legacy" de MediaPipe. Esa API **ya no
está disponible** en los paquetes de `mediapipe` que se instalan desde PyPI
en versiones recientes (comprobado con `mediapipe==0.10.33`: `mp.solutions`
directamente no existe, `AttributeError`). Por eso este proyecto usa la API
soportada actualmente, **MediaPipe Tasks** (`mediapipe.tasks.vision.HandLandmarker`),
que es funcionalmente equivalente pero requiere descargar un archivo de
modelo externo (`hand_landmarker.task`, ver `common/download_model.py`) en
vez de traerlo empaquetado. Vale la pena mencionar este hallazgo en la
sección de metodología de la tesina: es un ejemplo concreto de por qué
apoyarse en un prototipo de terceros sin auditar tiene costos de
mantenimiento que no son evidentes a primera vista.

## 5. Estático vs. dinámico: dos modelos, no uno forzado

El prototipo original intentaba meter señas estáticas (un vector) y dinámicas
(una secuencia) en el mismo esquema, lo cual rompía el entrenamiento en la
práctica (ver el análisis del repo previo). Acá se separan desde el diseño:

- **Estático** (abecedario): un vector de 138 valores por muestra → `ml/train.py` → RandomForest / MLP.
- **Dinámico** (palabras con movimiento): una secuencia de vectores de 138 valores → `ml/train_dynamic_lstm.py` → LSTM bidireccional.

El backend (`ModeloLSE`) hoy sirve el modelo estático. Servir también el
modelo dinámico requiere agregar una segunda instancia de `ModeloLSE` (o una
subclase) que mantenga una ventana deslizante de frames por sesión de
WebSocket antes de predecir — el `TraductorService` ya tiene la estructura
para extenderse así (el buffer de suavizado es conceptualmente parecido al
buffer que necesitaría el modelo dinámico).

## 6. Alternativa de arquitectura a evaluar: landmarks en el cliente

El diagrama que compartiste envía **frames de video** del navegador al
backend, y el backend corre MediaPipe del lado del servidor. Es la opción más
simple de implementar (es la que armamos acá) y es totalmente válida para el
alcance de la tesina.

Existe una alternativa que vale la pena documentar y, si el tiempo lo permite,
comparar empíricamente: correr MediaPipe directamente en el navegador con
`@mediapipe/tasks-vision` (JS/WASM), y enviar por WebSocket solo el **vector de
landmarks** (unos pocos KB/seg) en vez de frames JPEG completos.

| | Landmarks en servidor (actual) | Landmarks en cliente (alternativa) |
|---|---|---|
| Ancho de banda | Alto (video comprimido cada ~120ms) | Bajo (solo floats) |
| CPU del backend | Alta (MediaPipe corre por conexión) | Baja (solo inferencia del modelo) |
| Complejidad de implementación | Baja | Media (requiere JS/WASM) |
| Apto para capa gratuita (Render/Railway) | Limitado — puede saturar con pocos usuarios concurrentes | Mejor, dado el límite explícito del anteproyecto |

Documentar esta comparación (incluso con una medición real de latencia/ancho
de banda de ambos enfoques) es un experimento de bajo costo con valor
académico real para la sección de resultados de la tesina.

## 7. Cómo migrar de Random Forest a un modelo de Deep Learning

Gracias a que `ModeloLSE` es la única clase que sabe cómo está serializado el
modelo, migrar no implica tocar `TraductorService`, `WebSocketHandler` ni el
frontend:

1. Entrenar el nuevo modelo (`ml/train.py` ya entrena y compara MLP vs. RF; para CNN/LSTM ver `ml/train_dynamic_lstm.py` como punto de partida).
2. Exportar a un formato que `ModeloLSE.cargar()` sepa leer. Para mantener todo en `.joblib` con scikit-learn (MLP) no hace falta cambiar nada. Para PyTorch/TensorFlow, se agrega una rama en `cargar()`/`predecir()` (o una subclase `ModeloLSEtorch`) que cargue `.pt`/`.h5`/`.onnx` en vez de `.joblib`.
3. `backend/app/main.py` apunta `RUTA_MODELO` al nuevo archivo. Nada más cambia.

## 8. Deploy

Conforme a las limitaciones del anteproyecto (capa gratuita de Render/Railway/GCP):

- **Backend**: contenedor Docker con `backend/requirements.txt`, expone `/health`, `/model/info` y `/ws/translate`. `main.py` ya sirve el frontend estático desde la misma app (`StaticFiles`), así que un solo servicio cubre todo — importante para no gastar dos slots de la capa gratuita.
- **Modelo**: se versiona junto al backend (`backend/models/modelo_lse.joblib`), no se entrena en producción.
- **CORS**: restringir `allow_origins` al dominio real antes de la entrega final (hoy está en `"*"` para desarrollo).

## 9. Experimentos sugeridos para la tesina

1. Comparación Random Forest vs. MLP sobre el mismo split (`ml/train.py` ya genera ambos reportes y matrices de confusión en `ml/reports/`).
2. Ablación: con vs. sin normalización wrist-relative (comentar esa línea en `common/features.py`, re-entrenar, comparar F1).
3. Ablación: una mano vs. dos manos, si el vocabulario final incluye señas bimanuales.
4. Medición de latencia end-to-end (marca de tiempo en el frontend al enviar vs. al recibir resultado).
5. Landmarks en servidor vs. en cliente (§6), con medición de ancho de banda real.
6. Si se llega a implementar el modelo dinámico: comparación LSTM vs. "aplanar secuencia + RandomForest" (para mostrar objetivamente por qué el enfoque tabular no alcanza).
