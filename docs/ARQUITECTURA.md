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
| `Prediccion` | `backend/app/prediccion.py` | Value object etiqueta/confianza/estabilidad |
| `TraductorService` | `backend/app/traductor_service.py` | Orquesta preprocesado + modelo (sin estado por cliente) |
| `SesionTraduccion` | `backend/app/traductor_service.py` | *(agregada)* Suavizado + historial de UNA conexión |
| `WebSocketHandler` | `backend/app/websocket_handler.py` | Maneja conexiones y el loop de recepción/envío |

Nota sobre `SesionTraduccion` (clase agregada al diagrama): originalmente
`TraductorService` guardaba el buffer de suavizado y el historial como atributos
propios, tal como figura en el diagrama de clases. El problema es que ese
servicio se construye **una sola vez al arrancar la aplicación** y se comparte
entre todas las conexiones WebSocket, así que esos atributos eran estado global:
con dos usuarios conectados a la vez, los frames de uno alimentaban el buffer de
voto mayoritario del otro, y la llamada a `reiniciar_buffer()` al conectarse un
cliente nuevo le borraba el suavizado a quien ya estaba usando el sistema.

La separación deja a `TraductorService` sin estado mutable (solo preprocesador y
modelo, que sí son compartibles) y mueve todo lo que es "conversación con un
cliente" a `SesionTraduccion`, que el `WebSocketHandler` instancia por conexión.
Es un caso concreto de por qué un diagrama de clases correcto en lo estructural
puede esconder un problema de concurrencia: el diagrama no distingue entre
objetos de aplicación y objetos de sesión.

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
voto mayoritario** dentro de `SesionTraduccion.traducir()` (buffer de las
últimas N predicciones) antes de devolver el resultado. Es el mismo patrón que
ya usaba el prototipo de escritorio para estabilizar la predicción entre
frames — ahí sí estaba bien resuelto, y lo trasladamos tal cual.

Dos correcciones sobre ese suavizado, respecto de la primera implementación:

- **El buffer registra también los frames sin seña.** Antes solo se agregaban
  las predicciones válidas, así que el buffer nunca se vaciaba: al bajar las
  manos, el voto mayoritario seguía devolviendo la última seña con `valida:
  true` indefinidamente. Incorporando los frames vacíos al voto, el resultado
  decae solo cuando la persona deja de hacer la seña.

- **`confianza` y `estabilidad` son campos distintos.** El suavizado
  sobrescribía `confianza` con la proporción de votos del buffer, de modo que un
  mismo nombre significaba dos cosas según en qué punto del pipeline se lo
  mirara, y ambas se comparaban contra el mismo umbral. Hoy viajan separadas:
  `confianza` es la probabilidad que asigna el modelo y `estabilidad` es la
  consistencia temporal de la detección. Además de ser más correcto, da **dos
  métricas independientes** para la sección de resultados: permite distinguir
  "el modelo duda" de "el modelo está seguro pero la detección parpadea", que
  tienen causas y soluciones distintas.

Además, el `WebSocketHandler` ejecuta el pipeline con `run_in_threadpool`:
MediaPipe, OpenCV y scikit-learn son sincrónicos y bloqueantes, y llamarlos
directamente desde el handler `async` bloquea el event loop completo mientras
corren, serializando a todos los clientes conectados. Como contrapartida, el
acceso al `HandLandmarker` (que no es thread-safe) se serializa con un lock
dentro de `Preprocesador`.

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

### 4.1 Contrato de orientación de la imagen (espejado)

Los slots fijos por mano tienen una consecuencia que no es evidente y que
costó un bug real: **quién decide qué mano es "izquierda" es MediaPipe, a partir
de la imagen que recibe.** Si el pipeline de captura y el de inferencia le pasan
imágenes con orientación opuesta, la misma seña cae en el slot contrario al que
se usó para entrenar, y el vector queda espejado respecto de lo aprendido.

Eso era exactamente lo que pasaba:

- `ml/capture_dataset.py` aplicaba `cv2.flip(frame, 1)` → MediaPipe veía la
  imagen **espejada**.
- El frontend mostraba el video espejado con `transform: scaleX(-1)`, pero eso
  es una transformación de CSS, puramente visual: `drawImage` lee el frame
  original del elemento `<video>`, sin espejar. El backend recibía la imagen
  **cruda**.

Lo llamativo del caso es que **ningún chequeo existente podía detectarlo**: los
vectores tenían la longitud correcta, la `FEATURE_VERSION` coincidía, no se
lanzaba ninguna excepción. El único síntoma era accuracy baja sin explicación —
que es justo el síntoma que uno tiende a atribuir al modelo o a la falta de
datos, y no a un desajuste del preprocesamiento.

La solución fue fijar una convención única y explícita en `common/features.py`
(`ESPEJADO_CANONICO`): **la imagen se procesa siempre espejada**. Se eligió esa
orientación y no la cruda por dos razones: mantiene válidos los datasets ya
capturados, y es la orientación natural para el usuario (se ve como en un
espejo, y el modelo ve lo mismo que ve la persona).

Vale la pena mencionarlo en la metodología de la tesina: el módulo `common/`
resuelve la duplicación de *código* entre captura e inferencia, pero no alcanza
por sí solo cuando el desajuste está en cómo se **prepara la entrada** antes de
llegar a ese código compartido. Es un límite concreto de la estrategia
"un solo módulo compartido", y la mitigación —documentar el contrato en el mismo
lugar que el esquema que protege— es una decisión de diseño defendible.

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

### 6.1 Estructura del cliente

El cliente está separado en tres archivos por responsabilidad —
`index.html` (estructura), `styles.css` (presentación) y `app.js` (comportamiento)—
en vez del archivo único con todo embebido de la primera versión.

**Overlay de landmarks.** El backend devuelve, junto con la predicción, los
landmarks de cada mano en coordenadas normalizadas `[0, 1]`, y el cliente los
dibuja sobre un `<canvas>` superpuesto al video. Tres decisiones que importan:

- Se envían `x, y` únicamente (sin `z`) y redondeados a 4 decimales: con dos
  manos son 84 números, menos de 1 KB por frame, despreciable frente al JPEG.
- Las coordenadas están en el sistema del frame **ya espejado** (§4.1) y el
  canvas del overlay **no** lleva la transformación CSS que sí tiene el `<video>`,
  así que caen directamente sobre lo que el usuario ve, sin reinvertir la `x`.
- El esqueleto (qué puntos se unen con qué) lo sirve el backend en
  `/model/info` en lugar de estar duplicado en el JavaScript. Es la misma razón
  por la que existe `common/`: si el esqueleto cambiara, no puede quedar una
  copia desactualizada del otro lado de la red.

**Contrapresión.** El cliente no envía un frame nuevo si ya hay 2 sin
responder. Además de evitar que se acumule trabajo que el backend no puede
absorber, es lo que hace que la medición de latencia sea válida (ver §9.4).

**Accesibilidad.** No es un agregado cosmético en un proyecto cuyo objetivo es
justamente la accesibilidad, así que las decisiones están tomadas explícitamente:

| Decisión | Motivo |
|---|---|
| `aria-live="polite"` solo en el texto traducido | Cambia únicamente al confirmarse una seña. Las métricas se actualizan ~8 veces por segundo: en una región viva harían el lector de pantalla inutilizable. |
| El overlay es `aria-hidden` | Es decorativo: toda la información que transmite está también como texto en el panel de detección. |
| Barras con `role="progressbar"` y `aria-valuetext` | Un porcentaje leído como "72 por ciento" es más claro que un valor sin unidad. |
| Áreas táctiles de 24 px mínimo | WCAG 2.5.8 (AA). Los botones de ayuda miden 16 px por diseño y se agrandan con un pseudo-elemento transparente. |
| Contraste verificado, no estimado | Todos los pares de color superan 4.5:1 en ambos temas (mínimo medido: 4.93:1). |
| `prefers-reduced-motion` | Desactiva animaciones para quienes las configuraron así en el sistema. |

## 7. Cómo migrar de Random Forest a un modelo de Deep Learning

Gracias a que `ModeloLSE` es la única clase que sabe cómo está serializado el
modelo, migrar no implica tocar `TraductorService`, `WebSocketHandler` ni el
frontend:

1. Entrenar el nuevo modelo (`ml/train.py` ya entrena y compara MLP vs. RF; para CNN/LSTM ver `ml/train_dynamic_lstm.py` como punto de partida).
2. Exportar a un formato que `ModeloLSE.cargar()` sepa leer. Para mantener todo en `.joblib` con scikit-learn (MLP) no hace falta cambiar nada. Para PyTorch/TensorFlow, se agrega una rama en `cargar()`/`predecir()` (o una subclase `ModeloLSEtorch`) que cargue `.pt`/`.h5`/`.onnx` en vez de `.joblib`.
3. `backend/app/main.py` apunta `RUTA_MODELO` al nuevo archivo. Nada más cambia.

## 8. Deploy

Conforme a las limitaciones del anteproyecto (capa gratuita de Render/Railway/GCP):

- **Backend**: contenedor Docker (`Dockerfile` en la raíz) con `backend/requirements.txt`, expone `/health`, `/model/info` y `/ws/translate`. `main.py` ya sirve el frontend estático desde la misma app (`StaticFiles`), así que un solo servicio cubre todo — importante para no gastar dos slots de la capa gratuita.
- **Modelo de MediaPipe**: se descarga durante el *build*, no al arrancar. Si se bajara en el arranque, cada despliegue (y cada reinicio del contenedor, que en la capa gratuita ocurre seguido por inactividad) quedaría atado a que `storage.googleapis.com` esté accesible desde el entorno de producción.
- **Modelo entrenado**: se versiona junto al backend (`backend/models/modelo_lse.joblib`), no se entrena en producción.
- **Puerto**: Render y Railway lo inyectan por `$PORT`; el contenedor lo respeta y cae a 8000 en local.
- **Usuario**: el contenedor corre como usuario sin privilegios, no como root.
- **CORS**: restringir `allow_origins` al dominio real antes de la entrega final (hoy está en `"*"` para desarrollo).

## 9. Experimentos sugeridos para la tesina

1. Comparación Random Forest vs. MLP sobre el mismo split (`ml/train.py` ya genera ambos reportes y matrices de confusión en `ml/reports/`).
2. Ablación: con vs. sin normalización wrist-relative (comentar esa línea en `common/features.py`, re-entrenar, comparar F1).
3. Ablación: una mano vs. dos manos, si el vocabulario final incluye señas bimanuales.
4. Medición de latencia end-to-end. **Ya implementada** en el frontend: el panel
   "Rendimiento" muestra la latencia total (round-trip medido en el navegador),
   el cómputo del servidor (`ms_servidor`, que el backend informa en cada
   respuesta) y la diferencia entre ambos, que corresponde a red + codificación
   y decodificación JPEG. Los valores se promedian sobre las últimas 30
   respuestas. Separar las dos componentes es lo que permite decidir con datos
   si conviene optimizar el modelo o el transporte — y es exactamente el número
   que hace falta para evaluar la alternativa de §6.

   Detalle metodológico que conviene mencionar: el cliente limita a 2 los frames
   "en vuelo" (enviados sin respuesta). Sin ese tope, cuando el backend tarda
   más que el intervalo de envío la cola crece sin límite y la latencia medida
   deja de reflejar el tiempo de procesamiento: pasa a medir, sobre todo, el
   tiempo de espera en cola. Es un error de medición fácil de cometer y difícil
   de notar, porque el número resultante *parece* razonable al principio y
   empeora gradualmente.
5. Landmarks en servidor vs. en cliente (§6), con medición de ancho de banda real.
6. Si se llega a implementar el modelo dinámico: comparación LSTM vs. "aplanar secuencia + RandomForest" (para mostrar objetivamente por qué el enfoque tabular no alcanza).
