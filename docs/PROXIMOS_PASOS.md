# Próximos pasos y ajustes al anteproyecto

_Actualizado: 8 de agosto de 2026 — después de la primera corrida válida de entrenamiento sobre LSA64._

---

## 1. Dónde quedamos hoy

Primera corrida metodológicamente válida sobre LSA64:

| Métrica | Valor |
|---|---|
| Test accuracy | **86,05 %** (595 muestras, 64 clases) |
| Val accuracy | 81,75 % (mejor epoch 24 de 32, cortó por early stopping) |
| Split | 7 sujetos train / 1 val / 2 test — **sin fuga entre splits** |
| Clases con accuracy ≥ 80 % | 48 de 64 |
| Secuencias útiles | 2908 de 3200 (**292 descartadas, 9,1 %**) |
| Extracción de landmarks | 39 min, cacheada en `.lsa64_cache/features_lsa64.npz` |

Artefactos generados: `backend/models/modelo_lsa64_lstm.pt` y `modelo_lsa64_lstm.json`.

**Contexto importante:** ese 86 % es *signer-independent* (el modelo nunca vio a las dos
personas del test). La literatura sobre LSA64 en ese mismo régimen reporta desde 74,2 %
hasta ~91,7 % de media según el método; los valores de 99 % que se citan suelen ser
*signer-dependent* o con fusión de características. O sea: el punto de partida es
razonable y comparable. Ver §3.1.

### Bugs corregidos en esta sesión

1. **Etiquetado roto.** `infer_label_from_path` descartaba el primer campo de
   `CCC_SSS_RRR.mp4` — es decir, la seña — y armaba la etiqueta con sujeto+repetición.
   Producía 50 clases falsas en vez de 64. Cualquier resultado anterior no medía nada.
2. **Fuga de datos.** El sujeto salía `None`, así que el split agrupaba por archivo:
   la misma persona en train y en test.
3. **Sin caché de features.** Cada corrida repetía los 39 min de MediaPipe. Ahora la
   extracción está separada del entrenamiento (`ml/extract_features.py`).

15 tests cubren estos casos (`ml/tests/`).

---

## 2. Lo que sigue

Ordenado por valor para la tesina. Los tiempos asumen el caché ya generado.

### Prioridad alta — bloquean la sección de resultados

**2.1 Validación leave-one-subject-out (10 folds)** · ~20 min de cómputo

El 86 % sale de **un solo split con 2 sujetos de test**. Si el tribunal pregunta "¿y si
hubieran sido otros dos?", hoy no hay respuesta. LOSO entrena 10 veces dejando afuera a
una persona distinta y reporta **media ± desvío**, más la accuracy por sujeto. Es el
protocolo estándar en los trabajos sobre LSA64 y convierte un número suelto en un
resultado defendible.

**2.2 Matriz de confusión y análisis de errores** · ~10 min

`clase_08` da 0,00 con sus 10 muestras de test completas — no es falta de datos, es
confusión sistemática con otra seña. Hace falta la matriz para ver contra cuál, y para
detectar si los errores se concentran en las 22 señas bimanuales. Además es material
directo para el análisis cualitativo del capítulo de resultados.

**2.3 Baseline no neuronal (Random Forest)** · ~30 min

El README y el planteo del proyecto prometen la comparación explícita Random Forest vs.
red neuronal. Se puede correr sobre el mismo caché con features agregadas
(media/desvío/mín/máx por secuencia). Sin baseline, no hay forma de justificar que la
BiLSTM aporta algo.

### Prioridad media — mejoran el resultado

**2.4 Recuperar los 292 videos descartados** · ~45 min (requiere re-extraer)

La pérdida no está repartida pareja: `clase_40` conservó 16 de 50 muestras y `clase_14`
20 de 50. Son señas donde MediaPipe no engancha las manos. Probar
`min_hand_detection_confidence` en 0,4 y medir cuántas se recuperan. Hoy hay clases
entrenando con un tercio de los datos.

**2.5 Atacar el overfitting** · ~1 h

Loss de entrenamiento 0,064 contra val 0,79 — el modelo memoriza. A probar, en orden:
weight decay, subir dropout, aumentar `hidden_size` con más regularización, y data
augmentation barata sobre landmarks (espejado horizontal, jitter temporal, ruido
gaussiano). El espejado tiene un detalle: hay que intercambiar los slots de mano
izquierda/derecha del vector de features.

**2.6 Nombres reales de las señas** · ~30 min

Hoy las etiquetas son `clase_01`..`clase_64`. Para la demo y para las tablas de la
tesina hacen falta los nombres reales, tomados del paper de Ronchetti et al. (2016) y
cargados con `--labels-map labels.json`. Sin esto el frontend muestra "clase_08".

### Prioridad baja — cierre del proyecto

**2.7 Verificar el backend con el modelo nuevo.** Las etiquetas y la cantidad de clases
cambiaron respecto de lo que el backend cargaba antes. Hay que probar el flujo completo
cámara → WebSocket → predicción.

**2.8 Revisar `ml/train.py` y `ml/train_dynamic_lstm.py`.** No los toqué en esta sesión.
Dado el tipo de bug que apareció en el pipeline de LSA64, conviene auditar el split y el
etiquetado de esos dos también antes de reportar cualquier número que salga de ahí.

**2.9 Deploy** en Render/Railway con Docker, y **2.10 validación de usabilidad** con
usuarios reales de LSA — ambos ya figuran como pendientes en el README.

### Sugerencia de secuencia (agosto → noviembre 2026)

| Período | Foco |
|---|---|
| Agosto | 2.1, 2.2, 2.3 — cerrar la evidencia experimental del modelo de palabras |
| Septiembre | 2.4, 2.5, 2.6 — mejorar el modelo y decidir el alcance del abecedario (§3.2) |
| Octubre | 2.7, 2.9 — integración de punta a punta y deploy |
| Noviembre | 2.10 y redacción final |

---

## 3. Cambios necesarios en el anteproyecto

### 3.1 Críticos

**a) El abecedario no está en LSA64.**

El anteproyecto compromete en Alcances el "reconocimiento del abecedario de la Lengua de
Señas Argentina", pero LSA64 contiene **64 señas de palabras completas, ninguna letra**.
Las configuraciones manuales están en un dataset distinto del mismo grupo, **LSA16**
(16 configuraciones, 800 imágenes, 10 sujetos). Hoy `data/sign_language_dataset/` está
vacío, así que ese alcance no tiene datos detrás.

Tres salidas posibles, hay que elegir una y dejarla escrita:

1. Incorporar LSA16 y reformular el alcance como "16 configuraciones manuales de LSA"
   en vez de "el abecedario" — es lo más honesto y no requiere capturar nada.
2. Capturar dataset propio del abecedario, con el costo de tiempo y el requisito de
   consentimiento informado de los participantes.
3. Sacar el abecedario del alcance y concentrar la tesina en las 64 señas de palabras.

Sea cual sea, **el anteproyecto no puede quedar prometiendo "el abecedario" sin
especificar la fuente de datos.**

**b) La comparación con el 99,4 % de Mindlin necesita contexto.**

El anteproyecto cita ese número como antecedente. Si el trabajo de ustedes reporta 86 %,
el tribunal va a preguntar por la diferencia. La respuesta casi seguro es el protocolo de
evaluación: los valores altos en LSA64 suelen ser *signer-dependent* (la misma persona en
train y test) o usar fusión de características. En régimen *signer-independent* la
literatura va de 74,2 % a ~91,7 % de media.

Hay que: (i) verificar en la tesis de Mindlin qué protocolo usó y decirlo explícitamente
al citarla, y (ii) declarar el protocolo propio en Resultados esperados. Un 86 %
signer-independent bien explicado vale más que un 99 % sin protocolo.

**c) Falta un objetivo específico de evaluación.**

Los objetivos específicos son "diseñar la arquitectura / entrenar / desarrollar la web".
No hay ninguno sobre *evaluar*. Agregar algo como: "Evaluar el modelo mediante validación
independiente de sujeto y comparar contra una línea base no neuronal". Sin eso, los
capítulos 2.1 y 2.3 no responden a ningún objetivo declarado.

### 3.2 De coherencia con lo implementado

| Dice el anteproyecto | Está implementado | Acción |
|---|---|---|
| Frontend en **React.js** | `frontend/index.html`, HTML/JS puro | Actualizar, o justificar el cambio |
| FastAPI como **servicio REST** | WebSocket | Corregir: WebSocket es la decisión correcta para tiempo real |
| **TensorFlow / PyTorch** | PyTorch | Dejar solo PyTorch y sacar la doc de TensorFlow de la bibliografía |
| **Jupyter Notebook** para entrenar | Scripts (`ml/*.py`) | Actualizar |
| **GPU dedicada: Google Colab Pro** | Entrenamiento local | Ver punto siguiente |
| Anexo B: "capas convolucionales, recurrentes o Transformer" | BiLSTM sobre landmarks | Ya se puede concretar el anexo |
| Cronograma (Gantt) | Encabezado vacío | Completar — es lo primero que mira un tribunal |

**Sobre Colab Pro:** el recurso está de más para este enfoque y conviene explicarlo, porque
es una decisión técnica defendible. El cuello de botella no es la red (280K parámetros,
entrena en segundos) sino la extracción de landmarks con MediaPipe, que corre en CPU. Con
8 núcleos locales son 39 minutos; en la capa gratuita de Colab, con 2 vCPU, serían horas.
Reformular Recursos Tecnológicos como: entrenamiento local (Ryzen 7 de 8 núcleos, 32 GB
RAM, RTX 3050), y Colab solo como contingencia si se pasa a modelos sobre video crudo.
Ajustar también "2 notebooks con 8 núcleos y 16 GB", que no refleja el equipamiento real.

### 3.3 Recomendados

**Terminología.** El documento usa "sordomudas" varias veces. La comunidad sorda
considera el término inadecuado —"mudo" es impreciso, ya que la mayoría de las personas
sordas pueden vocalizar— y la forma aceptada es "personas sordas" o "comunidad sorda".
En una tesina cuya justificación central es la inclusión, es un detalle que un tribunal
puede señalar y que se corrige con buscar y reemplazar.

**Limitaciones a agregar:**

- Los resultados se miden sobre LSA64, grabado en condiciones controladas de laboratorio;
  la generalización a webcams y entornos reales es una limitación abierta.
- El preprocesamiento con MediaPipe descarta ~9 % de los videos por falta de detección de
  manos, con impacto desigual entre clases.
- El sistema reconoce señas aisladas, no lengua de señas continua (que incluye transiciones
  y gramática espacial). Ya está insinuado en "no se contemplará interpretación de
  estructuras gramaticales complejas", pero conviene decirlo con precisión.

**Resultados esperados.** Reemplazar "niveles adecuados de precisión" por una métrica y un
protocolo concretos, ahora que hay evidencia: por ejemplo, "accuracy signer-independent
sobre LSA64, evaluada con validación leave-one-subject-out y comparada contra una línea
base Random Forest".

**Consideraciones éticas.** Si se captura dataset propio (§3.1a, opción 2), hace falta
una sección sobre consentimiento informado y tratamiento de datos biométricos de los
participantes.

**Bibliografía a incorporar.** Hay un trabajo dedicado exactamente al problema de
evaluación que ustedes ahora están usando: *Investigating Signer-Independent Sign Language
Recognition on the LSA64 Dataset*. Es la cita natural para justificar el protocolo.
Agregar también la referencia a LSA16 si se toma esa opción para el abecedario.

---

## 4. Referencias consultadas

- [LSA64: An Argentinian Sign Language Dataset (arXiv)](https://arxiv.org/abs/2310.17429)
- [LSA64 — página del dataset, Facundo Quiroga](https://facundoq.github.io/datasets/lsa64/)
- [Investigating Signer-Independent Sign Language Recognition on the LSA64 Dataset](https://www.researchgate.net/publication/363174384_Investigating_Signer-Independent_Sign_Language_Recognition_on_the_LSA64_Dataset)
- [Enhancing Signer-Independent Recognition of Isolated Sign Language (MDPI Electronics)](https://www.mdpi.com/2079-9292/13/7/1188)
- [Sign Language Processing — recursos del grupo LIDI/UNLP](https://facundoq.github.io/sign_language.html)
