# Guía de estudio — cómo entender este proyecto de nuevo

Para cuando pasó el tiempo y hay que reconstruir qué se hizo y por qué. Está ordenada:
cada nivel se apoya en el anterior. Si tenés poco tiempo, los niveles 0 a 4 son el núcleo
y alcanzan para defender el trabajo.

Una advertencia sobre el método: **no leas los archivos sueltos**. Lo difícil de recuperar
no es qué hace cada módulo —eso se lee en diez minutos— sino *por qué está así*. Por eso
cada nivel está armado alrededor de una pregunta, no de una carpeta.

---

## Nivel 0 — El mapa mental (15 min)

**Qué hace el sistema.** El navegador saca fotos de la cámara, se las manda a un servidor
Python por WebSocket, el servidor le pasa cada foto a MediaPipe para sacar la posición de
las manos, acumula esas posiciones en una ventana de tiempo, se la da a una red neuronal
que devuelve qué seña es, y el resultado vuelve al navegador y se muestra en pantalla.

**Las tres capas** (y qué archivo mirar para cada una):

| Capa | Dónde vive | Responsabilidad |
|---|---|---|
| Cliente | `frontend/app.js` | Cámara, envío de frames, overlay, métricas |
| Backend | `backend/app/` | Recibe frames, orquesta, responde |
| Modelo | `ml/` (offline) | Entrena y produce el `.pt` que el backend carga |

Leer: `docs/ARQUITECTURA.md`, secciones 1 a 3. Son el diagrama de capas, el de clases y
el de secuencia, con la explicación de en qué se apartaron del diagrama original y por qué.

**La pregunta que contesta este nivel:** ¿dónde cae cada cosa?

---

## Nivel 1 — El camino de un frame (30 min)

Seguí un solo frame de punta a punta. Es el ejercicio que más rápido devuelve el modelo
mental completo.

1. `frontend/app.js` — la cámara, el canvas, el JPEG que se manda. Prestá atención a la
   **contrapresión**: el cliente no manda un frame nuevo si ya hay 2 sin responder.
2. `backend/app/websocket_handler.py` — recibe los bytes. Fijate que el trabajo pesado se
   saca del event loop con `run_in_threadpool`.
3. `backend/app/preprocesador.py` — MediaPipe convierte la imagen en 21 puntos por mano.
4. `common/features.py` — esos puntos se convierten en el vector de números que come el
   modelo. **Este es el archivo más importante del proyecto** (nivel 2).
5. `backend/app/traductor_service.py` — acumula la ventana temporal y llama al modelo.
   Mirá `SesionTraduccion`: hay una por conexión, y el porqué está en el nivel 6.
6. `backend/app/modelo_lse.py` + `backend/app/modelos/` — carga el `.pt` o el `.joblib`
   según la extensión y predice.

**La pregunta que contesta:** ¿qué le pasa a una imagen desde que sale de la cámara hasta
que vuelve convertida en una palabra?

---

## Nivel 2 — Las features: el corazón del asunto (45 min)

Todo el sistema se apoya en una decisión: **el modelo no ve imágenes, ve 138 números por
frame**.

```
[ mano_izquierda (69) | mano_derecha (69) ]

cada bloque de 69 =
   1  presencia (0.0 / 1.0)
  63  landmarks normalizados (21 puntos × x,y,z)
   5  ángulos de flexión por dedo
```

Leer `common/features.py` entero, con el comentario del encabezado. Lo que hay que poder
explicar:

- **Por qué landmarks y no píxeles.** La red no tiene que aprender a *encontrar* la mano,
  solo a clasificar la forma. Con 3200 videos, aprender a encontrarla sería imposible.
- **Por qué se normaliza respecto de la muñeca y de la escala.** Para que la misma seña dé
  un vector parecido sin importar a qué distancia de la cámara esté la persona.
- **Qué se pierde al normalizar.** La *ubicación* de la mano, que en una lengua de señas es
  un parámetro fonológico: dos señas con la misma configuración y el mismo movimiento
  hechas a distinta altura quedan indistinguibles. Esto explica buena parte de las
  confusiones del modelo. La variante v2 (`--con-posicion`) agrega la muñeca sin normalizar
  justamente para probar esto; se midió y no mejoró.
- **Slots fijos por mano.** La izquierda siempre va en el primer bloque, no según el orden
  en que MediaPipe las detecte. Si no, una seña bimanual daría vectores distintos según el
  capricho de la detección.
- **El contrato de espejado** (§4.1 de `ARQUITECTURA.md`). Lo cuento en el nivel 6 porque
  es una de las mejores historias para la defensa.

**La pregunta que contesta:** ¿qué es exactamente lo que aprende el modelo?

---

## Nivel 3 — El modelo y el pipeline de dos etapas (45 min)

**Por qué una red recurrente.** Una seña de palabra es una *secuencia*: 20 o 40 frames,
cada uno con sus 138 números. Un clasificador tabular necesita entrada de largo fijo; una
LSTM come secuencias de largo variable. "Bi" porque lee la secuencia en los dos sentidos,
lo que ayuda cuando el final de la seña desambigua el principio.

**Por qué dos etapas separadas** (`ml/extract_features.py` y luego `ml/train_lsa64.py`):
sacar los landmarks de los 3200 videos con MediaPipe son ~45 minutos de CPU; entrenar la
red sobre esos landmarks son segundos. Si estuvieran juntos, cambiar el learning rate
costaría 45 minutos. Separados, la extracción se paga una vez y queda cacheada en un
`.npz`.

Leer, en este orden:
1. `README.md`, sección "LSA64: pipeline en dos etapas".
2. `ml/lsa64/cache.py` — el formato del caché y, sobre todo, **qué parámetros lo
   invalidan**. Esa lista es una defensa contra entrenar con basura sin enterarse.
3. `common/models/lsa64.py` — la arquitectura de la red. Son 15 líneas.
4. `ml/lsa64/fitting.py` — el loop de entrenamiento, el early stopping.

**La pregunta que contesta:** ¿qué modelo es y por qué el pipeline está partido en dos?

---

## Nivel 4 — Cómo se mide (1 h) ← el nivel que más se pregunta

Este es el que distingue un trabajo serio de uno que informa un número lindo. Si vas corto
de tiempo, sacrificá otro, no este.

**El problema.** Un solo reparto de datos con 2 personas en test da un número que depende
de *quiénes* sean esas dos. En este proyecto, tres corridas sobre datos casi idénticos
dieron 86,05 %, 78,55 % y 81,25 %. Siete puntos y medio de diferencia sin que cambiara
nada real.

**La solución: leave-one-subject-out.** Se entrena 10 veces, dejando afuera a una persona
distinta cada vez, y se informa media ± desvío. Leer `ml/lsa64/evaluation.py`, que tiene
el razonamiento escrito en el docstring, y `ml/evaluate_loso.py`.

Conceptos que hay que poder explicar sin dudar:

- **Signer-dependent vs. signer-independent.** Si la misma persona aparece en
  entrenamiento y en test, el modelo puede estar memorizando a la persona en vez de
  aprender la seña. Los 99 % que se citan en la literatura de LSA64 son casi siempre
  signer-dependent. Es *la* pregunta que va a hacer el tribunal.
- **Fuga de datos (data leakage).** Por qué el split agrupa por persona y no por archivo,
  y por qué el código *verifica* que no haya fuga en vez de confiar.
- **Por qué el desvío es un resultado y no un defecto.** Hay 26 puntos entre la persona
  más fácil y la más difícil, sin outliers estadísticos. Eso dice algo real del problema.
- **Las dos pruebas pareadas** (`ml/comparar_modelos.py`): Wilcoxon sobre los 10 sujetos
  responde "¿la ventaja es consistente por persona?" y McNemar sobre las 2862 predicciones
  responde "¿uno acierta sistemáticamente más?". Pueden dar distinto, y de hecho dan
  distinto en los dos experimentos del trabajo. Hay que entender por qué: McNemar usa cada
  muestra y tiene mucho más poder; Wilcoxon con n=10 casi no detecta nada.

**La pregunta que contesta:** ¿por qué hay que creerle a estos números?

---

## Nivel 5 — Qué dio y qué significa (30 min)

| Resultado | Número | Dónde |
|---|---|---|
| BiLSTM, configuración de referencia | 79,8 % ± 8,9 % | `ml/reports/loso_lsa64.json` |
| BiLSTM con regularización | **81,7 % ± 7,9 %** | `ml/reports/loso_do40_fd30.json` |
| Línea base Random Forest | 78,2 % ± 7,5 % | `ml/reports/baseline_rf_loso.json` |

Las tres lecturas que hay que tener en la punta de la lengua:

1. **El orden temporal aporta poco.** Un bosque aleatorio que resume la secuencia en
   media/desvío/mín/máx —y por lo tanto *ignora el orden*— queda a 1,6 puntos de la red.
   Eso no invalida la BiLSTM, pero obliga a matizarla: con esta representación, la mayor
   parte de la información está en qué configuraciones aparecen, no en qué orden.
2. **La detección de manos es el factor limitante.** MediaPipe pierde entre el 24 % y el
   42 % de los frames según la persona, y esa tasa correlaciona 0,68 con la accuracy por
   persona. Mejorar la detección tiene más potencial que agrandar la red.
3. **La regularización funcionó y la grilla temporal no.** Dos experimentos, uno positivo
   y uno negativo, medidos con el mismo protocolo. El negativo también es resultado.

Leer `docs/PROXIMOS_PASOS.md` (§1) y el capítulo 14 de la tesina
(`docs/tesina/capitulos/26_cap14_resultados.md`), que es la versión redactada.

---

## Nivel 6 — Los bugs, que son material de tesina (45 min)

Esta parte es la que más impresiona en una defensa, porque muestra criterio y no solo
ejecución. Todos están documentados en `docs/PROXIMOS_PASOS.md` y en el capítulo 9.

- **El etiquetado roto.** Se armaba la etiqueta con sujeto+repetición en vez de la seña:
  producía 50 clases falsas en lugar de 64. Cualquier resultado anterior no medía nada.
- **La fuga de datos.** El sujeto salía `None`, así que el split agrupaba por archivo y la
  misma persona caía en train y en test.
- **El espejado** (§4.1 de `ARQUITECTURA.md`). La captura espejaba la imagen y el backend
  no, así que las manos caían en el slot contrario al del entrenamiento. Lo notable: **ningún
  chequeo podía detectarlo**. Los vectores tenían el largo correcto, la versión coincidía, no
  saltaba ninguna excepción. El único síntoma era accuracy baja sin explicación, que es lo
  que uno tiende a atribuir al modelo o a la falta de datos.
- **El estado compartido entre conexiones.** El buffer de suavizado vivía en el servicio,
  que se crea una vez para toda la aplicación: con dos usuarios conectados, los frames de
  uno alimentaban el voto del otro. Es un buen ejemplo de que un diagrama de clases
  correcto en lo estructural puede esconder un problema de concurrencia.
- **Los tests que no corrían.** Nueve tests end-to-end se salteaban solos porque el guard
  pedía un paquete que no estaba instalado. Pasaron meses sin que el backend se probara de
  punta a punta y la suite igual decía "todo verde".

El patrón común, y vale decirlo así en la defensa: **los errores que más costaron no
rompían nada; hacían que todo pareciera funcionar.**

---

## Teoría que conviene repasar aparte

Corta y concreta, solo lo que se usa:

- **Redes neuronales**: qué es una capa, función de activación, forward/backward,
  descenso por gradiente. Capítulo 3 de la tesina.
- **LSTM y por qué no una red común**: el problema de la memoria en secuencias, qué es una
  celda, qué aporta "bidireccional". Capítulo 3 y `docs/tesina/figuras/fig_celda_lstm.png`.
- **Sobreajuste**: cómo se ve en una curva de entrenamiento, qué es early stopping, qué es
  dropout, qué es data augmentation. Capítulo 2 y sección 14.3.
- **Random Forest**: árbol de decisión, por qué un bosque es mejor que un árbol. Capítulo 2.
- **Visión por computadora / MediaPipe**: qué es un landmark, por qué 21 puntos por mano.
  Capítulo 4.
- **WebSocket vs. REST**: por qué para tiempo real no sirve pedir-y-responder. Capítulo 6.
- **Estadística**: media y desvío, correlación, qué es un p-valor y qué *no* significa.
  Lo justo para defender Wilcoxon y McNemar.

---

## Preguntas que probablemente haga el tribunal

| Pregunta | Dónde está la respuesta |
|---|---|
| "Mindlin reporta 99,4 % y ustedes 80 %, ¿por qué?" | Protocolo: el de él es signer-dependent. Nivel 4 y §5.3.1 de la tesina |
| "¿Cómo saben que el modelo no memorizó a las personas?" | LOSO + la verificación de fuga en el código. Nivel 4 |
| "¿Por qué una red y no algo más simple?" | La línea base: la red gana, pero poco, y lo decimos. §14.7.1 |
| "¿Por qué 64 señas y no el abecedario?" | LSA64 no tiene letras; decisión de alcance documentada. `PROXIMOS_PASOS.md` §3.1 |
| "¿Esto anda en la realidad?" | Honestamente: todavía no se midió en vivo. §14.8.2 es la limitación declarada |
| "¿Por qué el desvío es tan alto?" | Retención de MediaPipe por persona, r = 0,68. Nivel 5 |
| "¿Qué harían con más tiempo?" | Mejorar la detección (M3) y cerrar la brecha con tiempo real (M5) |

La última fila importa: tener lista una respuesta concreta a "¿qué falta?" se lee como
dominio del tema, no como debilidad.

---

## Orden sugerido y tiempos

| Sesión | Qué | Tiempo |
|---|---|---|
| 1 | Niveles 0 y 1: el mapa y el camino de un frame | 45 min |
| 2 | Nivel 2: features, con `common/features.py` abierto | 45 min |
| 3 | Niveles 3 y 5: el modelo y los resultados | 1 h 15 |
| 4 | Nivel 4: la medición, despacio | 1 h |
| 5 | Nivel 6 + preguntas del tribunal | 45 min |
| 6 | Leer la tesina entera de corrido, ya con todo lo anterior | 2 h |

**Un truco que funciona mejor que releer:** después de cada sesión, escribí en un párrafo
y sin mirar qué entendiste. Donde no te salga la explicación, ahí está el agujero. Es la
misma lógica que usamos en el proyecto: la intuición que no se puede medir, no cuenta —y
ya nos pasó una vez de escribir una explicación linda que, al medirla, resultó falsa
(§14.7.2).
