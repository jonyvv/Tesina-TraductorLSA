# Capítulo 9 — Preparación de los datos y extracción de características

Este capítulo describe cómo se transforman los videos de LSA64 en las secuencias de vectores que recibe la red neuronal: la organización del conjunto de datos, el diseño del vector de características, la convención de orientación de la imagen, el proceso de extracción en dos etapas y el análisis de la calidad de la detección.

## 9.1 El conjunto LSA64

Las características generales de LSA64 se presentaron en la sección 5.2.1. Para el procesamiento interesan dos aspectos de su organización.

**Nomenclatura de los archivos.** Cada video se nombra con el patrón `CCC_SSS_RRR.mp4`, donde CCC es el número de seña (001 a 064), SSS el de la persona (001 a 010) y RRR el de la repetición (001 a 005). La etiqueta y la persona de cada video se obtienen, por lo tanto, del nombre del archivo. La función que realiza esa lectura (`parse_lsa64_stem`) fue el origen del error más grave del proyecto, descripto en la sección 9.11.

**Glosas.** La lista oficial de señas se publica sólo en inglés, aunque el conjunto sea de LSA. Para la aplicación y para este documento se tradujeron las glosas al castellano (ver el Anexo A), con dos criterios que van más allá de la traducción literal:

- **Localismos rioplatenses.** La seña 020, publicada como *Sweet milk*, es "Dulce de leche", y la 045, *Barbecue*, es "Asado".
- **Orden alfabético de la lista original.** Los tramos 029 a 043, 044 a 049 y 050 a 064 de la lista están ordenados alfabéticamente en castellano —no en inglés—, lo que indica que la lista original estaba en castellano y lo publicado es su traducción. Ese orden permitió resolver traducciones ambiguas: la seña 037, *Ship*, se ubica entre "Música" y "Ninguno", por lo que su glosa es "Nave" y no "Barco"; la 051, *Thanks*, se ubica entre "Aceptar" y "Apagar", por lo que es "Agradecer" y no "Gracias".

Además, se corrigió un error de una de las copias del conjunto disponibles en internet, que glosaba las señas 002 y 008 ambas como *Red*; la página del mantenedor del conjunto indica que la 008 es *Pink* ("Rosa"). Las glosas se mantienen en un archivo de configuración (`ml/lsa64_labels.json`) separado del entrenamiento, de modo que corregir una glosa no requiere volver a extraer las características (sección 10.7).

## 9.2 Muestreo temporal

Los videos de LSA64 tienen 60 cuadros por segundo, una resolución temporal mayor que la necesaria para describir el movimiento de una seña. La extracción toma uno de cada dos cuadros (`frame_step = 2`), lo que equivale a 30 cuadros por segundo, y limita cada secuencia a un máximo de 120 cuadros muestreados, es decir, 4 segundos; las secuencias más largas se truncan en ese punto.

Este muestreo tiene una consecuencia para el uso en tiempo real que se discute en la sección 14.8: el modelo aprende las señas a 30 cuadros por segundo, mientras que el cliente web envía, por defecto, 8 cuadros por segundo.

## 9.3 El vector de características

Cada cuadro se representa con un vector de **138 valores**. La @fig:vector muestra su estructura: dos bloques de 69 valores, uno para cada mano, en posiciones fijas.

![Estructura del vector de características (versión v1). Cada mano ocupa un bloque fijo de 69 valores: un indicador de presencia, las 63 coordenadas normalizadas de sus 21 puntos y los 5 ángulos de flexión de los dedos.](figuras/fig_vector_features.png){ancho=15 #fig:vector}

Cada bloque contiene:

- **1 indicador de presencia** (0 o 1), que vale 1 si la mano fue detectada en el cuadro.
- **63 coordenadas normalizadas**: las coordenadas x, y, z de los 21 puntos, después de la normalización descripta en la sección 9.5.
- **5 ángulos de flexión**, uno por dedo.

Si una mano no se detecta, su bloque completo vale cero, incluido el indicador de presencia. Este indicador permite al modelo distinguir entre "la mano no está" y "la mano está con sus puntos en el origen de coordenadas", dos situaciones que sin él serían idénticas.

## 9.4 Posiciones fijas por mano

El prototipo de escritorio tomaba "la primera mano detectada" en cada cuadro. MediaPipe no garantiza ningún orden entre las manos que detecta, de modo que, en una seña bimanual, la mano derecha podía aparecer primero en un cuadro y segunda en el siguiente. El mismo gesto producía vectores distintos según un detalle irrelevante del detector, y las señas bimanuales —22 de las 64 de LSA64— no podían reconocerse.

El sistema asigna cada mano a su bloque según la **lateralidad** que informa MediaPipe: el primer bloque es siempre la mano izquierda y el segundo, la derecha. Así, cada posición del vector tiene un significado estable en todos los cuadros y en todas las muestras.

## 9.5 Normalización

Las coordenadas que entrega MediaPipe están referidas a la imagen: una misma configuración de la mano produce coordenadas distintas si la mano está a la izquierda o a la derecha del encuadre, o más cerca o más lejos de la cámara. Para que el modelo aprenda la forma de la mano y no su posición en la imagen, las coordenadas se normalizan en dos pasos (función `normalize_landmarks` del módulo compartido):

1. **Traslación.** Se resta a todos los puntos la posición de la muñeca (punto 0), que pasa a ser el origen.
2. **Escala.** Se dividen todas las coordenadas por la distancia entre la muñeca y la base del dedo medio (punto 9), una medida de referencia del tamaño de la propia mano.

Después de la normalización, la misma configuración genera vectores prácticamente idénticos sin importar la distancia a la cámara ni el lugar del encuadre. Las pruebas automatizadas del módulo verifican estas dos invarianzas (sección 13.1).

Los **ángulos de flexión** se calculan sobre los puntos normalizados. Para cada dedo se toman tres puntos —la muñeca, la articulación intermedia y la punta— y se calcula el ángulo en la articulación intermedia a partir del producto escalar de los dos vectores que la unen con los extremos. Un dedo extendido produce un ángulo cercano a 180 grados (π radianes) y uno flexionado, un ángulo menor. Estos cinco valores resumen en forma explícita una información que la red podría deducir de las coordenadas, pero que así recibe directamente.

## 9.6 Variante con ubicación (v2)

La normalización respecto de la muñeca tiene un costo: al centrar cada mano en su propia muñeca, el vector conserva la **forma** de la mano pero descarta **dónde** está. Como se vio en la sección 1.3.1, la ubicación es uno de los parámetros formales de una seña, y dos señas con igual configuración y movimiento, realizadas a distinta altura, resultan indistinguibles en la versión v1 del vector.

Para medir el efecto de esa pérdida se implementó una variante experimental, **v2**, que agrega a cada bloque la posición de la muñeca sin normalizar: tres valores por mano, con lo que el vector pasa de 138 a 144 valores. Esta variante tiene una limitación conocida: la posición queda referida al encuadre de la cámara y no al cuerpo de la persona. En LSA64 eso es aceptable porque la cámara está fija y las personas se ubicaron sobre marcas en el piso, pero con una cámara web cada usuario se ubica distinto. Una versión utilizable en producción debería expresar la posición respecto de un punto del cuerpo —por ejemplo, los hombros o la nariz, obtenidos con el modelo de pose de MediaPipe—.

La versión del esquema forma parte de los metadatos de cada modelo, y el backend actual rechaza explícitamente un modelo v2, porque no genera esa variante. El resultado del experimento se presenta en la sección 14.6.

## 9.7 Convención de orientación de la imagen

La sección 4.4 explicó que MediaPipe deduce la lateralidad de cada mano a partir de la imagen, y que una imagen espejada intercambia izquierda y derecha. Como el vector tiene posiciones fijas por lateralidad, si el entrenamiento y la inferencia procesan imágenes con orientación opuesta, cada mano cae en el bloque contrario al aprendido.

Este desajuste se produjo efectivamente durante el desarrollo, en dos oportunidades:

1. El script de captura de datos propios espejaba la imagen antes de procesarla, como es habitual con las cámaras web. El cliente web, en cambio, mostraba el video espejado mediante una propiedad de CSS (`transform: scaleX(-1)`), pero esa propiedad sólo afecta la presentación: el cuadro que se dibujaba en el canvas y se enviaba al servidor salía **sin espejar**.
2. El pipeline de LSA64, desarrollado después, procesaba los videos sin espejar, mientras que la captura y la inferencia trabajaban con imágenes espejadas. Como 42 de las 64 señas se hacen con una sola mano, el modelo entrenado así predecía señas incorrectas al usarse desde el cliente.

El aspecto más llamativo es que **ningún control automático podía detectarlo**: los vectores tenían la longitud correcta, la versión del esquema coincidía y no se producía ninguna excepción. El único síntoma era una exactitud baja sin explicación aparente, que es el síntoma que uno tiende a atribuir al modelo o a la falta de datos.

La solución fue fijar una convención única y explícita en el módulo compartido, la constante `ESPEJADO_CANONICO`: **toda imagen se procesa espejada**. Se eligió esa orientación porque mantenía válidos los datos ya capturados y porque es la natural para el usuario, que se ve como en un espejo. El cliente espeja el cuadro en el canvas antes de codificarlo, y el pipeline de LSA64 espeja cada cuadro antes de pasarlo a MediaPipe. Además, la orientación pasó a formar parte de los metadatos del caché de características (sección 9.9), de modo que un caché generado con otra orientación se invalida automáticamente.

El episodio muestra un límite de la estrategia del módulo compartido: evita la duplicación del **código** de extracción, pero no protege contra diferencias en cómo se **prepara la entrada** antes de llegar a ese código. La mitigación fue documentar el contrato junto al esquema que protege y convertirlo, donde fue posible, en un dato verificable.

## 9.8 Datos propios para configuraciones estáticas

Para el reconocimiento de configuraciones estáticas se desarrolló un script de captura (`ml/capture_dataset.py`) que abre la cámara, detecta las manos y guarda cada muestra como un archivo JSON con su vector de características y sus metadatos: etiqueta, persona, sesión y condición de iluminación. El script de entrenamiento correspondiente (`ml/train.py`) agrupa las muestras por sesión al dividirlas, de modo que todas las muestras de una misma sesión de captura queden del mismo lado, y aborta con un mensaje explicativo si alguna clase tiene una única sesión, porque en ese caso el reparto dejaría clases enteras fuera del entrenamiento.

Como prueba de concepto se capturaron 120 muestras de dos letras (A y C), en dos sesiones, realizadas por uno de los autores. Esta cantidad permite verificar que el pipeline funciona de punta a punta, pero no constituye un conjunto de datos del que puedan extraerse conclusiones. [[PENDIENTE: según la decisión D1 del plan, completar con la captura del abecedario (varias personas y sesiones, con consentimiento informado) o incorporar LSA16, y reportar los resultados en el Capítulo 14.]]

## 9.9 Extracción en dos etapas

Procesar los 3200 videos de LSA64 implica ejecutar MediaPipe sobre más de 130 000 cuadros de alta resolución. En las primeras versiones, cada entrenamiento repetía esa extracción, de modo que probar un cambio en la red —que entrena en segundos— costaba casi una hora de espera. El pipeline se reorganizó en dos etapas independientes, que muestra la @fig:pipeline.

![Pipeline de datos en dos etapas. La extracción con MediaPipe se ejecuta una vez y se guarda en un caché; el entrenamiento y la evaluación leen el caché tantas veces como sea necesario.](figuras/fig_pipeline_dos_etapas.png){ancho=15 #fig:pipeline}

**Etapa 1: extracción** (`ml/extract_features.py`). Recorre los videos, extrae la secuencia de vectores de cada uno y guarda el resultado en un único archivo comprimido de NumPy (`.npz`), junto con la etiqueta y la persona de cada secuencia. La extracción se paraleliza en varios procesos, cada uno con su propia instancia de MediaPipe; con 8 procesos en un procesador de 8 núcleos tardó 47 minutos.

**Etapa 2: entrenamiento** (`ml/train_lsa64.py`, `ml/evaluate_loso.py`). Lee el caché y entrena o evalúa el modelo. Un entrenamiento completo tarda segundos; una validación LOSO de 10 particiones, alrededor de una hora.

### 9.9.1 Metadatos y compatibilidad del caché

Un caché es válido sólo para la configuración con que fue generado. Si se cambia el muestreo temporal, el esquema de características o la orientación, un caché anterior produciría datos incompatibles. Para evitar errores silenciosos, el caché guarda sus parámetros de generación —versión del formato, versión y longitud del vector, `frame_step`, `max_frames` y orientación— y, antes de usarlo, el entrenamiento los compara con la configuración pedida. Ante cualquier diferencia, el caché se rechaza con un mensaje que indica qué parámetro no coincide.

## 9.10 Calidad de la detección

No todos los cuadros tienen manos detectadas. La extracción guarda siempre la secuencia completa, incluidos los cuadros sin manos, y el filtrado se decide al entrenar. En la configuración oficial se descartan los cuadros donde no se detectó ninguna mano y, después, las secuencias que quedan con menos de 8 cuadros. Con ese criterio, de los 3200 videos se conservan **2862 secuencias**: 338 videos (10,6 %) se descartan por no tener suficientes cuadros con manos detectadas.

La pérdida no es uniforme. Por persona, la proporción de cuadros con manos detectadas —que se denomina **retención**— varía entre 58,5 % y 75,7 %. Por seña, la diferencia es aún mayor: algunas clases conservan casi todas sus muestras y otras pierden más de la mitad, como "Paciencia", que conserva 22 de sus 50 videos, u "Hombre", que conserva 23. Son señas en las que MediaPipe no logra detectar las manos con regularidad, posiblemente por oclusiones, movimientos rápidos o los guantes de color usados en la grabación (sección 4.5). La sección 14.5 muestra que la retención está asociada con la exactitud del modelo.

[[PENDIENTE (tarea M3 del plan): probar umbrales de detección más bajos y el modo VIDEO de MediaPipe, que usa seguimiento entre cuadros, para medir cuántas secuencias se recuperan.]]

## 9.11 Errores detectados en el pipeline de datos

Durante la revisión del pipeline de LSA64 se detectaron errores que invalidaban todas las mediciones previas. Se documentan aquí porque constituyen uno de los aprendizajes centrales del trabajo:

1. **Etiquetado incorrecto.** La función que obtenía la etiqueta descartaba el primer campo del nombre de archivo —el número de seña— y armaba la etiqueta con la persona y la repetición. El modelo se entrenaba con 50 clases artificiales en lugar de las 64 señas reales, y cualquier exactitud medida en esas condiciones no medía nada.
2. **Fuga de datos.** Por el mismo motivo, la persona de cada video no se identificaba y el reparto de los datos se hacía por archivo: la misma persona aparecía en entrenamiento y en prueba, lo que producía una evaluación dependiente del sujeto sin advertirlo (sección 2.4.2).
3. **Orientación inconsistente**, descripta en la sección 9.7.
4. **Entrenamiento no reproducible**, descripto en la sección 10.4.
5. **Ausencia de caché**, que multiplicaba el costo de cada experimento, resuelta con la extracción en dos etapas.

Cada uno de estos errores tiene hoy una prueba automatizada que lo reproduce (sección 13.1).

