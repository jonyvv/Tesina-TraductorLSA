# Capítulo 5 — Reconocimiento automático de lengua de señas: estado del arte

El reconocimiento automático de lengua de señas es un área activa de investigación que combina visión por computadora, aprendizaje automático y lingüística. Este capítulo delimita el problema que aborda la tesina dentro del área, describe los principales conjuntos de datos y revisa los antecedentes específicos sobre la LSA, con especial atención a los protocolos de evaluación con que se obtuvieron sus resultados.

## 5.1 Tareas del área

Koller (2020) distingue, en su revisión cuantitativa del estado del arte, tres tareas de dificultad creciente:

- **Reconocimiento de señas aisladas.** Cada video contiene una única seña, con un comienzo y un final definidos, y el objetivo es clasificarla dentro de un vocabulario cerrado. Es la tarea que aborda esta tesina.
- **Reconocimiento continuo.** El video contiene una frase completa y el sistema debe producir la secuencia de señas, sin conocer dónde termina una y empieza la siguiente. Agrega el problema de la segmentación y de las transiciones entre señas, que modifican su forma.
- **Traducción.** El sistema debe producir una oración gramatical en la lengua oral, lo que implica resolver las diferencias de orden y estructura entre ambas lenguas. Camgöz et al. (2018) formularon este problema como traducción automática neuronal de extremo a extremo.

La distinción importa para dimensionar correctamente el alcance del sistema: reconocer 64 señas aisladas es un problema útil y no trivial, pero está lejos de la traducción de la LSA.

## 5.2 Conjuntos de datos

El progreso del área está ligado a la disponibilidad de conjuntos de datos públicos, cuya construcción es costosa: requiere grabar a muchas personas, cada una realizando muchas señas, y etiquetar cada video. La @tab:datasets resume los conjuntos más relevantes para este trabajo.

Tabla: Conjuntos de datos de lengua de señas relevantes para el trabajo. {#tab:datasets}

| Conjunto | Lengua | Tipo | Clases | Personas | Videos / imágenes | Referencia |
|---|---|---|---|---|---|---|
| LSA64 | Argentina | Señas aisladas | 64 | 10 | 3200 videos | Ronchetti et al. (2016b) |
| LSA16 | Argentina | Configuraciones manuales | 16 | 10 | 800 imágenes | Ronchetti et al. (2016a) |
| WLASL | Americana | Señas aisladas | 2000 | 119 | 21 083 videos | Li et al. (2020) |
| AUTSL | Turca | Señas aisladas | 226 | 43 | 38 336 videos | Sincan y Keles (2020) |

### 5.2.1 LSA64

**LSA64** fue desarrollado por el Instituto de Investigación en Informática LIDI de la Universidad Nacional de La Plata y constituye el primer conjunto de datos público de LSA orientado al aprendizaje automático (Ronchetti et al., 2016b). Contiene 3200 videos de 64 señas de uso cotidiano, realizadas por 10 personas no expertas, diestras, con 5 repeticiones de cada seña. Las grabaciones se hicieron en 1920 × 1080 píxeles a 60 cuadros por segundo, con fondo blanco, vestimenta negra y guantes de colores para facilitar el seguimiento de las manos. De las 64 señas, 42 se realizan con una mano y 22 con ambas. Una versión ampliada de su descripción fue publicada posteriormente por los mismos autores (Ronchetti et al., 2023). Se distribuye bajo una licencia Creative Commons de uso no comercial, compatible con un trabajo académico.

LSA64 es el conjunto utilizado para entrenar y evaluar el modelo de señas de esta tesina. Su tamaño es modesto para los estándares del aprendizaje profundo —50 videos por seña—, lo que favorece los enfoques basados en el esqueleto (sección 4.2.2).

### 5.2.2 LSA16

**LSA16**, del mismo grupo de investigación, contiene 800 imágenes de 16 configuraciones manuales de la LSA realizadas por 10 personas (Ronchetti et al., 2016a). Las configuraciones manuales son las formas básicas que adopta la mano y se relacionan con las letras del alfabeto manual, aunque no son lo mismo: 16 configuraciones no cubren el abecedario. Se menciona aquí porque es la alternativa pública más cercana para el reconocimiento de configuraciones estáticas, un alcance todavía en definición en este proyecto (ver el Capítulo 8).

## 5.3 Antecedentes sobre LSA64

La @tab:antecedentes resume los antecedentes revisados sobre LSA64. La columna de protocolo es la más importante: como se explicó en la sección 2.4.3, una cifra de exactitud no puede interpretarse sin saber si las personas de la prueba estaban o no en el entrenamiento.

Tabla: Resultados publicados sobre LSA64 según el protocolo de evaluación. {#tab:antecedentes}

| Trabajo | Entrada | Modelo | Protocolo | Exactitud |
|---|---|---|---|---|
| Mindlin (2021) | Video RGB, 128 × 128, 16 cuadros | Red convolucional 3D | Dependiente del sujeto (5 particiones aleatorias) | 99,4 % |
| Marais et al. (2022) | Video RGB | InceptionV3 + GRU | Dependiente del sujeto | 97,03 % |
| Marais et al. (2022) | Video RGB | InceptionV3 + GRU | Independiente del sujeto | 74,22 % |
| Akdag y Baykan (2024) | Video RGB y fusión de características | Red R3(2+1)D | Independiente del sujeto | 98,53 % |
| Esta tesina | Landmarks de MediaPipe (138 valores por cuadro) | BiLSTM | Independiente del sujeto (LOSO, 10 particiones) | 79,8 % ± 8,9 % |

### 5.3.1 Mindlin (2021)

La tesina de licenciatura de Mindlin (2021), dirigida por Franco Ronchetti en la Universidad Nacional de La Plata, entrenó modelos convolucionales 3D, convolucionales recurrentes y recurrentes sobre LSA64, con distintos preprocesamientos del video, y alcanzó un 99,4 % de exactitud con el mejor de ellos. Es el antecedente que se citó en el anteproyecto de este trabajo.

Al revisar su metodología se constató un dato que el anteproyecto omitía: la evaluación se realizó con validación cruzada de 5 particiones armadas tomando videos al azar de cada clase, y el propio autor aclara que "esta separación no es independiente del sujeto". Es decir, las personas de la prueba también estaban en el entrenamiento. El 99,4 % mide, por lo tanto, la capacidad de reconocer señas de personas conocidas, y no es comparable con el resultado de esta tesina, que mide el desempeño sobre personas nuevas. El mismo trabajo concluye que LSA64 "no representa un desafío para el estado del arte" bajo ese protocolo, y propone como trabajo futuro el uso de datos con desafíos del mundo real.

### 5.3.2 Marais et al. (2022)

El trabajo de Marais et al. (2022) es el antecedente más directamente relacionado con el problema de evaluación. Con una misma arquitectura —una red convolucional InceptionV3 para extraer características de cada cuadro seguida de una red recurrente GRU—, compararon ambos protocolos sobre LSA64: 97,03 % dependiente del sujeto frente a 74,22 % independiente del sujeto. La caída de más de 20 puntos, con el mismo modelo y los mismos datos, cuantifica la magnitud del optimismo que introduce mezclar personas entre entrenamiento y prueba.

### 5.3.3 Akdag y Baykan (2024)

Akdag y Baykan (2024) reportaron 98,53 % de exactitud independiente del sujeto sobre LSA64, con una red convolucional 3D que combina bloques residuales y convoluciones espaciotemporales separadas, operando sobre el video completo y fusionando distintas características. Es el mejor resultado independiente del sujeto encontrado, y muestra que el techo del problema es alto. El costo es un modelo pesado sobre video crudo, que requiere una GPU para la inferencia y no es compatible con una aplicación web en la capa gratuita de un servicio en la nube, que es una restricción explícita de este proyecto.

### 5.3.4 Posicionamiento del resultado propio

El resultado de esta tesina, 79,8 % ± 8,9 %, se ubica por encima del resultado independiente del sujeto de Marais et al. (2022) y claramente por debajo del de Akdag y Baykan (2024). La diferencia con este último se explica principalmente por la representación de entrada: el modelo propio no ve la imagen, sino sólo la geometría de las manos que entrega MediaPipe, y pierde la información de los cuadros donde el detector falla (sección 14.5). A cambio, el modelo tiene 290 880 parámetros, se entrena en segundos sobre CPU y permite la inferencia en tiempo real en un servidor modesto. La elección no es la de mayor exactitud posible, sino la de mejor equilibrio entre exactitud y viabilidad para el objetivo de una aplicación web accesible.

## 5.4 Aplicaciones existentes

Fuera del ámbito académico existen aplicaciones de reconocimiento de lengua de señas, en su mayoría para la Lengua de Señas Americana y basadas en el alfabeto manual, y diversas aplicaciones para el aprendizaje de la LSA que funcionan como diccionarios. En la revisión realizada para el anteproyecto no se encontraron aplicaciones accesibles desde el navegador que reconozcan señas de la LSA en tiempo real a partir de la cámara del dispositivo. Los trabajos académicos sobre LSA64 revisados se limitan a experimentos de laboratorio sin una interfaz de uso.

## 5.5 Brecha que aborda el trabajo

A partir de la revisión, se identifica la siguiente brecha: los trabajos existentes sobre LSA reportan resultados de clasificación sobre videos pregrabados, muchas veces con protocolos dependientes del sujeto, y no integran el modelo en un sistema utilizable. Esta tesina aporta:

1. Un modelo para las 64 señas de LSA64 evaluado con un protocolo independiente del sujeto (LOSO), con análisis por persona y por clase.
2. La integración de ese modelo en una aplicación web en tiempo real, que funciona con la cámara del dispositivo y está diseñada para ejecutarse en infraestructura gratuita.
3. La documentación, con evidencia, de los errores metodológicos que producen resultados optimistas en este tipo de sistemas y de cómo detectarlos.
