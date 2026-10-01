# Capítulo 4 — Visión por computadora y estimación de la pose de la mano

Para que una red neuronal reconozca una seña, primero hay que convertir el video en números. Este capítulo describe cómo se representa una imagen digital, compara los dos grandes enfoques para extraer información de las manos y presenta MediaPipe Hands, la herramienta que utiliza el sistema para obtener la geometría de las manos en cada cuadro.

## 4.1 Imagen digital y video

Una **imagen digital** es una matriz de píxeles. En una imagen en color, cada píxel se describe con tres valores —rojo, verde y azul (RGB)— entre 0 y 255. Una imagen de 1920 × 1080 píxeles, la resolución de los videos de LSA64, contiene por lo tanto más de seis millones de valores. Un **video** es una sucesión de imágenes, o **cuadros** (*frames*), capturadas a una frecuencia constante medida en cuadros por segundo (fps). LSA64 fue grabado a 60 fps (Ronchetti et al., 2016b), de modo que una seña de dos segundos ocupa unos 120 cuadros.

La **visión por computadora** es el área que estudia cómo extraer información útil de imágenes y video. Su desafío central es la distancia entre la representación cruda —millones de valores de intensidad— y el contenido semántico —"una mano con el índice extendido a la altura del mentón"—. La misma seña produce imágenes muy distintas según la iluminación, el fondo, la ropa, el color de piel, la distancia a la cámara y la posición dentro del encuadre.

## 4.2 Enfoques para el reconocimiento de señas

Los sistemas de reconocimiento de lengua de señas basados en video siguen, en líneas generales, uno de dos enfoques (Koller, 2020).

### 4.2.1 Enfoque basado en la apariencia

El modelo recibe directamente los píxeles —usualmente reducidos a una resolución menor— y aprende por sí mismo qué patrones visuales son relevantes. Las redes convolucionales 2D y 3D mencionadas en la sección 3.8 son el ejemplo típico. Este enfoque puede capturar todo lo que está en la imagen, incluidos rasgos que un esqueleto no representa, como la expresión facial. A cambio, requiere muchos datos para aprender a ignorar lo irrelevante —el fondo, la ropa, la iluminación—, demanda mucho cómputo y es sensible a cambios en las condiciones de grabación.

### 4.2.2 Enfoque basado en el esqueleto

Un modelo previo, entrenado sobre grandes volúmenes de datos para una tarea general, localiza en cada cuadro un conjunto de puntos de referencia o **landmarks** del cuerpo o de las manos —articulaciones y extremos de los dedos—. El clasificador de señas recibe únicamente las coordenadas de esos puntos.

Para este proyecto, el enfoque basado en el esqueleto ofrece ventajas decisivas:

- **Compacidad.** Un cuadro se reduce de seis millones de valores a unas decenas de coordenadas, lo que permite entrenar un clasificador pequeño, en segundos y sin GPU.
- **Invarianza a factores irrelevantes.** El fondo, la ropa, la iluminación y el color de piel quedan absorbidos por el detector, que ya fue entrenado para ignorarlos. El clasificador trabaja sólo con la geometría.
- **Transferencia.** El detector fue entrenado con cientos de miles de imágenes de manos que ningún proyecto académico podría reunir. Su conocimiento se reutiliza sin costo.
- **Privacidad y ancho de banda.** Las coordenadas no permiten reconstruir la imagen de la persona, y ocupan una fracción mínima del tamaño de un cuadro.

Su desventaja es que el clasificador queda limitado por la calidad del detector: si este no encuentra la mano en un cuadro, esa información se pierde por completo. El Capítulo 14 muestra que esta limitación es, de hecho, la principal fuente de variación en el desempeño del sistema.

## 4.3 MediaPipe

**MediaPipe** es un marco de trabajo de código abierto desarrollado por Google para construir procesos de percepción sobre video y audio (Lugaresi et al., 2019). Ofrece soluciones listas para usar —detección de rostros, estimación de la pose corporal, segmentación, seguimiento de manos— optimizadas para ejecutarse en tiempo real en dispositivos comunes, incluidos teléfonos y navegadores.

### 4.3.1 MediaPipe Hands

La solución de manos de MediaPipe, descripta por Zhang et al. (2020), estima la posición de 21 puntos de referencia por mano a partir de una imagen RGB común. Funciona como un proceso de dos etapas:

1. Un **detector de palmas** localiza las manos en la imagen completa y devuelve un recuadro orientado alrededor de cada una. Detectar la palma, un objeto relativamente rígido, es más fácil que detectar la mano completa con sus dedos articulados.
2. Un **modelo de landmarks** recibe el recorte de cada mano y predice las coordenadas de los 21 puntos. En video, el recuadro de un cuadro se usa para recortar el siguiente, de modo que el detector de palmas sólo se ejecuta cuando se pierde el seguimiento.

Para cada mano detectada, MediaPipe devuelve:

- Las coordenadas x e y de cada punto, normalizadas al intervalo [0, 1] respecto del ancho y el alto de la imagen.
- Una coordenada z que representa la profundidad relativa de cada punto respecto de la muñeca: valores menores indican puntos más cercanos a la cámara.
- La **lateralidad** (*handedness*): si se trata de una mano izquierda o derecha, junto con una puntuación de confianza.

La @fig:landmarks muestra la numeración de los 21 puntos: el 0 corresponde a la muñeca, y cada dedo aporta cuatro puntos, desde la articulación con la palma hasta la punta (pulgar: 1 a 4; índice: 5 a 8; medio: 9 a 12; anular: 13 a 16; meñique: 17 a 20).

![Los 21 puntos de referencia de la mano que estima MediaPipe Hands. Las líneas indican las conexiones del esqueleto que el sistema dibuja sobre el video.](figuras/fig_landmarks_mano.png){ancho=9 #fig:landmarks}

### 4.3.2 La API Tasks y la API heredada

MediaPipe ofreció durante años una interfaz de programación conocida como *Solutions* (`mediapipe.solutions.hands`), que es la que utilizan la mayoría de los tutoriales y la que usaba el prototipo de escritorio que sirvió como punto de partida de este proyecto. Al construir el sistema se comprobó que esa interfaz ya no está incluida en las versiones actuales del paquete distribuido por PyPI[^mp]: en la versión 0.10.33, el atributo `solutions` directamente no existe. El sistema utiliza en su lugar la interfaz soportada, **MediaPipe Tasks** (`HandLandmarker`), que ofrece el mismo modelo pero requiere descargar explícitamente su archivo de pesos (`hand_landmarker.task`).

El hallazgo es menor en lo técnico, pero ilustra un riesgo concreto de construir sobre prototipos o tutoriales de terceros sin auditarlos: el código puede dejar de funcionar no por un error propio, sino por cambios en las dependencias. Por eso el proyecto fija las versiones de sus dependencias y documenta el paso de descarga del modelo (ver el Anexo C).

[^mp]: Es un problema conocido y reportado en el repositorio oficial de MediaPipe (google-ai-edge/mediapipe, *issues* 6192, 6200 y 6204).

## 4.4 Lateralidad y orientación de la imagen

La lateralidad que informa MediaPipe se deduce de la imagen: el modelo decide si una mano es izquierda o derecha según cómo se ve. Esto tiene una consecuencia que no es evidente. Las cámaras web suelen mostrarse **espejadas** —como un espejo— porque así resulta natural para el usuario. Si una imagen se procesa espejada, la mano derecha de la persona aparece del lado derecho de la imagen y tiene el aspecto de una mano izquierda; MediaPipe la etiquetará en consecuencia.

Mientras la orientación sea la misma durante el entrenamiento y durante el uso, esto no es un problema: el modelo aprende con una convención y se usa con la misma. El problema aparece cuando las dos etapas usan orientaciones distintas, porque las manos quedan intercambiadas respecto de lo aprendido. La sección 9.7 relata cómo este desajuste se produjo efectivamente en el proyecto, por qué ningún control automático podía detectarlo y cómo se resolvió.

## 4.5 Limitaciones de la detección

El detector de manos no es infalible. Sus fallas más comunes son:

- **Desenfoque por movimiento.** En los movimientos rápidos, la mano aparece borrosa y el detector puede no encontrarla.
- **Oclusiones.** Cuando una mano tapa a la otra, o cuando la mano pasa por delante de la cara, los puntos ocultos se estiman con poca precisión o la mano no se detecta.
- **Manos fuera del encuadre o de perfil.** Algunas señas se realizan cerca del borde de la imagen o con la mano de canto respecto de la cámara.
- **Apariencias atípicas.** El detector fue entrenado con manos desnudas. En LSA64, las personas usaron guantes de colores fluorescentes para facilitar la segmentación (Ronchetti et al., 2016b), lo que podría afectar la detección de MediaPipe. Esta hipótesis no fue verificada en este trabajo, pero es consistente con la tasa de detección observada (sección 9.10).

Cuando el detector no encuentra una mano, el sistema lo registra explícitamente mediante un indicador de presencia (sección 9.4), en lugar de producir coordenadas en cero que el modelo podría confundir con una mano real.

