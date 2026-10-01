# Capítulo 12 — Cliente web

El cliente es la parte del sistema con la que interactúan las personas. Este capítulo describe su estructura, el ciclo de captura y envío de cuadros, la presentación de los resultados, las métricas de desempeño que calcula y las decisiones de accesibilidad.

## 12.1 Estructura

El cliente es una página web de una sola pantalla, separada en tres archivos por responsabilidad:

- `frontend/index.html`: la **estructura**, con marcado semántico (encabezado, secciones con título, controles agrupados) y los atributos de accesibilidad.
- `frontend/styles.css`: la **presentación**, organizada a partir de variables de diseño —colores, espaciados, tipografías— que definen un tema claro y uno oscuro.
- `frontend/app.js`: el **comportamiento**: acceso a la cámara, comunicación con el servidor, dibujo del esqueleto, métricas y edición del texto.

Una primera versión reunía todo en un único archivo HTML con estilos y código embebidos. La separación facilitó el mantenimiento y permite que cada archivo se lea de forma independiente.

La @fig:interfaz muestra la interfaz en su tema claro. A la izquierda se ubica el video con el esqueleto de las manos superpuesto y, en su esquina, la seña detectada; debajo, los botones para iniciar y detener la cámara. A la derecha, tres paneles muestran la detección (confianza, estabilidad y manos detectadas), el rendimiento y las opciones. En la parte inferior se acumula el texto traducido.

![Interfaz del cliente web en su tema claro, antes de iniciar la cámara.](figuras/cap_interfaz_claro.png){ancho=15 #fig:interfaz}

[[PENDIENTE: agregar una captura con la cámara activa, el esqueleto dibujado y una seña reconocida (requiere hacerla con una cámara real; ver tarea F1 del plan).]]

## 12.2 Captura y envío de cuadros

Al presionar "Iniciar cámara", el cliente solicita acceso a la cámara con `getUserMedia`, pidiendo una resolución ideal de 960 × 720 píxeles, y muestra el video en pantalla. En paralelo, consulta `/model/info` para conocer las clases y las conexiones del esqueleto, y abre el canal WebSocket.

Un temporizador envía cuadros a la frecuencia configurada, de 8 cuadros por segundo por defecto, ajustable por el usuario. Para cada envío, el cliente:

1. Dibuja el cuadro actual del video en un canvas oculto de 480 píxeles de ancho, **espejado** horizontalmente mediante una transformación del contexto de dibujo (sección 9.7).
2. Codifica el canvas como JPEG con calidad 0,7.
3. Envía los bytes resultantes como un mensaje binario y registra el instante del envío.

La reducción a 480 píxeles y la compresión JPEG reducen el volumen de datos sin afectar la detección: MediaPipe trabaja internamente con imágenes de menor resolución.

### 12.2.1 Control de contrapresión

Si el servidor tarda más en procesar un cuadro que el intervalo entre envíos, los cuadros se acumulan en una cola que crece sin límite: la latencia aumenta constantemente y el usuario ve resultados cada vez más atrasados. Para evitarlo, el cliente lleva la cuenta de los cuadros enviados que todavía no tienen respuesta y **no envía uno nuevo si ya hay dos pendientes**. Con ese límite, el servidor siempre tiene trabajo, pero nunca acumula atraso.

Este control tiene también una consecuencia metodológica. Sin él, la latencia medida en el cliente dejaría de reflejar el tiempo de procesamiento y pasaría a medir, sobre todo, el tiempo de espera en la cola. Es un error de medición fácil de cometer y difícil de notar, porque el número resultante parece razonable al principio y empeora gradualmente.

### 12.2.2 Reconexión

Si la conexión WebSocket se interrumpe —por un corte de red o un reinicio del servidor, frecuente en las capas gratuitas—, el cliente intenta reconectarse automáticamente con **espera exponencial**: 0,5 segundos, luego 1, 2, 4 y hasta un máximo de 8 segundos entre intentos. Así se recupera rápido de cortes breves sin saturar al servidor con intentos durante una caída prolongada. El indicador de estado del encabezado informa en todo momento si el cliente está conectado, conectando o desconectado.

## 12.3 Presentación de los resultados

Cada respuesta del servidor actualiza la interfaz:

- La **seña detectada** se muestra sobre el video, en la esquina inferior.
- Las barras de **confianza** y **estabilidad** muestran ambos valores en porcentaje. Un ícono de ayuda junto a cada una explica qué significa.
- El contador de **manos detectadas** permite al usuario saber si está bien ubicado frente a la cámara.
- Cuando una seña se confirma como válida y es distinta de la anterior, se **agrega al texto traducido**. La comparación con la seña anterior evita que una seña sostenida se escriba muchas veces. El agregado automático puede desactivarse.

El texto traducido puede copiarse al portapapeles, corregirse borrando la última seña o limpiarse por completo.

## 12.4 Esqueleto de las manos

El servidor devuelve, junto con cada predicción, las coordenadas de los puntos de las manos detectadas. El cliente las dibuja sobre un canvas superpuesto al video, uniendo los puntos según las conexiones del esqueleto recibidas de `/model/info`. Este dibujo cumple una función práctica: le muestra al usuario, en tiempo real, si el sistema está "viendo" sus manos. Si el esqueleto no aparece, el problema es de detección —iluminación, encuadre, distancia— y no de reconocimiento.

Las coordenadas llegan en el sistema de la imagen ya espejada, y el video también se muestra espejado. Por eso el canvas del esqueleto no lleva ninguna transformación adicional: los puntos caen directamente sobre las manos que ve el usuario. Los grosores de las líneas se escalan según el tamaño del video para mantener la proporción en cualquier pantalla.

## 12.5 Métricas de desempeño

El panel de rendimiento muestra cuatro valores, promediados sobre las últimas 30 respuestas:

- **Latencia total**: el tiempo entre el envío de un cuadro y la recepción de su respuesta, medido en el navegador.
- **Cómputo del servidor**: el tiempo de procesamiento que el servidor informa en cada respuesta.
- **Red y codificación**: la diferencia entre los dos anteriores, que corresponde a la transmisión por la red más la codificación y decodificación del JPEG.
- **Cuadros por segundo** efectivamente procesados.

Separar la latencia en sus dos componentes permite decidir con datos dónde optimizar. Si domina el cómputo del servidor, conviene optimizar el modelo o el preprocesamiento; si domina la red, conviene reducir el tamaño de los cuadros o trasladar la detección al cliente (sección 8.8.1). Las mediciones se presentan en la sección 14.8.

## 12.6 Accesibilidad

Las decisiones de accesibilidad se tomaron de forma explícita, siguiendo las pautas WCAG 2.2 (sección 6.7). La @tab:accesibilidad las resume.

Tabla: Decisiones de accesibilidad del cliente web. {#tab:accesibilidad}

| Decisión | Motivo |
|---|---|
| Región viva (`aria-live="polite"`) sólo en el texto traducido | Cambia únicamente al confirmarse una seña. Las métricas se actualizan unas ocho veces por segundo: anunciadas por un lector de pantalla harían la herramienta inutilizable |
| El canvas del esqueleto se oculta a las tecnologías de asistencia | Es decorativo: la información que transmite está disponible como texto en el panel de detección |
| Barras con `role="progressbar"` y `aria-valuetext` | Un valor leído como "Confianza: 72 por ciento" es más claro que un número sin unidad |
| Mensajes de error con `role="alert"` y estado de conexión con `role="status"` | Se anuncian sin que el usuario tenga que buscarlos |
| Objetivos táctiles de al menos 24 × 24 píxeles | Criterio 2.5.8 (nivel AA). Los íconos de ayuda miden 16 píxeles y se agrandan con un área transparente |
| Contraste verificado en ambos temas | Todos los pares de color superan 4,5:1; el mínimo medido es 4,93:1 |
| Foco visible en todos los controles | Permite el uso completo con teclado |
| Respeto de `prefers-reduced-motion` | Desactiva las animaciones para quienes lo configuraron en su sistema |
| Tema claro y oscuro | Sigue la preferencia del sistema operativo y puede cambiarse manualmente; la elección se recuerda |

La @fig:interfaz-oscuro muestra el tema oscuro, que se aplica automáticamente cuando el sistema operativo del usuario lo tiene configurado.

![Interfaz del cliente web en su tema oscuro.](figuras/cap_interfaz_oscuro.png){ancho=15 #fig:interfaz-oscuro}

## 12.7 Diseño adaptable

La interfaz se adapta al ancho de la pantalla: en pantallas angostas, los paneles laterales pasan a ubicarse debajo del video, en una sola columna, y los botones ocupan todo el ancho para facilitar su uso táctil. [[PENDIENTE: verificar en un teléfono real; en una prueba con un navegador sin interfaz gráfica a 400 píxeles de ancho algunos paneles parecían desbordar a la derecha. Ver tarea F2 del plan.]]

