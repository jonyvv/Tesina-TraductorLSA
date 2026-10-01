# Capítulo 14 — Resultados y evaluación

Este capítulo presenta los resultados de la evaluación del modelo y del sistema. Todos los números provienen de los reportes versionados en el repositorio (`ml/reports/` y `backend/models/modelo_lsa64_lstm.json`), y todas las figuras se generan a partir de ellos con un script (`docs/tesina/figuras/generar_figuras.py`), de modo que pueden reproducirse.

## 14.1 Por qué un único reparto no alcanza

La primera estimación del desempeño se obtuvo con el reparto único descripto en la sección 10.2: 7 personas para entrenar, 1 para validar y 2 para probar. Tres ejecuciones sobre datos prácticamente idénticos, con pequeñas variaciones de la configuración, dieron exactitudes de prueba de **86,05 %**, **78,55 %** y **81,25 %**. Una diferencia de 7,5 puntos entre ejecuciones casi iguales no permite afirmar nada: ni cuál es el desempeño del modelo ni si un cambio lo mejora.

El problema no es de medición, sino de muestra. Con dos personas en la prueba, la estimación depende fuertemente de quiénes sean esas dos personas. Este hallazgo motivó la incorporación del tercer objetivo específico —evaluar con un protocolo independiente del sujeto— y la adopción de la validación leave-one-subject-out como protocolo oficial de la tesina.

## 14.2 Resultado principal

Con la validación LOSO de 10 particiones (sección 10.6), sobre las 2862 secuencias de LSA64 que superan el filtrado y las 64 clases, el modelo obtuvo:

> Exactitud independiente del sujeto: **79,8 % ± 8,9 %** (media ± desvío entre las 10 particiones).

La exactitud calculada sobre las 2862 predicciones agrupadas —cada secuencia predicha una vez por un modelo que nunca vio a esa persona— es de 79,9 %, prácticamente igual a la media de las particiones. La ejecución completa tardó 55 minutos. La @tab:loso detalla cada partición y la @fig:loso-sujeto la representa gráficamente.

Tabla: Resultados por partición de la validación leave-one-subject-out. {#tab:loso}

| Persona de prueba | Persona de validación | Secuencias de entrenamiento | Secuencias de prueba | Mejor época | Exactitud |
|---|---|---|---|---|---|
| S01 | S02 | 2270 | 286 | 34 | 86,4 % |
| S02 | S03 | 2274 | 306 | 7 | 75,5 % |
| S03 | S04 | 2283 | 282 | 12 | 63,1 % |
| S04 | S05 | 2309 | 297 | 15 | 82,2 % |
| S05 | S06 | 2310 | 256 | 16 | 87,1 % |
| S06 | S07 | 2262 | 296 | 20 | 80,7 % |
| S07 | S08 | 2297 | 304 | 14 | 89,5 % |
| S08 | S09 | 2297 | 261 | 9 | 66,3 % |
| S09 | S10 | 2288 | 304 | 24 | 85,5 % |
| S10 | S01 | 2306 | 270 | 26 | 81,9 % |
| **Media ± desvío** | | | | | **79,8 % ± 8,9 %** |

![Exactitud por persona en la validación leave-one-subject-out. La línea indica la media y la banda, un desvío. En rojo, las personas por debajo de la media menos un desvío.](figuras/fig_loso_por_sujeto.png){ancho=14.5 #fig:loso-sujeto}

**El desvío es un hallazgo, no un defecto de la medición.** Entre la persona con mejor resultado (S07, 89,5 %) y la de peor resultado (S03, 63,1 %) hay 26 puntos. Ninguna partición es un valor atípico según la regla habitual de 1,5 veces el rango intercuartílico: el límite inferior es 62,8 % y el peor valor es 63,1 %. La variación es, por lo tanto, una propiedad del problema: el modelo funciona bien para la mayoría de las personas y notablemente peor para dos de ellas, S03 y S08. Un número único, obtenido con un reparto que dejara a esas dos personas en el entrenamiento, habría ocultado este comportamiento.

## 14.3 Curva de entrenamiento y sobreajuste

La @fig:curva muestra la evolución del entrenamiento del modelo servido (reparto único, semilla 42). La pérdida de entrenamiento desciende de forma sostenida, de 2,55 en la primera época a 0,16 en la última, mientras que la exactitud de validación deja de mejorar a partir de la época 11 (81,6 %) y oscila alrededor del 78 % hasta que la detención temprana corta el entrenamiento en la época 19.

![Pérdida de entrenamiento y exactitud de validación por época del modelo servido. La línea punteada marca la mejor época, cuyos pesos se conservan.](figuras/fig_curva_entrenamiento.png){ancho=14 #fig:curva}

Es el patrón típico del sobreajuste (sección 2.3): a partir de cierto punto, el modelo sigue aprendiendo particularidades del conjunto de entrenamiento que no generalizan a la persona de validación. La detención temprana evita que el modelo empeore, pero no resuelve la causa. Las técnicas de regularización evaluadas para atacarla se presentan en la sección 14.6.

El modelo servido obtuvo una exactitud de prueba de **81,25 %** sobre las personas S01 y S02, coherente con el resultado LOSO de esas dos personas (86,4 % y 75,5 %).

## 14.4 Análisis por clase

La exactitud no es pareja entre las 64 señas. La @fig:por-clase ordena las clases según su exactitud en las predicciones agrupadas del LOSO. La mediana es 81,4 %; 34 clases superan el 80 % y 18 superan el 90 %, con cuatro reconocidas sin errores (Rojo, Cajón, Agua y Dónde). En el otro extremo, 8 clases quedan por debajo del 60 %.

![Exactitud por clase en las predicciones agrupadas del LOSO, ordenada de menor a mayor. En rojo, las clases por debajo de 60 %; en naranja, entre 60 % y 80 %. La línea punteada indica la exactitud global.](figuras/fig_accuracy_por_clase.png){ancho=13.5 #fig:por-clase}

Las clases con peor desempeño son Burlarse (14,8 %), Hombre (47,8 %), Aparecer (48,7 %), Comida (48,8 %), Aceptar (52,1 %), Aprender (54,0 %), Fideos (54,0 %) y Paciencia (59,1 %). Varias de ellas coinciden con las clases que más secuencias perdieron por falta de detección de las manos (sección 9.10): Hombre conserva 23 de sus 50 videos, Paciencia 22 y Burlarse 27. Con menos de la mitad de los datos, esas clases están subrepresentadas en el entrenamiento.

### 14.4.1 Señas de una y dos manos

Las 42 señas de una mano alcanzan una exactitud de 81,9 % y las 22 señas bimanuales, de 76,4 %. La diferencia es esperable: en una seña bimanual hay dos manos que detectar en cada cuadro, y la probabilidad de perder alguna es mayor. De los 574 errores del LOSO, 170 confunden una seña de una mano con una de dos o viceversa.

### 14.4.2 Matriz de confusión

La @fig:confusion presenta la matriz de confusión de las 2862 predicciones agrupadas, normalizada por fila. La diagonal concentra la mayor parte de las predicciones; los errores se reparten de forma dispersa, con algunos pares que destacan.

![Matriz de confusión de las predicciones agrupadas del LOSO, normalizada por fila. Los números de los ejes corresponden a la numeración de LSA64 (ver el Anexo A).](figuras/fig_matriz_confusion.png){ancho=13.5 #fig:confusion}

La @tab:confusiones enumera los pares más confundidos.

Tabla: Pares de señas más confundidos en el LOSO. {#tab:confusiones}

| Seña real | Predicha como | Casos | Secuencias de la seña real |
|---|---|---|---|
| Aprender | Caramelo | 12 | 50 |
| Ayudar | Agradecer | 11 | 47 |
| Espumadera | Brillante | 10 | 50 |
| Caramelo | Aprender | 9 | 46 |
| Hijo | Mujer | 8 | 44 |
| Hombre | Aprender | 7 | 23 |
| Amarillo | Bañarse | 7 | 50 |
| Colores | Aprender | 7 | 47 |
| Rosa | Argentina | 7 | 48 |
| Enemigo | Perfume | 7 | 40 |
| Ninguno | Chicle | 7 | 50 |
| Fideos | Yogur | 7 | 50 |

La confusión más frecuente es simétrica: Aprender se confunde con Caramelo y Caramelo con Aprender. Según el análisis del modelo servido, las confusiones se concentran en señas que comparten configuración manual y difieren principalmente en la **ubicación**, precisamente el parámetro que la normalización respecto de la muñeca descarta (sección 9.6). [[PENDIENTE (tarea M4 del plan): verificar con los videos de LSA64, par por par, si las señas más confundidas efectivamente comparten configuración y difieren en ubicación, y describirlo aquí con una figura de ejemplo.]]

## 14.5 Detección de manos y desempeño

La sección 9.10 mostró que la proporción de cuadros con manos detectadas varía entre personas. La @fig:retencion relaciona esa retención con la exactitud LOSO de cada persona.

![Relación entre la retención de cuadros con manos detectadas y la exactitud LOSO por persona.](figuras/fig_retencion_vs_accuracy.png){ancho=12.5 #fig:retencion}

La correlación es de **0,68**: las personas con peor detección tienden a obtener peor exactitud, y las dos personas con peor resultado, S03 y S08, son también las dos con peor retención (63,5 % y 58,5 %). Con 10 personas, la correlación es sugestiva pero no concluyente. La interpretación es consistente con el diseño del sistema: el clasificador sólo ve lo que MediaPipe detecta, y una persona cuyas manos se pierden en un tercio de los cuadros le entrega al modelo secuencias incompletas.

## 14.6 Experimentos de mejora

### 14.6.1 Hipótesis descartada: preservar la grilla temporal

**Hipótesis.** Al descartar los cuadros sin manos, la secuencia se comprime de forma irregular: un video de 44 cuadros muestreados puede quedar en 18. Se sospechaba que ese "salto" en el tiempo, y no sólo la pérdida de información, perjudicaba al modelo. Si así fuera, conservar los cuadros vacíos —con sus vectores en cero, manteniendo el ritmo original de la seña— debería mejorar el desempeño, especialmente en las personas con peor detección.

**Resultado.** La hipótesis resultó falsa. La @tab:grilla y la @fig:grilla comparan ambas representaciones con el mismo protocolo LOSO y el mismo caché.

Tabla: Comparación entre filtrar los cuadros sin manos y conservar la grilla temporal completa. {#tab:grilla}

| Persona | Retención | Cuadros filtrados | Grilla completa | Diferencia |
|---|---|---|---|---|
| S01 | 75,7 % | 86,4 % | 79,4 % | −7,0 |
| S02 | 73,7 % | 75,5 % | 72,9 % | −2,6 |
| S03 | 63,5 % | 63,1 % | 62,8 % | −0,4 |
| S04 | 68,3 % | 82,2 % | 80,1 % | −2,0 |
| S05 | 65,5 % | 87,1 % | 74,6 % | −12,5 |
| S06 | 73,7 % | 80,7 % | 75,7 % | −5,1 |
| S07 | 73,2 % | 89,5 % | 82,2 % | −7,2 |
| S08 | 58,5 % | 66,3 % | 72,8 % | +6,5 |
| S09 | 74,6 % | 85,5 % | 85,9 % | +0,3 |
| S10 | 69,7 % | 81,9 % | 78,9 % | −3,0 |
| **Media** | | **79,8 %** | **76,5 %** | **−3,3** |
| **Desvío** | | **8,9 %** | **6,4 %** | |

![Exactitud por persona con los cuadros sin manos filtrados (representación oficial) y con la grilla temporal completa.](figuras/fig_filtrado_vs_grilla.png){ancho=14.5 #fig:grilla}

Ocho de las diez personas empeoraron. El desvío baja, pero nivelando hacia abajo: el máximo cae de 89,5 % a 85,9 % mientras el mínimo se mantiene alrededor del 63 %. La correlación entre retención y exactitud apenas cambia (de 0,68 a 0,60), lo que indica que rellenar con ceros no desacopla el desempeño de la calidad de la detección, que era el objetivo.

**Interpretación.** El problema no es que los cuadros se hayan quitado, sino que faltan. Un vector en cero conserva la posición temporal pero no aporta información, y obliga a la red a dedicar capacidad a ignorar un tercio de su entrada; con un modelo que ya sobreajusta, el balance es negativo. La BiLSTM tolera la compresión irregular del tiempo mejor de lo que se suponía. Sólo la persona S08, la de peor retención, mejoró notablemente (+6,5), pero S05, con 65,5 % de retención, empeoró 12,5 puntos, de modo que no hay un patrón. **La representación con cuadros filtrados se mantiene como oficial**, y un relleno de los huecos sólo debería volver a evaluarse con un mecanismo distinto, como la interpolación o una máscara explícita.

### 14.6.2 Regularización y representación con ubicación

Para evaluar rápidamente las técnicas contra el sobreajuste se utilizó la evaluación parcial sobre las dos personas más difíciles, S03 y S08 (sección 10.6.1). La @fig:screening resume los resultados.

![Exactitud media sobre las personas S03 y S08 para la configuración de referencia y tres variantes.](figuras/fig_screening.png){ancho=12.5 #fig:screening}

- **Configuración de referencia**: 64,70 %.
- **Dropout 0,4** (en lugar de 0,2): 66,56 %, una mejora de 1,9 puntos, presente en las dos personas.
- **Descarte aleatorio de cuadros con probabilidad 0,30**: 66,38 %, una mejora de 1,7 puntos, también en las dos personas. Es la técnica que simula directamente a una persona con mala detección, el caso en que el modelo falla.
- **Representación v2 con la ubicación de la muñeca**: 64,26 %, sin mejora.

Estos resultados deben leerse con cautela: son dos particiones de diez y las diferencias son pequeñas. Indican qué variantes vale la pena confirmar, no que mejoren el modelo. [[PENDIENTE (tarea M1 del plan): confirmar con el LOSO completo la combinación de dropout 0,4 y descarte de cuadros 0,30, y, si mejora, reentrenar el modelo servido.]]

El resultado de la representación v2 merece un comentario. Que agregar la ubicación no mejore, cuando el análisis de confusiones sugiere que la ubicación es la información faltante, tiene al menos dos explicaciones posibles: que la posición de la muñeca en el encuadre no sea una buena medida de la ubicación respecto del cuerpo, o que las dos personas evaluadas fallen por la detección y no por la ubicación. Distinguir ambas requiere evaluar v2 con el LOSO completo y, eventualmente, una representación de la ubicación relativa al cuerpo.

## 14.7 Comparación con los antecedentes

La @tab:antecedentes del Capítulo 5 ubica el resultado propio respecto de la literatura. En el mismo protocolo —independiente del sujeto—, el 79,8 % supera el 74,22 % de Marais et al. (2022) y queda por debajo del 98,53 % de Akdag y Baykan (2024), obtenido con un modelo convolucional 3D sobre el video completo. El 99,4 % de Mindlin (2021) corresponde a un protocolo dependiente del sujeto y no es comparable.

### 14.7.1 Línea base no neuronal

Para justificar el uso de una red recurrente, es necesario compararla con un modelo más simple. La línea base prevista es un bosque aleatorio (sección 2.6) entrenado sobre un resumen de cada secuencia —media, desvío, mínimo y máximo de cada característica—, evaluado con el mismo protocolo LOSO. [[PENDIENTE (tarea M2 del plan): ejecutar la línea base y completar este apartado con su resultado. Si el bosque aleatorio obtiene un resultado cercano al de la BiLSTM, la conclusión es que el orden temporal aporta poco con esta representación, lo que también es un resultado relevante.]]

## 14.8 Desempeño en tiempo real

### 14.8.1 Latencia

[[PENDIENTE (tarea F3 del plan): medir la latencia con el panel de rendimiento del cliente, en tres escenarios — servidor local, servidor en la red local y servidor desplegado — y registrar latencia total, cómputo del servidor y red, a 8 y 15 cuadros por segundo. Presentar los resultados en una tabla.]]

### 14.8.2 Brecha entre el entrenamiento y el uso en tiempo real

La exactitud del LOSO mide la capacidad del modelo para clasificar una seña completa, recortada de antemano, grabada en condiciones de laboratorio. El uso en la aplicación difiere de esa situación en tres aspectos, que constituyen la principal limitación conocida del sistema:

1. **Frecuencia de muestreo.** El modelo se entrenó con secuencias a 30 cuadros por segundo (sección 9.2). El cliente envía, por defecto, 8 cuadros por segundo. Una seña de un segundo ocupa 30 cuadros en el entrenamiento y 8 en la aplicación: el modelo la ve "acelerada" casi cuatro veces.
2. **Segmentación.** En el entrenamiento, cada secuencia es exactamente una seña, de principio a fin. En la aplicación no se sabe dónde empieza y termina una seña: la sesión clasifica una ventana deslizante de los últimos 8 cuadros con manos detectadas (sección 11.5), que puede contener el final de una seña, el comienzo de otra o una transición.
3. **Condiciones de captura.** LSA64 se grabó con fondo blanco, ropa negra, guantes de color e iluminación controlada. Una cámara web doméstica captura escenas más variadas.

Ninguno de estos aspectos se refleja en la cifra del LOSO, y es esperable que el desempeño percibido en la aplicación sea menor. [[PENDIENTE (tareas M5 y F3 del plan): (a) alinear la frecuencia — enviar a 15 cuadros por segundo o volver a entrenar con un muestreo equivalente al del cliente —; (b) reemplazar la ventana fija por una segmentación según la presencia de las manos, que acumule la seña completa y clasifique al bajar las manos; (c) medir el desempeño real con un protocolo simple: un conjunto de señas realizadas frente a la cámara por personas que no estén en LSA64, registrando aciertos, errores y señas no detectadas.]]

## 14.9 Discusión

Los resultados permiten extraer algunas conclusiones sobre el modelo y sobre el proceso.

**El protocolo de evaluación determina la cifra.** La misma familia de modelos obtiene valores cercanos al 99 % o en torno al 75 % según se mezclen o no las personas entre entrenamiento y prueba (Capítulo 5). En este trabajo, un reparto único habría permitido informar 86 % eligiendo, sin malicia, la ejecución más favorable. La cifra que se informa, 79,8 % ± 8,9 %, es menor, pero es la que corresponde al uso real del sistema: reconocer a personas que el modelo nunca vio.

**La calidad de la detección es el principal factor limitante.** La exactitud por persona se asocia con la retención de cuadros, las clases con peor desempeño son en buena medida las que más datos perdieron, y las señas bimanuales, con dos manos que detectar, funcionan peor. Mejorar la detección —con otros umbrales, con el modo de seguimiento de MediaPipe o con la recuperación de los videos descartados— tiene, probablemente, más potencial que modificar la red.

**La ubicación es la información faltante más plausible.** Las confusiones apuntan a señas que difieren en la ubicación, y la representación v1 descarta ese parámetro por diseño. La primera prueba de agregarlo no dio resultado, pero la prueba fue parcial y la representación de la ubicación, mejorable.

**Los resultados negativos también son resultados.** La hipótesis de la grilla temporal era razonable y resultó falsa. Haberla evaluado con un protocolo riguroso evitó adoptar un cambio que empeoraba el modelo en 3,3 puntos, y dejó documentado por qué no conviene volver a proponerlo en su forma simple.

## 14.10 Síntesis

El modelo alcanza una exactitud independiente del sujeto de 79,8 % ± 8,9 % sobre las 64 señas de LSA64, con una variación entre personas que está asociada con la calidad de la detección de las manos. Las confusiones se concentran en señas que comparten configuración. Preservar la grilla temporal con ceros empeora el modelo; el dropout más alto y el descarte aleatorio de cuadros muestran mejoras preliminares que deben confirmarse. La principal limitación del sistema es la diferencia entre las condiciones del entrenamiento y las del uso en tiempo real, que deberá medirse y reducirse en la etapa final del proyecto.
