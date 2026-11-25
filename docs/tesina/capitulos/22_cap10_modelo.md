# Capítulo 10 — Modelo de reconocimiento y entrenamiento

Este capítulo describe la red neuronal que clasifica las señas, el procedimiento de entrenamiento, los mecanismos que garantizan la validez y la reproducibilidad de las mediciones, y la implementación de la validación leave-one-subject-out.

## 10.1 Arquitectura de la red

El clasificador es una red **BiLSTM** de una capa, implementada en PyTorch en el módulo compartido (`common/models/lsa64.py`), de modo que la misma definición se usa para entrenar y para inferir. La @fig:bilstm muestra su estructura:

1. **Entrada**: una secuencia de T vectores de 138 características, con T variable entre 8 y 120.
2. **Capa LSTM bidireccional** con 128 unidades por dirección. Las secuencias de un lote se empaquetan con su longitud real, de modo que la recurrencia se detiene en el último cuadro válido de cada una (sección 3.7.1).
3. **Concatenación** del estado oculto final de la dirección hacia adelante y de la dirección hacia atrás, que da un vector de 256 valores que resume la secuencia completa.
4. **Dropout** con probabilidad 0,4 en el modelo servido (0,2 en la configuración de referencia, sección 10.3).
5. **Capa lineal** de 256 a 64 valores, uno por seña, seguida de softmax para obtener las probabilidades.

![Arquitectura de la red BiLSTM. Cada cuadro de la secuencia alimenta dos LSTM que la recorren en sentidos opuestos; sus estados finales se concatenan y una capa lineal produce una puntuación por seña.](figuras/fig_bilstm.png){ancho=14.5 #fig:bilstm}

La red tiene **290 880 parámetros**: 274 432 en la capa LSTM bidireccional y 16 448 en la capa lineal. Es un modelo deliberadamente pequeño: con unas 2000 secuencias de entrenamiento por partición, una red más grande sólo aumentaría el sobreajuste. El Anexo B detalla las dimensiones de cada capa.

## 10.2 Reparto de los datos

Se utilizan dos esquemas de reparto, con propósitos distintos.

**Reparto único**, para entrenar el modelo que se sirve en la aplicación. Las 10 personas se ordenan al azar con una semilla fija y se asignan 7 al entrenamiento, 1 a la validación y 2 a la prueba. El reparto es siempre **por persona**: todas las secuencias de una persona quedan del mismo lado. Con la semilla 42, el entrenamiento usa 2014 secuencias, la validación 256 y la prueba 592.

**Validación leave-one-subject-out**, para medir el desempeño (sección 10.6). Es el protocolo con el que se reportan los resultados.

### 10.2.1 Verificación de fuga

Después del error de etiquetado de la sección 9.11, que mezclaba a las personas entre los conjuntos sin que nada lo advirtiera, se incorporó una verificación explícita: antes de entrenar, el sistema calcula el conjunto de personas de cada partición y comprueba que sean disjuntos. Si encuentra una persona en más de una partición, lo informa en la consola y lo registra en el reporte de resultados, en el campo `fugas_detectadas`.

## 10.3 Procedimiento de entrenamiento

El entrenamiento minimiza la entropía cruzada con el optimizador AdamW. La @tab:hiperparametros resume las dos configuraciones usadas: la **de referencia**, punto de partida de las mejoras, y la **definitiva**, con dos técnicas de regularización (sección 10.5), que es la del modelo servido y la del resultado principal (sección 14.6.3).

Tabla: Hiperparámetros de las configuraciones de referencia y definitiva. {#tab:hiperparametros}

| Hiperparámetro | Referencia | Definitiva (modelo servido) |
|---|---|---|
| Unidades LSTM por dirección | 128 | 128 |
| Dropout antes de la capa lineal | 0,2 | 0,4 |
| Descarte aleatorio de cuadros | 0 | 0,30 |
| Optimizador | AdamW, tasa 0,001, sin penalización de pesos | Ídem |
| Tamaño de lote | 8 | 8 |
| Épocas máximas | 40 (reparto único) / 60 (LOSO) | 40 (reparto único) / 60 (LOSO) |
| Paciencia de la detención temprana | 8 épocas | 8 épocas |
| Muestreo temporal (`frame_step`) | 2 (30 cuadros por segundo) | 2 (30 cuadros por segundo) |
| Longitud de la secuencia | Entre 8 y 120 cuadros | Entre 8 y 120 cuadros |
| Semilla | 42 | 42 |

Al final de cada época se calcula la exactitud sobre el conjunto de validación. Cuando mejora, se guarda una copia de los pesos; si no mejora durante 8 épocas seguidas, el entrenamiento se detiene y se restauran los mejores pesos guardados. En el reparto único, el modelo servido tuvo su mejor época en la 19 y se detuvo en la 27 (sección 14.3).

## 10.4 Reproducibilidad

En una primera versión, la semilla sólo determinaba el reparto de los datos. La inicialización de los pesos, las máscaras de dropout y el orden de los lotes quedaban librados al azar, y dos ejecuciones sobre los mismos datos diferían en varios puntos de exactitud. En esas condiciones es imposible saber si un cambio mejora el modelo o si la diferencia es ruido.

La función `set_seed` fija las semillas de Python, NumPy y PyTorch, y el cargador de datos usa un generador propio también inicializado con la semilla, porque el desordenamiento de los lotes no depende de la semilla global. Con esto, el entrenamiento es determinista en un mismo equipo: misma configuración, mismo resultado, época por época. Se comprobó además que filtrar los cuadros vacíos de un caché de secuencias completas reconstruye bit a bit el caché anterior (las 2862 secuencias), y que el resultado del LOSO se reproduce exactamente incluso después de repetir la extracción completa con MediaPipe.

Entre equipos distintos la reproducción no es exacta: la configuración definitiva, repetida en otro equipo con GPU, dio 81,0 % ± 6,1 % en lugar de 81,7 % ± 7,9 %, probablemente por diferencias mínimas en el cálculo numérico que se acumulan a lo largo de las épocas. Por eso dos configuraciones sólo se comparan si sus resultados salen del mismo equipo, y los obtenidos con GPU se informan aparte (sección 14.6.4).

## 10.5 Regularización y aumentación de datos

Para atacar el sobreajuste observado (sección 14.3) se incorporaron al entrenamiento las técnicas de la sección 3.4, todas como opciones de línea de comandos **desactivadas por defecto**, de modo que una ejecución sin opciones reproduzca exactamente la configuración de referencia:

- `--dropout p`: probabilidad de dropout antes de la capa lineal.
- `--weight-decay λ`: penalización de pesos, aplicada en forma desacoplada por AdamW.
- `--aug-frame-drop p`: descarta cada cuadro con probabilidad p. Simula a una persona con mala detección de manos, que es justamente el caso en que el modelo falla.
- `--aug-time-scale s`: cambia la duración de la secuencia por un factor aleatorio entre 1 − s y 1 + s, remuestreando por vecino más cercano. Simula señas más rápidas o más lentas.
- `--aug-noise σ`: suma ruido gaussiano a las coordenadas normalizadas. Sólo se aplica a los cuadros con manos detectadas: aplicarlo a un cuadro vacío crearía una "mano fantasma" en el origen.

La aumentación se aplica en cada época con una variante aleatoria distinta de cada secuencia, y sólo sobre el conjunto de entrenamiento: nunca sobre validación ni prueba, que deben medir el desempeño sobre datos reales. La configuración definitiva usa dos de estas opciones, `--dropout 0.4` y `--aug-frame-drop 0.30`. Los resultados de todas se presentan en la sección 14.6.

## 10.6 Validación leave-one-subject-out

El script `ml/evaluate_loso.py` implementa el protocolo LOSO sobre el mismo caché y con el mismo ciclo de entrenamiento que el reparto único, de modo que los resultados son comparables. Para cada una de las 10 personas:

- La persona es el conjunto de **prueba**.
- La persona siguiente, en orden cíclico, es el conjunto de **validación**, usado para la detención temprana.
- Las 8 personas restantes son el conjunto de **entrenamiento**.

La asignación es determinista: no depende de ninguna semilla y cada persona cumple una vez el rol de prueba y una vez el de validación. El script entrena los 10 modelos y genera un reporte JSON con la exactitud de cada partición, la media y el desvío, la exactitud por clase y por persona, y las predicciones de las 10 particiones agrupadas. Esas predicciones agrupadas permiten construir la matriz de confusión de todo el conjunto de datos (sección 14.4).

El LOSO no guarda modelos: es un protocolo de medición. El modelo que se sirve en la aplicación es el del reparto único.

### 10.6.1 Evaluación parcial sobre personas difíciles

Una validación LOSO completa tarda alrededor de una hora, demasiado para explorar muchas variantes. Por otro lado, el reparto único no sirve para elegir entre variantes, porque las dos personas con peor desempeño, S03 y S08, caen en su conjunto de entrenamiento. Se agregó por eso la opción `--solo-sujetos`, que ejecuta sólo las particiones de las personas indicadas. Evaluar una variante sobre ellas tarda unos diez minutos. Las variantes prometedoras deben confirmarse luego con el LOSO completo.

## 10.7 Artefactos del entrenamiento

El entrenamiento produce dos archivos:

- **El modelo** (`backend/models/modelo_lsa64_lstm.pt`): un *checkpoint* de PyTorch con los pesos y los metadatos necesarios para reconstruir la red sin datos externos (Anexo B.3).
- **El resumen** (`backend/models/modelo_lsa64_lstm.json`): los hiperparámetros, el reparto de personas, la exactitud de validación y de prueba, la exactitud por clase y la historia de la pérdida y la exactitud por época.

### 10.7.1 Glosas y orden de las clases

La traducción de las etiquetas canónicas (`clase_01` a `clase_64`) a las glosas en castellano produjo un error instructivo. La primera implementación traducía las etiquetas antes de entrenar. Pero el codificador de etiquetas asigna los índices de las clases en orden alfabético, y con las glosas el orden cambia —"Aceptar" pasa a ser la primera clase—, lo que altera la correspondencia entre clases y neuronas de salida y, con ella, el resultado del entrenamiento. Con la configuración de referencia se midió una diferencia de exactitud de 0,8125 a 0,8159, con la mejor época pasando de la 11 a la 32, sobre los mismos datos. Una opción de presentación no puede modificar el resultado de un experimento.

La corrección consistió en entrenar siempre con las etiquetas canónicas, cuyo orden alfabético coincide con el numérico, y aplicar las glosas al final, sólo sobre los nombres que se guardan en el modelo y en el reporte. Con la corrección, el modelo reentrenado con las glosas reproduce exactamente el resultado de referencia: 0,8125, mejor época 11.

