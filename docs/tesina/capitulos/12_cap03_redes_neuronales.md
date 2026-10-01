# Capítulo 3 — Redes neuronales y aprendizaje profundo

El modelo central del sistema es una red neuronal recurrente. Este capítulo recorre los conceptos necesarios para entenderla: la neurona artificial, el entrenamiento por descenso del gradiente, las técnicas de regularización y, finalmente, las redes recurrentes y la arquitectura LSTM bidireccional.

## 3.1 La neurona artificial

Una **red neuronal artificial** es un modelo compuesto por unidades simples interconectadas, inspiradas lejanamente en las neuronas biológicas. Cada unidad recibe un vector de entradas x, calcula una suma ponderada con un vector de pesos w, le agrega un sesgo b y aplica una **función de activación** no lineal g:

> salida = g(w · x + b)

Los pesos y el sesgo son los **parámetros** del modelo, los valores que se ajustan durante el entrenamiento. La función de activación es indispensable: sin ella, cualquier combinación de neuronas se reduce a una única función lineal y la red no puede representar relaciones complejas. Las activaciones más usadas son la sigmoide σ(z) = 1 / (1 + e^(−z)), que comprime cualquier valor al intervalo (0, 1); la tangente hiperbólica, que lo comprime a (−1, 1); y la unidad lineal rectificada (ReLU), que devuelve max(0, z) (Goodfellow et al., 2016).

## 3.2 Redes multicapa y aprendizaje profundo

Las neuronas se organizan en **capas**. En un **perceptrón multicapa** (MLP), la entrada atraviesa una o más capas ocultas antes de llegar a la capa de salida, y cada neurona de una capa se conecta con todas las de la capa siguiente. Con suficientes neuronas, un MLP puede aproximar cualquier función continua, lo que explica su versatilidad.

El término **aprendizaje profundo** (*deep learning*) se refiere a redes con muchas capas, en las que cada capa construye representaciones progresivamente más abstractas a partir de la anterior (Goodfellow et al., 2016). Su principal ventaja frente a los métodos clásicos es que aprende las características directamente de los datos en lugar de depender de características diseñadas a mano. Esa capacidad tiene un costo: requiere muchos datos y mucho cómputo, dos recursos que, en el caso de LSA64, son escasos. Esta tensión explica una decisión central del sistema, desarrollada en el Capítulo 4: en lugar de aprender directamente de los píxeles, la red recibe una representación compacta de las manos calculada por un modelo previamente entrenado.

### 3.2.1 Capa de salida y función softmax

En un problema de clasificación con K clases, la capa de salida tiene K neuronas, una por clase. Sus valores, llamados *logits*, se transforman en una distribución de probabilidad mediante la función **softmax**, que exponencia cada valor y lo divide por la suma de todos. El resultado son K números positivos que suman uno; el mayor indica la clase predicha y su valor es la confianza del modelo. En el sistema, K = 64.

## 3.3 Entrenamiento

### 3.3.1 Función de pérdida

Entrenar una red significa encontrar los parámetros que minimizan una **función de pérdida**, que mide qué tan lejos están las predicciones de las etiquetas correctas. Para clasificación, la pérdida estándar es la **entropía cruzada**: el logaritmo negativo de la probabilidad que el modelo asignó a la clase correcta. Si el modelo asigna probabilidad 1 a la clase correcta, la pérdida es 0; si le asigna una probabilidad cercana a 0, la pérdida crece sin límite.

### 3.3.2 Descenso del gradiente y retropropagación

El **descenso del gradiente** ajusta cada parámetro en la dirección que más reduce la pérdida, en pasos cuyo tamaño controla la **tasa de aprendizaje**. El gradiente de la pérdida respecto de cada parámetro se calcula eficientemente con el algoritmo de **retropropagación** (*backpropagation*), que aplica la regla de la cadena desde la salida hacia la entrada (Rumelhart et al., 1986).

En la práctica, el gradiente no se calcula sobre todo el conjunto de entrenamiento sino sobre pequeños subconjuntos aleatorios llamados **lotes** (*batches*). Una pasada completa por todos los lotes es una **época**. El entrenamiento consiste en repetir épocas hasta que la pérdida deja de mejorar.

### 3.3.3 Optimizadores

El descenso del gradiente básico usa la misma tasa de aprendizaje para todos los parámetros. Los optimizadores modernos la adaptan individualmente. **Adam** (Kingma y Ba, 2015) mantiene un promedio móvil del gradiente y de su cuadrado para cada parámetro, lo que acelera la convergencia y lo hace poco sensible a la elección de la tasa inicial; es el optimizador utilizado por defecto en el sistema. **AdamW** (Loshchilov y Hutter, 2019) es una variante que aplica la penalización de pesos de forma desacoplada del gradiente, y se incorporó para los experimentos de regularización (sección 10.5).

## 3.4 Regularización

Una red con cientos de miles de parámetros entrenada sobre unos pocos miles de ejemplos tiene capacidad de sobra para memorizarlos. Las técnicas de **regularización** limitan esa capacidad o hacen más difícil la memorización, con el objetivo de mejorar la generalización (Goodfellow et al., 2016). Las utilizadas en este trabajo son:

- **Detención temprana** (*early stopping*). Se evalúa el modelo sobre el conjunto de validación al final de cada época y se conserva el mejor; si no mejora durante un número de épocas fijado (la **paciencia**), el entrenamiento se detiene (Prechelt, 1998).
- **Dropout.** Durante el entrenamiento, cada neurona de una capa se desactiva al azar con una probabilidad p en cada paso. La red no puede depender de ninguna neurona en particular y aprende representaciones más redundantes y robustas (Srivastava et al., 2014). En la inferencia todas las neuronas están activas.
- **Penalización de pesos** (*weight decay*). Agrega a la pérdida un término proporcional a la magnitud de los pesos, lo que favorece soluciones más simples.
- **Aumentación de datos** (*data augmentation*). Genera variantes artificiales de los ejemplos de entrenamiento aplicando transformaciones que no cambian su clase. Para secuencias de landmarks, en este trabajo se implementaron tres: ruido gaussiano sobre las coordenadas, descarte aleatorio de cuadros y cambio de escala temporal (sección 10.5).

## 3.5 Redes neuronales recurrentes

Un MLP recibe un vector de tamaño fijo y no tiene noción de orden. Una seña, en cambio, es una **secuencia**: una sucesión de posturas de la mano cuyo orden importa —la misma configuración moviéndose hacia arriba o hacia abajo puede ser dos señas distintas— y cuya duración varía de una persona a otra y de una repetición a otra.

Las **redes neuronales recurrentes** (RNN) procesan secuencias elemento por elemento, manteniendo un **estado oculto** que resume lo visto hasta el momento. En cada paso t, la red combina la entrada actual x_t con el estado anterior h_(t−1) para producir el nuevo estado h_t. Los mismos pesos se aplican en todos los pasos, de modo que la red puede procesar secuencias de cualquier longitud con una cantidad fija de parámetros. Al final de la secuencia, el último estado oculto resume la secuencia completa y puede alimentar una capa de clasificación.

### 3.5.1 El problema del gradiente que se desvanece

Entrenar una RNN implica retropropagar el error a través de todos los pasos de la secuencia. En cada paso, el gradiente se multiplica por factores que suelen ser menores que uno, y después de decenas de pasos se vuelve prácticamente nulo. Bengio et al. (1994) mostraron que, como consecuencia, las RNN simples tienen enormes dificultades para aprender dependencias entre elementos alejados de la secuencia: el comienzo de una seña casi no influye en el aprendizaje cuando la red llega al final.

## 3.6 Long Short-Term Memory

La arquitectura **LSTM** (*Long Short-Term Memory*), propuesta por Hochreiter y Schmidhuber (1997), resuelve ese problema incorporando, además del estado oculto, una **celda de memoria** c_t por la que la información puede fluir a lo largo de la secuencia con modificaciones mínimas. El acceso a esa memoria está regulado por tres **compuertas**, cada una implementada como una capa con activación sigmoide que produce valores entre 0 (cerrado) y 1 (abierto):

- La **compuerta de olvido** f_t decide qué parte de la memoria anterior se conserva.
- La **compuerta de entrada** i_t decide qué parte de la nueva información candidata ĉ_t se escribe en la memoria.
- La **compuerta de salida** o_t decide qué parte de la memoria se expone como estado oculto.

La actualización de la memoria es c_t = f_t ⊗ c_(t−1) ⊕ i_t ⊗ ĉ_t, y el estado oculto es h_t = o_t ⊗ tanh(c_t), donde ⊗ denota el producto elemento a elemento. Como la memoria se actualiza mediante una suma y no mediante multiplicaciones sucesivas, el gradiente puede atravesar muchos pasos sin desvanecerse. La @fig:celda-lstm esquematiza la celda.

![Esquema de una celda LSTM. Las compuertas de olvido, entrada y salida regulan qué información se conserva, se escribe y se expone en cada paso de la secuencia.](figuras/fig_celda_lstm.png){ancho=13 #fig:celda-lstm}

## 3.7 LSTM bidireccional

Una LSTM procesa la secuencia de izquierda a derecha, de modo que su estado en el paso t sólo conoce el pasado. Cuando se clasifica una secuencia completa —como una seña ya terminada—, no hay razón para limitarse a esa dirección. Una **red recurrente bidireccional** (Schuster y Paliwal, 1997) combina dos redes: una recorre la secuencia hacia adelante y otra hacia atrás. Concatenando el estado final de cada una se obtiene una representación que tiene en cuenta tanto cómo empieza como cómo termina la seña. Esta arquitectura, denominada **BiLSTM**, es la que utiliza el sistema (sección 10.1).

### 3.7.1 Secuencias de longitud variable

Para entrenar en lotes, todas las secuencias de un lote deben tener la misma longitud, por lo que las más cortas se completan con ceros (*padding*). Si la red procesara esos ceros como si fueran datos, el estado final de las secuencias cortas quedaría contaminado por pasos vacíos. Los marcos de trabajo como PyTorch (Paszke et al., 2019) resuelven esto "empaquetando" las secuencias junto con su longitud real, de modo que la recurrencia se detiene en el último elemento válido de cada una. El modelo del sistema utiliza este mecanismo (ver el Anexo B).

## 3.8 Otras arquitecturas

El estado del arte en reconocimiento de video utiliza también otras arquitecturas, que se mencionan aquí porque aparecen en los antecedentes del Capítulo 5:

- **Redes convolucionales** (CNN). Aplican filtros que se desplazan sobre la imagen y detectan patrones locales —bordes, texturas, formas— de manera independiente de su posición. Son el estándar para el procesamiento de imágenes.
- **Redes convolucionales 3D**. Extienden los filtros a la dimensión temporal y procesan bloques de cuadros de video, capturando apariencia y movimiento a la vez. Las variantes R(2+1)D separan la convolución espacial de la temporal (Tran et al., 2018). Logran los mejores resultados sobre LSA64, pero requieren mucho más cómputo que un modelo sobre landmarks.
- **Transformers**. Reemplazan la recurrencia por mecanismos de atención que relacionan directamente todos los elementos de la secuencia entre sí (Vaswani et al., 2017). Se proponen como trabajo futuro en las conclusiones.

