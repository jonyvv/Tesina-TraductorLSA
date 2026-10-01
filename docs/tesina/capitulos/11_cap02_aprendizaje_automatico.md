# Capítulo 2 — Inteligencia artificial y aprendizaje automático

Este capítulo presenta los conceptos de aprendizaje automático sobre los que se apoya el sistema, con énfasis en un aspecto que resultó central durante el desarrollo: cómo medir correctamente el desempeño de un modelo. Los detalles de las redes neuronales se tratan en el Capítulo 3.

## 2.1 Inteligencia artificial

Russell y Norvig (2021) organizan las definiciones de **inteligencia artificial** (IA) según dos ejes: si se interesan por el pensamiento o por el comportamiento, y si toman como referencia el desempeño humano o un ideal de racionalidad. La perspectiva que adopta este trabajo es la del comportamiento: un sistema es útil si, dada una entrada —el video de una persona haciendo una seña—, produce la salida correcta —la palabra que esa seña representa—, sin que importe si el mecanismo interno se parece al humano.

Durante décadas, la IA se construyó principalmente con reglas escritas a mano por especialistas. Ese enfoque funciona cuando el conocimiento del dominio puede expresarse de forma explícita, pero fracasa en tareas perceptivas como reconocer una seña: nadie puede escribir las reglas que distinguen la seña "Leche" de la seña "Agua" a partir de los píxeles de una imagen, aunque cualquier persona que sepa LSA las distinga sin esfuerzo. El aprendizaje automático resuelve ese problema invirtiendo el enfoque.

## 2.2 Aprendizaje automático

El **aprendizaje automático** (*machine learning*) es la rama de la IA que estudia algoritmos capaces de mejorar su desempeño en una tarea a partir de la experiencia, es decir, de los datos (Géron, 2022). En lugar de programar las reglas, se programa un procedimiento que las infiere a partir de ejemplos.

Según el tipo de experiencia disponible, se distinguen tres grandes familias:

- **Aprendizaje supervisado.** Cada ejemplo de entrenamiento está acompañado de la respuesta correcta, llamada **etiqueta**. El algoritmo aprende una función que asigna entradas a etiquetas. Es el caso de este trabajo: cada video de LSA64 viene etiquetado con la seña que representa.
- **Aprendizaje no supervisado.** Los ejemplos no tienen etiqueta y el objetivo es descubrir estructura en los datos, como grupos de ejemplos similares.
- **Aprendizaje por refuerzo.** Un agente aprende a actuar en un entorno a partir de recompensas y castigos.

### 2.2.1 Clasificación supervisada

Dentro del aprendizaje supervisado, la **clasificación** es la tarea de asignar cada entrada a una de un conjunto finito de clases. Formalmente, a partir de un conjunto de entrenamiento de pares (x, y), donde x es la representación de la entrada e y su clase, se busca una función f tal que f(x) sea igual a y para entradas nuevas, no vistas durante el entrenamiento.

La representación x se construye a partir de **características** (*features*): valores numéricos que describen la entrada. Elegir buenas características es una de las decisiones más importantes de cualquier sistema de aprendizaje automático. En este trabajo, cada cuadro de video se representa con 138 valores derivados de la geometría de las manos (ver el Capítulo 9), y cada seña es una secuencia de esos vectores.

Los clasificadores modernos no suelen devolver sólo una clase, sino una distribución de probabilidad sobre todas las clases. La clase con mayor probabilidad es la predicción, y esa probabilidad puede interpretarse como la **confianza** del modelo. El sistema aprovecha ese valor para descartar predicciones dudosas (sección 11.4).

## 2.3 Generalización y sobreajuste

El objetivo del aprendizaje no es memorizar el conjunto de entrenamiento, sino **generalizar**: funcionar bien con datos nuevos. Un modelo puede fallar en ese objetivo de dos maneras opuestas (Goodfellow et al., 2016):

- **Subajuste** (*underfitting*): el modelo es demasiado simple para capturar la relación entre entradas y clases, y comete muchos errores incluso sobre los datos de entrenamiento.
- **Sobreajuste** (*overfitting*): el modelo es tan flexible que aprende también las particularidades irrelevantes del conjunto de entrenamiento —el ruido, o los rasgos de las personas que lo grabaron— y su desempeño cae con datos nuevos.

El síntoma típico del sobreajuste es una brecha creciente entre el error de entrenamiento, que sigue bajando, y el error sobre datos no vistos, que se estanca o sube. Ese patrón se observó en el modelo de esta tesina y se analiza en la sección 14.3. Las técnicas para combatirlo se denominan **regularización** y se describen en la sección 3.4.

## 2.4 Partición de los datos

Para estimar cómo se comportará el modelo con datos nuevos, es imprescindible reservar una parte de los datos que el modelo nunca vea durante el entrenamiento. La práctica habitual es dividir los datos en tres conjuntos (Géron, 2022):

- **Entrenamiento**: los ejemplos con los que se ajustan los parámetros del modelo.
- **Validación**: ejemplos que se usan durante el desarrollo para tomar decisiones —elegir hiperparámetros, decidir cuándo detener el entrenamiento— sin contaminar la medición final.
- **Prueba** (*test*): ejemplos que se usan una única vez, al final, para medir el desempeño. Si el conjunto de prueba influye en cualquier decisión, deja de ser una estimación honesta.

### 2.4.1 Validación cruzada

Cuando los datos son pocos, un único reparto produce una estimación inestable: el resultado depende de qué ejemplos cayeron, por azar, en el conjunto de prueba. La **validación cruzada de k particiones** (*k-fold cross-validation*) divide los datos en k grupos, entrena k modelos —cada uno dejando afuera un grupo distinto para la prueba— y promedia los resultados (Kohavi, 1995). Además de la media, el desvío entre particiones informa cuánto varía el desempeño según los datos de prueba.

### 2.4.2 Fuga de datos y dependencia entre ejemplos

La validación cruzada supone que los ejemplos son independientes entre sí. Esa suposición no se cumple cuando hay grupos de ejemplos fuertemente relacionados. En un conjunto de datos de lengua de señas, las cinco repeticiones de una seña hechas por la misma persona son muy parecidas entre sí: comparten la forma de las manos, la estatura, el ritmo y el estilo personal de quien seña.

Si esas repeticiones se reparten al azar entre entrenamiento y prueba, el modelo puede acertar en la prueba simplemente porque ya vio a esa misma persona haciendo esa misma seña. A esto se lo denomina **fuga de datos** (*data leakage*): información del conjunto de prueba que se filtra al entrenamiento y produce una estimación optimista que no se sostiene en la práctica (Hastie et al., 2009). La solución es particionar **por grupo**: todos los ejemplos de una misma persona deben quedar del mismo lado del reparto.

### 2.4.3 Evaluación dependiente e independiente del sujeto

En el reconocimiento de lengua de señas, esta distinción tiene nombre propio:

- **Evaluación dependiente del sujeto** (*signer-dependent*): las personas que aparecen en la prueba también aparecen en el entrenamiento. Mide qué tan bien reconoce el modelo a personas que ya conoce.
- **Evaluación independiente del sujeto** (*signer-independent*): las personas de la prueba nunca aparecen en el entrenamiento. Mide qué tan bien generaliza a personas nuevas, que es exactamente la situación de uso de una aplicación pública.

La variante más exigente de la evaluación independiente es la validación **leave-one-subject-out** (LOSO): con N personas, se entrenan N modelos, cada uno dejando afuera a una persona distinta para la prueba. Cada ejemplo del conjunto de datos es predicho exactamente una vez por un modelo que nunca vio a quien lo realizó. La @fig:protocolos ilustra la diferencia entre ambos esquemas.

![Distribución de los videos en un esquema dependiente del sujeto (izquierda), donde los videos de prueba de cada persona se mezclan con los de entrenamiento, y en un fold de LOSO (derecha), donde una persona completa queda fuera del entrenamiento.](figuras/fig_protocolos.png){ancho=15 #fig:protocolos}

Como se verá en el Capítulo 5, las cifras de exactitud reportadas en la literatura sobre LSA64 varían enormemente según el protocolo: los trabajos con evaluación dependiente del sujeto reportan valores cercanos al 99 %, mientras que con evaluación independiente los valores son sensiblemente menores. Comparar cifras obtenidas con protocolos distintos no tiene sentido, y por eso esta tesina declara explícitamente el suyo.

## 2.5 Métricas de evaluación

### 2.5.1 Exactitud

La **exactitud** (*accuracy*) es la proporción de ejemplos clasificados correctamente:

> exactitud = predicciones correctas / total de predicciones

Es la métrica más intuitiva y la más usada en la literatura de reconocimiento de señas aisladas, lo que facilita la comparación. En conjuntos balanceados, como LSA64 —donde cada seña tiene el mismo número de videos—, es además una métrica representativa. Su principal debilidad es que resume todo en un número y oculta qué clases fallan y con cuáles se confunden.

### 2.5.2 Precisión, exhaustividad y F1

Para una clase c, la **precisión** es la proporción de las predicciones "c" que eran efectivamente c, y la **exhaustividad** (*recall*) es la proporción de los ejemplos de c que el modelo reconoció como tales. La medida **F1** es la media armónica de ambas. Promediando estas métricas sobre todas las clases (promedio *macro*) se obtiene una visión que no se deja dominar por las clases más frecuentes.

### 2.5.3 Matriz de confusión

La **matriz de confusión** es una tabla en la que la fila i y la columna j contienen la cantidad de ejemplos de la clase i que el modelo predijo como clase j. La diagonal concentra los aciertos; cada valor fuera de la diagonal es un tipo de error concreto. Para un problema de 64 clases es la herramienta de análisis más informativa, porque permite pasar de "el modelo acierta el 80 %" a "el modelo confunde sistemáticamente Aprender con Caramelo", que es un hallazgo interpretable y accionable (sección 14.4).

## 2.6 Bosques aleatorios

No todo problema de clasificación requiere redes neuronales. El **bosque aleatorio** (*random forest*) propuesto por Breiman (2001) combina muchos árboles de decisión, cada uno entrenado sobre una muestra aleatoria de los ejemplos y considerando en cada división un subconjunto aleatorio de las características. La predicción final es el voto mayoritario de los árboles. Es un algoritmo robusto, rápido de entrenar, poco sensible a los hiperparámetros y difícil de sobreajustar, lo que lo convierte en una **línea base** natural: un punto de comparación simple contra el cual justificar el uso de un modelo más complejo.

Su limitación para este problema es que trabaja con vectores de longitud fija, y no con secuencias de longitud variable. Para aplicarlo a una seña hay que resumir la secuencia en un único vector —por ejemplo, con la media, el desvío, el mínimo y el máximo de cada característica a lo largo del tiempo—, lo que descarta el orden temporal de los movimientos. En el sistema, el bosque aleatorio se utiliza para las configuraciones estáticas, donde un único cuadro basta (sección 8.4), y se propone como línea base para las señas dinámicas (sección 14.7).

