# Capítulo 8 — Análisis y diseño del sistema

Este capítulo inicia la segunda parte del trabajo. Presenta los requisitos del sistema, sus casos de uso y la arquitectura adoptada, junto con las decisiones de diseño más relevantes y las alternativas descartadas.

## 8.1 Requisitos funcionales

La @tab:rf enumera los requisitos funcionales, derivados de los objetivos y alcances del proyecto.

Tabla: Requisitos funcionales del sistema. {#tab:rf}

| Código | Requisito | Estado |
|---|---|---|
| RF-01 | Capturar video de la cámara del dispositivo desde el navegador, con permiso del usuario | Implementado |
| RF-02 | Reconocer, en tiempo real, las 64 señas del vocabulario de LSA64 | Implementado |
| RF-03 | Mostrar la seña reconocida junto con la confianza del modelo y la estabilidad de la detección | Implementado |
| RF-04 | Acumular las señas reconocidas en un texto que el usuario pueda copiar, corregir o borrar | Implementado |
| RF-05 | Dibujar el esqueleto de las manos detectadas sobre el video, con opción a ocultarlo | Implementado |
| RF-06 | Permitir ajustar la frecuencia de envío de cuadros | Implementado |
| RF-07 | Informar al usuario el estado de la conexión y reconectar automáticamente si se interrumpe | Implementado |
| RF-08 | Reconocer configuraciones manuales estáticas (alfabeto manual) | Parcial: pipeline implementado, datos pendientes |
| RF-09 | Exponer el estado del servicio y la información del modelo cargado | Implementado |

## 8.2 Requisitos no funcionales

Tabla: Requisitos no funcionales del sistema. {#tab:rnf}

| Código | Categoría | Requisito |
|---|---|---|
| RNF-01 | Desempeño | La latencia entre el envío de un cuadro y la recepción de su resultado debe permitir una interacción fluida (objetivo: menos de 300 ms en una red local) |
| RNF-02 | Exactitud | El desempeño del modelo debe medirse sobre personas no vistas en el entrenamiento |
| RNF-03 | Portabilidad | El cliente debe funcionar en navegadores modernos de escritorio y móviles, sin instalación |
| RNF-04 | Accesibilidad | La interfaz debe cumplir los criterios de nivel AA de WCAG 2.2 aplicables |
| RNF-05 | Costo | El despliegue debe funcionar dentro de la capa gratuita de una plataforma en la nube |
| RNF-06 | Privacidad | El video de los usuarios no debe almacenarse |
| RNF-07 | Concurrencia | Usuarios simultáneos no deben interferir entre sí en sus resultados |
| RNF-08 | Mantenibilidad | El preprocesamiento debe estar definido en un único lugar para entrenamiento e inferencia |
| RNF-09 | Robustez | El servidor debe arrancar aunque falte un componente, informando el problema en lugar de fallar |

## 8.3 Actores y casos de uso

El sistema tiene dos actores humanos. La **persona que seña** se ubica frente a la cámara y realiza las señas; la **persona que lee** observa la traducción en la pantalla. En el uso típico —un mostrador de atención, por ejemplo— ambas comparten el mismo dispositivo. Un tercer actor, el **desarrollador**, interactúa con el sistema para entrenar y publicar nuevos modelos. La @tab:casos resume los casos de uso.

Tabla: Casos de uso principales. {#tab:casos}

| Caso de uso | Actor | Descripción |
|---|---|---|
| CU-01 Iniciar traducción | Persona que seña | Inicia la cámara; el sistema solicita permiso, abre la conexión con el servidor y comienza a enviar cuadros |
| CU-02 Realizar una seña | Persona que seña | Realiza una seña frente a la cámara; el sistema muestra el esqueleto de las manos y, al estabilizarse la predicción, la seña reconocida |
| CU-03 Leer la traducción | Persona que lee | Lee la seña actual y el texto acumulado |
| CU-04 Editar el texto | Cualquiera | Copia el texto, borra la última seña o limpia todo |
| CU-05 Ajustar opciones | Cualquiera | Cambia la frecuencia de envío, oculta el esqueleto o cambia el tema visual |
| CU-06 Detener | Cualquiera | Detiene la cámara y cierra la conexión |
| CU-07 Entrenar y publicar un modelo | Desarrollador | Extrae características, entrena, evalúa y copia el modelo al servidor |

## 8.4 Señas estáticas y dinámicas

Una decisión temprana del diseño fue tratar como problemas distintos el reconocimiento de configuraciones estáticas y el de señas con movimiento. El prototipo de escritorio que sirvió de punto de partida intentaba representar ambos con un mismo esquema, lo que en la práctica impedía entrenar correctamente cualquiera de los dos.

- **Configuraciones estáticas** (como la mayoría de las letras del alfabeto manual): un único cuadro contiene toda la información. Cada muestra es un vector de 138 valores y el problema es una clasificación tabular. Se resuelve con un bosque aleatorio o un perceptrón multicapa, que se entrenan y comparan con el script `ml/train.py`.
- **Señas dinámicas** (como las de LSA64): la información está en la evolución temporal. Cada muestra es una secuencia de vectores de longitud variable y se resuelve con una BiLSTM (Capítulo 10).

El backend sirve uno u otro tipo de modelo según el archivo cargado, sin cambios en el resto del sistema (sección 11.6). El reconocimiento dinámico sobre LSA64 es el que está completo y evaluado. Para el estático existe el pipeline completo de captura y entrenamiento, y una prueba de concepto con dos letras capturadas por uno de los autores, pero no un conjunto de datos suficiente para reportar resultados (sección 9.8).

## 8.5 Arquitectura del sistema

La arquitectura se organiza en tres capas —cliente, backend y modelo—, más un módulo de código compartido. La @fig:arquitectura presenta la vista general.

![Arquitectura del sistema. El cliente envía cuadros JPEG por WebSocket; el backend extrae las manos, clasifica la secuencia y devuelve el resultado. El código de extracción de características y la definición de la red se comparten entre el backend y el pipeline de entrenamiento.](figuras/fig_arquitectura.png){ancho=14.5 #fig:arquitectura}

- **Cliente** (`frontend/`). Una página web que captura el video, lo espeja y lo codifica como JPEG, lo envía al servidor, y presenta los resultados. No contiene lógica de reconocimiento.
- **Backend** (`backend/`). Un servicio FastAPI que recibe los cuadros por WebSocket, detecta las manos con MediaPipe, construye el vector de características, mantiene la ventana de cuadros de cada conexión, invoca al modelo y suaviza los resultados. Sirve además los archivos del cliente, de modo que todo el sistema es un único servicio.
- **Modelo** (`ml/`). Los scripts que extraen las características de LSA64, entrenan la red y la evalúan. Se ejecutan fuera de línea; su producto es un archivo de pesos que el backend carga al arrancar.
- **Código compartido** (`common/`). La extracción y normalización de características (`features.py`) y la definición de la red (`models/lsa64.py`).

### 8.5.1 El módulo compartido

El módulo `common/` no figuraba en el diseño inicial y es, probablemente, la decisión de arquitectura más importante del proyecto. En el prototipo de escritorio, la extracción de características estaba escrita dos veces: una en el script de captura y otra en el de reconocimiento. Si una de las dos copias se modifica y la otra no, el modelo recibe durante el uso vectores distintos de los que vio al entrenar. No se produce ningún error: el modelo simplemente predice mal, y el síntoma —baja exactitud— suele atribuirse al modelo o a la falta de datos, no al preprocesamiento.

Con un único módulo importado por ambos lados, ese tipo de divergencia es imposible por construcción. El módulo define además una **versión del esquema de características** (`FEATURE_VERSION`), que se guarda dentro de cada modelo entrenado y que el backend verifica al cargarlo: si no coincide con la versión vigente, el servidor se niega a usar el modelo y lo informa en lugar de servir predicciones sin sentido. La sección 9.7 muestra que el módulo compartido, aun así, no alcanzó para prevenir todos los desajustes.

## 8.6 Modelo de clases del backend

La @tab:clases presenta las clases del backend y sus responsabilidades. El diseño parte de un diagrama de clases elaborado al inicio del proyecto, con dos modificaciones que se justifican a continuación.

Tabla: Clases del backend y sus responsabilidades. {#tab:clases}

| Clase | Archivo | Responsabilidad |
|---|---|---|
| `Frame` | `backend/app/frame.py` | Encapsula los bytes recibidos y los decodifica a una matriz de imagen |
| `Preprocesador` | `backend/app/preprocesador.py` | Ejecuta MediaPipe sobre la imagen y construye el vector de características |
| `ModeloLSE` | `backend/app/modelo_lse.py` | Carga el modelo, valida su compatibilidad y produce predicciones |
| `ModeloAdaptador` y subclases | `backend/app/modelos/` | Encapsulan el formato del modelo (scikit-learn o PyTorch) |
| `Prediccion` | `backend/app/prediccion.py` | Objeto de valor con la etiqueta, la confianza y la estabilidad |
| `TraductorService` | `backend/app/traductor_service.py` | Orquesta preprocesamiento y modelo; no guarda estado de clientes |
| `SesionTraduccion` | `backend/app/traductor_service.py` | Estado de una conexión: suavizado, ventana de cuadros e historial |
| `WebSocketHandler` | `backend/app/websocket_handler.py` | Acepta conexiones y ejecuta el ciclo de recepción y envío |

**Primera modificación: la clase `SesionTraduccion`.** En el diagrama original, `TraductorService` guardaba el historial de predicciones como atributo propio. Pero el servicio se construye una única vez al arrancar el servidor y se comparte entre todas las conexiones, así que ese atributo era, en la práctica, estado global: con dos usuarios conectados, las predicciones de uno se mezclaban con las del otro. La solución fue separar lo que es de la aplicación —el preprocesador y el modelo, que pueden compartirse— de lo que es de cada conversación, que pasó a una clase nueva instanciada por conexión. El caso ilustra un límite de los diagramas de clases: son correctos en lo estructural, pero no distinguen entre objetos de aplicación y objetos de sesión, y esa diferencia es la que determina el comportamiento con varios usuarios.

**Segunda modificación: la cámara vive en el cliente.** El diagrama original incluía una clase `CamaraCaptura` del lado del servidor. Sin embargo, la cámara pertenece al dispositivo del usuario y sólo es accesible mediante las interfaces del navegador. La captura se implementó en JavaScript, en el cliente, y lo único que viaja al servidor son los cuadros ya capturados.

## 8.7 Dinámica de una traducción

La @fig:secuencia muestra el diagrama de secuencia de una sesión de traducción. Una vez abierta la conexión, el ciclo se repite a la frecuencia configurada en el cliente: el navegador envía un cuadro, el backend lo procesa fuera del bucle de eventos, y el resultado vuelve como un mensaje JSON que incluye la seña, su confianza, la estabilidad, los puntos de las manos para el dibujo y el tiempo de procesamiento del servidor.

![Diagrama de secuencia de una sesión de traducción.](figuras/fig_secuencia.png){ancho=14 #fig:secuencia}

## 8.8 Decisiones de diseño

### 8.8.1 Dónde ejecutar la detección de manos

La arquitectura implementada envía cuadros completos al servidor y ejecuta MediaPipe allí. Existe una alternativa: MediaPipe tiene una versión para navegador (`@mediapipe/tasks-vision`, en JavaScript y WebAssembly), que permitiría detectar las manos en el dispositivo del usuario y enviar al servidor sólo los vectores de características. La @tab:donde-mediapipe compara ambas opciones.

Tabla: Detección de manos en el servidor frente a detección en el cliente. {#tab:donde-mediapipe}

| Criterio | En el servidor (implementado) | En el cliente (alternativa) |
|---|---|---|
| Datos por cuadro | Imagen JPEG (decenas de KB) | Vector de 138 valores (menos de 1 KB) |
| Cómputo del servidor | Alto: MediaPipe por cada cuadro de cada usuario | Bajo: sólo la red neuronal |
| Requisitos del dispositivo | Mínimos | Debe ejecutar MediaPipe con fluidez |
| Garantía de preprocesamiento idéntico | Total: el mismo código Python que en el entrenamiento | Requiere una segunda implementación en JavaScript |
| Adecuación a la capa gratuita | Limitada: pocos usuarios simultáneos | Mejor |

Se eligió la detección en el servidor porque garantiza que el preprocesamiento sea idéntico al del entrenamiento —el requisito RNF-08—, que es la fuente de errores más grave identificada en el proyecto. La alternativa es más eficiente y escalable, pero duplicaría la extracción de características en dos lenguajes, precisamente lo que el módulo compartido busca evitar. Queda como mejora futura, condicionada a una verificación automática de que ambas implementaciones producen los mismos vectores.

### 8.8.2 Cliente sin marco de trabajo

El anteproyecto preveía desarrollar el cliente con React. Durante el diseño se optó por HTML, CSS y JavaScript sin marco de trabajo, separados en tres archivos por responsabilidad. La interfaz consiste en una única pantalla con un video, un panel de resultados y algunos controles: un marco de trabajo agregaría un proceso de compilación, dependencias y un segundo servidor durante el desarrollo, sin un beneficio proporcional. Además, al no requerir compilación, el backend puede servir los archivos del cliente directamente y todo el sistema se despliega como un único servicio, lo que ahorra recursos en la capa gratuita.

### 8.8.3 Otras decisiones

- **PyTorch en lugar de TensorFlow.** El anteproyecto mencionaba ambos. Se usó PyTorch por su soporte directo para secuencias de longitud variable empaquetadas (sección 3.7.1) y porque permite servir el modelo con la versión para CPU de la biblioteca.
- **Scripts en lugar de cuadernos Jupyter.** El entrenamiento se implementó como scripts de Python con parámetros de línea de comandos, en lugar de cuadernos. Los scripts se versionan mejor, se prueban con pytest y se ejecutan de forma reproducible, requisitos que surgieron al detectar los problemas de reproducibilidad de la sección 10.4.
- **Entrenamiento local en lugar de Google Colab.** La red tiene menos de 300 000 parámetros y se entrena en segundos en una CPU. El paso costoso es la extracción de landmarks con MediaPipe, que corre en CPU y se paraleliza por procesos: con los 8 núcleos de un equipo de escritorio tarda unos 47 minutos, mientras que en la capa gratuita de Colab, con dos núcleos virtuales, tardaría varias horas. Una GPU no acelera este paso.

