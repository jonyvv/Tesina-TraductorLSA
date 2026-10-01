# Capítulo 11 — Backend de inferencia

El backend recibe los cuadros de video, los convierte en predicciones y las devuelve al cliente. Este capítulo describe su organización, su ciclo de vida, el manejo de conexiones concurrentes, el suavizado temporal de las predicciones y los mecanismos que impiden servir un modelo incompatible.

## 11.1 Organización del código

El backend es una aplicación FastAPI ubicada en `backend/app/`, cuyas clases se presentaron en la sección 8.6. Su punto de entrada, `main.py`, construye el grafo de dependencias al arrancar: crea el `Preprocesador` y el `ModeloLSE`, los combina en un `TraductorService` y entrega este último al `WebSocketHandler`. Ninguna clase crea por su cuenta los objetos que necesita; los recibe ya construidos, lo que permite reemplazarlos por dobles de prueba en las pruebas automatizadas (sección 13.1).

## 11.2 Arranque y servicios expuestos

### 11.2.1 Selección del modelo

Al arrancar, el backend decide qué modelo cargar con el siguiente orden de prioridad:

1. El archivo indicado en la variable de entorno `LSA_MODEL_PATH`, si está definida. Permite probar otro modelo sin modificar el código.
2. La BiLSTM de LSA64 (`modelo_lsa64_lstm.pt`), si existe.
3. El modelo de configuraciones estáticas (`modelo_lse.joblib`).

### 11.2.2 Arranque tolerante a fallas

En un repositorio recién clonado pueden faltar el archivo de pesos de MediaPipe —que se descarga aparte— o el modelo entrenado. Una primera versión del backend fallaba al arrancar en esos casos, con un error poco informativo. La versión actual captura esas situaciones, registra un aviso que indica el comando que hay que ejecutar para resolverlas, y arranca igual. Los servicios que dependen de los componentes faltantes responden con un mensaje de error explicativo en lugar de fallar. La inicialización se realiza en el manejador del ciclo de vida de la aplicación (`lifespan`), que además libera los recursos nativos de MediaPipe al apagar el servidor.

### 11.2.3 Servicios

El backend expone dos servicios HTTP y un canal WebSocket. FastAPI genera automáticamente su documentación navegable, que muestra la @fig:swagger.

- `GET /health`: estado del servicio. Informa si el preprocesador y el modelo están listos, qué modelo está cargado, sus clases, y si el sistema está en condiciones de traducir. Es el servicio que consultan las plataformas en la nube para saber si el contenedor está sano.
- `GET /model/info`: información que el cliente necesita para funcionar: las clases, el umbral de confianza, la versión de las características, si el modelo requiere una secuencia de cuadros y de qué longitud, y las conexiones del esqueleto de la mano para dibujarlo. El esqueleto se sirve desde el backend, en lugar de estar escrito también en el JavaScript, por la misma razón por la que existe el módulo compartido: una única definición, sin copias que puedan quedar desactualizadas.
- `WS /ws/translate`: el canal de traducción, descripto a continuación.

Además, el backend sirve los archivos estáticos del cliente en la raíz del sitio, de modo que todo el sistema funciona como un único servicio.

![Documentación de la API generada automáticamente por FastAPI.](figuras/cap_swagger.png){ancho=14 #fig:swagger}

## 11.3 El canal WebSocket y la concurrencia

Cuando un cliente se conecta a `/ws/translate`, el `WebSocketHandler` acepta la conexión y crea una `SesionTraduccion` exclusiva para ella. Luego entra en un ciclo: espera un mensaje binario con un cuadro JPEG, lo traduce y responde con un mensaje de texto en formato JSON. El ciclo termina cuando el cliente se desconecta, momento en que la sesión se descarta.

Como se explicó en la sección 6.5.2, el procesamiento de un cuadro es cómputo sincrónico y pesado. Si se ejecutara directamente en el manejador asincrónico, bloquearía el bucle de eventos y serializaría a todos los usuarios conectados. Por eso la traducción se delega a un hilo auxiliar mediante `run_in_threadpool`:

```
resultado = await run_in_threadpool(sesion.traducir, datos)
```

El detector de MediaPipe, en cambio, se comparte entre todas las conexiones —crear uno por conexión multiplicaría el uso de memoria— y no admite llamadas simultáneas desde varios hilos. El `Preprocesador` protege su uso con un cerrojo (`threading.Lock`). El resultado es que la detección de manos de distintos usuarios se ejecuta de a una por vez, pero el resto del trabajo —decodificación, red neuronal, comunicación— avanza en paralelo, y una conexión lenta no bloquea a las demás.

Si el procesamiento de un cuadro falla —por ejemplo, porque llegan bytes que no son una imagen válida—, el error se registra y se devuelve al cliente como un mensaje con un campo `error`, sin cerrar la conexión.

## 11.4 La sesión de traducción

`SesionTraduccion` contiene todo el estado de una conversación: el búfer de suavizado, la ventana de cuadros y el historial de señas reconocidas.

### 11.4.1 Formato de la respuesta

Por cada cuadro recibido, la sesión devuelve un mensaje como el siguiente:

```
{
  "seña": "Leche",
  "confianza": 0.9132,
  "estabilidad": 0.8,
  "valida": true,
  "landmarks": [{"mano": "Right", "puntos": [[0.4812, 0.6623], ...]}],
  "manos": 1,
  "ms_servidor": 41.27
}
```

Los puntos de las manos viajan sólo con sus coordenadas x e y, redondeadas a cuatro decimales: con dos manos son 84 números, menos de 1 KB, una fracción despreciable frente al cuadro JPEG que se envió. El campo `ms_servidor` es el tiempo de procesamiento medido en el servidor, que el cliente usa para separar el costo de cómputo del costo de red (sección 12.5).

### 11.4.2 Suavizado por voto mayoritario

Una predicción por cuadro, presentada directamente, parpadea: un cuadro borroso o una transición entre señas producen predicciones aisladas incorrectas. La sesión estabiliza el resultado con un **voto mayoritario** sobre las últimas 10 predicciones. La etiqueta que se informa es la más votada, y sólo se informa si el búfer tiene al menos 5 predicciones.

En la primera implementación, sólo se agregaban al búfer las predicciones válidas. El efecto era que el búfer nunca se vaciaba: cuando la persona bajaba las manos, el voto seguía devolviendo la última seña como válida indefinidamente. La corrección fue registrar también los cuadros sin una seña válida, como votos "vacíos". Así, cuando la persona deja de señar, los votos vacíos ganan la mayoría y el resultado se apaga solo.

### 11.4.3 Confianza y estabilidad

La versión inicial del suavizado reemplazaba la confianza del modelo por la proporción de votos de la etiqueta ganadora, de modo que el campo `confianza` significaba cosas distintas según el punto del sistema en que se lo mirara, y ambos valores se comparaban contra el mismo umbral. Se separaron en dos campos:

- **Confianza**: la probabilidad que asigna el modelo a la seña, promediada sobre los cuadros que votaron por ella. Responde a la pregunta "¿qué tan seguro está el modelo?".
- **Estabilidad**: la proporción de votos de la seña ganadora en el búfer. Responde a "¿qué tan consistente fue la detección en el tiempo?".

Una predicción se considera **válida** sólo si ambos valores superan su umbral (0,6 en ambos casos). Además de ser más correcta, la separación permite distinguir dos situaciones con causas y soluciones distintas: un modelo que duda —confianza baja— y un modelo seguro cuya detección parpadea —estabilidad baja—.

## 11.5 La ventana temporal

Con un modelo de secuencias, una predicción requiere varios cuadros. La sesión acumula los vectores de los cuadros con manos detectadas en una ventana deslizante de longitud fija y, cuando la ventana se completa, la entrega al modelo; a partir de ahí, cada cuadro nuevo desplaza al más antiguo y produce una predicción. Mientras la ventana no está completa, la sesión devuelve una predicción vacía, pero con los puntos de las manos, para que el cliente pueda dibujar el esqueleto desde el primer cuadro.

Si un cuadro llega sin manos, la ventana se vacía: la secuencia acumulada dejó de ser continua y se considera que la seña terminó. La ventana, como el búfer de suavizado, es estado de una conversación y vive en la sesión; si estuviera en el servicio compartido, las señas de dos usuarios se mezclarían en la misma secuencia.

La longitud de la ventana se toma del propio modelo: es la longitud mínima de secuencia con que fue entrenado, 8 cuadros. Esta elección es la principal limitación conocida de la integración en tiempo real y se analiza en la sección 14.8: el modelo fue entrenado con señas completas muestreadas a 30 cuadros por segundo, y en la aplicación recibe ventanas de 8 cuadros a la frecuencia de envío del cliente.

## 11.6 Modelos intercambiables

`ModeloLSE` es la única clase del backend que conoce cómo se carga y se consulta un modelo, y delega los detalles de cada formato en un **adaptador**. Una fábrica elige el adaptador según la extensión del archivo:

- `SklearnJoblibAdapter`, para los modelos de scikit-learn (`.joblib`) de las configuraciones estáticas. Además de cargar el modelo, verifica que las columnas de probabilidades correspondan a las clases en el orden correcto: en scikit-learn, la columna i corresponde a `classes_[i]`, que no necesariamente es la clase i.
- `TorchCheckpointAdapter`, para la BiLSTM de PyTorch (`.pt`). Reconstruye la red con la definición del módulo compartido y los metadatos del checkpoint, y la pone en modo de evaluación (sin dropout).

Gracias a este diseño, migrar de un modelo estático a uno secuencial, o agregar un tercer formato, no requiere modificar el servicio, el manejador WebSocket ni el cliente.

## 11.7 Validación de compatibilidad

Al cargar un modelo, `ModeloLSE` verifica que su versión de características y la longitud de su vector coincidan con las que genera el backend. Si no coinciden, rechaza el modelo con un mensaje que explica la diferencia y cómo resolverla. Esta verificación existe porque el desajuste entre el preprocesamiento del entrenamiento y el de la inferencia no produce errores, sino predicciones incorrectas (sección 8.5.1).

La implementación de esta validación tuvo un error sutil, detectado y corregido con una prueba de regresión. En la primera versión, el modelo se asignaba al objeto antes de validarlo. Si la validación fallaba, la excepción se registraba, pero el modelo rechazado quedaba igualmente cargado: el servicio `/health` informaba que había un modelo listo y el sistema servía precisamente el modelo que acababa de rechazar. La corrección consiste en realizar todas las validaciones sobre variables locales y asignar el modelo al objeto sólo cuando todas pasaron.

