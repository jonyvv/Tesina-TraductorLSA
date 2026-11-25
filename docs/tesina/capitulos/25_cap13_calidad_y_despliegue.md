# Capítulo 13 — Aseguramiento de la calidad y despliegue

Este capítulo describe las pruebas automatizadas del sistema, las consideraciones de seguridad y privacidad, y el empaquetado del sistema y el esquema con el que se publica en internet.

## 13.1 Pruebas automatizadas

El proyecto cuenta con **132 pruebas automatizadas** escritas con pytest, organizadas en dos carpetas: `tests/`, para el backend y el módulo compartido, y `ml/tests/`, para el pipeline de datos y entrenamiento de LSA64. Todas se ejecutan sin cámara, sin un modelo entrenado y, salvo las de integración, sin MediaPipe: utilizan **dobles de prueba** —objetos que imitan la interfaz de un componente real con un comportamiento controlado— y datos sintéticos. Esto permite ejecutarlas en cualquier máquina en menos de un minuto. La @tab:pruebas resume su distribución.

Tabla: Distribución de las pruebas automatizadas. {#tab:pruebas}

| Archivo | Pruebas | Qué verifica |
|---|---|---|
| `tests/test_features.py` | 14 | Invarianza a traslación y escala, posiciones fijas por mano, indicador de presencia, serialización de los puntos para el dibujo |
| `tests/test_modelo_lse.py` | 14 | Carga de modelos, rechazo de esquemas incompatibles, correspondencia entre columnas y clases |
| `tests/test_traductor_service.py` | 20 | Suavizado, separación de confianza y estabilidad, aislamiento entre sesiones, ventana temporal |
| `tests/test_backend_e2e.py` | 9 | Arranque de la aplicación, servicios HTTP, ida y vuelta de un cuadro por WebSocket, conexiones simultáneas |
| `tests/test_cors.py` | 9 | Orígenes permitidos: lectura de la variable de entorno, aceptación y rechazo del encabezado `Origin` en el WebSocket (sección 13.2) |
| `ml/tests/test_lsa64_data.py` | 9 | Lectura de la nomenclatura de LSA64, etiqueta y persona de cada video, reparto por persona, guardado e invalidación del caché |
| `ml/tests/test_lsa64_training.py` | 9 | Uso del caché sin releer los videos, rechazo de cachés incompatibles, filtrado de cuadros vacíos, longitud mínima, detección de fuga |
| `ml/tests/test_lsa64_loso.py` | 8 | Una partición por persona, particiones completas y disjuntas, determinismo, detección de una fuga inyectada |
| `ml/tests/test_lsa64_augment.py` | 15 | Aumentación: identidad con las opciones apagadas, que no toque los cuadros vacíos ni el indicador de presencia, determinismo con semilla, sólo en entrenamiento |
| `ml/tests/test_lsa64_labels_map.py` | 6 | Que las glosas se apliquen al final sobre las etiquetas canónicas, conserven el orden y nunca lleguen al caché |
| `ml/tests/test_features_posicion.py` | 12 | Variante v2 del vector: índices derivados del ancho real, compatibilidad con v1 |
| `ml/tests/test_baseline_rf.py` | 7 | Resumen de secuencias de la línea base: largo y orden de los agregados, secuencias de un cuadro, que el resumen descarte el orden temporal |
| **Total** | **132** | |

### 13.1.1 Pruebas de regresión

Varias de las pruebas son **pruebas de regresión**: reproducen un error real encontrado durante el desarrollo, fallan con el código defectuoso y pasan con el corregido. Cada una documenta, en su descripción, qué fallaba y por qué. Algunos ejemplos:

- Que un modelo rechazado por esquema incompatible no quede cargado (sección 11.7).
- Que el suavizado decaiga cuando la persona baja las manos (sección 11.4.2).
- Que dos conexiones simultáneas no compartan el búfer de suavizado (sección 8.6).
- Que la etiqueta de un video de LSA64 sea la seña y no la persona (sección 9.11).
- Que aplicar las glosas no cambie el índice de ninguna clase (sección 10.7.1).

El valor de estas pruebas no está sólo en verificar el código actual, sino en impedir que un cambio futuro reintroduzca un error que ya costó encontrar.

### 13.1.2 Pruebas de integración

Las pruebas de `test_backend_e2e.py` levantan la aplicación completa con el cliente de pruebas de FastAPI y ejercitan los servicios reales: consultan `/health` y `/model/info`, verifican que se sirvan los archivos del cliente, envían un cuadro JPEG por WebSocket y validan la estructura de la respuesta, comprueban que un cuadro corrupto produzca un mensaje de error sin cerrar la conexión y que dos conexiones simultáneas no se interfieran. Requieren el archivo de pesos de MediaPipe; si no está, se omiten automáticamente en lugar de fallar.

### 13.1.3 Limitaciones de las pruebas

Las pruebas verifican la lógica del sistema, pero no la calidad del reconocimiento con personas reales frente a una cámara, que sólo puede evaluarse experimentalmente (Capítulo 14). Tampoco hay pruebas automatizadas del código JavaScript del cliente. [[PENDIENTE: configurar la ejecución automática de las pruebas en cada envío al repositorio, por ejemplo con GitHub Actions.]]

## 13.2 Seguridad y privacidad

- **Sin almacenamiento de video.** Los cuadros recibidos se decodifican, se procesan en memoria y se descartan. El servidor no guarda imágenes ni secuencias de los usuarios (requisito RNF-06).
- **Usuario sin privilegios.** El contenedor ejecuta la aplicación con un usuario sin privilegios de administrador, de modo que una eventual vulnerabilidad en una dependencia tenga un alcance limitado.
- **Carga de modelos.** El formato de PyTorch permite, en general, ejecutar código arbitrario al cargar un archivo de pesos. El backend sólo carga el modelo versionado en el propio repositorio, generado por los scripts del proyecto, y nunca archivos provistos por los usuarios.
- **Orígenes permitidos (CORS).** Durante el desarrollo, el servidor acepta solicitudes de cualquier origen. Para la versión desplegada se restringe al dominio de la aplicación mediante la variable de entorno `LSA_CORS_ORIGINS`, que se configura en el entorno del servicio sin modificar el código. La restricción se aplica en dos lugares, porque el middleware de CORS **no interviene en las conexiones WebSocket**: el navegador no aplica la política de mismo origen al abrirlas, de modo que el punto de acceso `/ws/translate` —donde ocurre todo el procesamiento— valida por su cuenta el encabezado `Origin` y rechaza la conexión antes de aceptarla si no coincide. Sin esa segunda validación, la restricción sólo alcanzaría a los puntos de acceso informativos y cualquier sitio podría consumir el modelo desde el navegador de sus visitantes, con el costo de cómputo que eso implica para el único equipo que lo sirve.
- **HTTPS.** Los navegadores sólo permiten el acceso a la cámara en contextos seguros (sección 6.4). Tanto el alojamiento estático de la interfaz como el túnel que expone el servicio proveen certificados válidos de forma automática (sección 6.6.3).

## 13.3 Contenedor

El sistema se empaqueta en una única imagen de Docker, definida en el `Dockerfile` de la raíz del repositorio, que sirve la API, el canal WebSocket y el cliente. Las decisiones principales son:

- **Imagen base liviana** (`python:3.12-slim`), con sólo las bibliotecas de sistema que MediaPipe requiere: GLib, y EGL y OpenGL ES, que su biblioteca nativa enlaza aunque la inferencia se haga en CPU. Se utiliza la versión de OpenCV sin interfaz gráfica, que no necesita las bibliotecas de ventanas. Como MediaPipe declara como dependencia la versión con interfaz gráfica, el `Dockerfile` la reemplaza por la otra después de instalar.
- **Versiones fijas.** MediaPipe, OpenCV, NumPy y PyTorch se instalan en las mismas versiones exactas que el entorno de entrenamiento. La primera construcción de la imagen, con versiones mínimas en lugar de exactas, instaló una versión mayor de MediaPipe distinta de la usada para extraer las características: el contenedor no llegó a arrancar, pero si lo hubiera hecho habría roto en silencio el requisito de preprocesamiento idéntico (RNF-08).
- **Dependencias en una capa propia.** El archivo de dependencias se copia e instala antes que el código. Docker reutiliza esa capa mientras las dependencias no cambien, de modo que modificar el código no obliga a reinstalar MediaPipe y PyTorch en cada construcción.
- **Modelo de MediaPipe descargado durante la construcción.** Si se descargara al arrancar, cada despliegue y cada reinicio del contenedor dependerían de que el servidor de Google sea accesible en ese momento. Descargándolo en la construcción, la imagen queda autocontenida.
- **Modelo entrenado incluido en la imagen.** El modelo se versiona junto con el código y se copia a la imagen; nunca se entrena en producción.
- **Puerto configurable.** El contenedor respeta la variable de entorno `PORT` y usa el 8000 en forma local, que es el que se expone a través del túnel.

### 13.3.1 Tamaño de la imagen

PyTorch es necesario para servir la BiLSTM, pero la distribución por defecto incluye las bibliotecas de CUDA para GPU, que ocupan varios gigabytes y no se usan: la inferencia de una ventana de cuadros por vez se resuelve en CPU en pocos milisegundos. El `Dockerfile` instala la distribución sólo para CPU, desde el índice de paquetes correspondiente, antes que el resto de las dependencias, lo que reduce varios gigabytes la imagen y acorta su construcción y su arranque.

La imagen resultante ocupa 1,9 GB sin comprimir y 571 MB comprimida, que es lo que se transfiere al publicarla o descargarla. Las bibliotecas de Python explican 1,54 GB, la mayor parte por PyTorch, MediaPipe y OpenCV; les siguen las bibliotecas gráficas que pide MediaPipe, con unos 210 MB porque la implementación de EGL de la distribución arrastra el compilador LLVM, y la imagen base de Python, con unos 135 MB. El contenedor responde al servicio de estado unos 15 segundos después de arrancar, el tiempo que tardan en cargarse el detector de manos y el modelo.

## 13.4 Despliegue

El requisito de costo (RNF-05) es publicar el sistema sin costo de infraestructura. La primera opción fue desplegar la imagen completa en la capa gratuita de una plataforma como servicio (sección 6.6.2), y se descartó por dos razones: el entorno de inferencia, aun con PyTorch para CPU, deja una imagen cercana a los límites de espacio de esas capas; y la suspensión por inactividad obliga a recargar el modelo y el detector al volver, de modo que la primera seña de cada sesión paga un arranque en frío, justamente cuando se la necesita.

El esquema adoptado separa las dos mitades (sección 6.6.3): la interfaz web se publica como sitio estático servido por HTTPS, y el servicio de inferencia —el contenedor de la sección 13.3— se ejecuta en una computadora de los autores, expuesto mediante un túnel con certificado TLS. El cliente puede apuntar a ese dominio porque la dirección del servicio se toma, en ese orden, del parámetro `backend` de la página, de la que el usuario haya guardado en el navegador o del archivo `config.js` que acompaña a la interfaz publicada; sin ninguna de las tres se usa el mismo origen que sirvió la página, que es el caso del desarrollo local. Como la capa gratuita de ngrok asigna a cada cuenta un dominio fijo, que no cambia al reiniciar el túnel, esa dirección se configura una sola vez.

A cambio, la disponibilidad queda atada a esa computadora y al túnel, la capa gratuita del túnel limita el tráfico mensual y los usuarios simultáneos están limitados, porque cada cuadro de cada conexión ejecuta MediaPipe: trasladar la detección al cliente (sección 8.8.1) es la vía para escalar.

[[PENDIENTE: despliegue todavía no realizado. Se documentarán acá la dirección pública de la interfaz, los pasos seguidos y la latencia medida desde el navegador contra el servicio expuesto por el túnel.]]

