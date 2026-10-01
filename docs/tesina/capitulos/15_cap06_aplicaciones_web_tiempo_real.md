# Capítulo 6 — Aplicaciones web en tiempo real

El modelo de reconocimiento sólo es útil si llega a las personas. Este capítulo presenta las tecnologías que permiten llevarlo a una aplicación web: la arquitectura cliente-servidor, los protocolos de comunicación, las interfaces del navegador para acceder a la cámara, el marco de trabajo del backend, los contenedores para el despliegue y las pautas de accesibilidad.

## 6.1 Arquitectura cliente-servidor

En una **arquitectura cliente-servidor**, un programa —el servidor— ofrece un servicio a través de la red, y otros programas —los clientes— lo consumen enviándole solicitudes. En una aplicación web, el cliente es el navegador del usuario, que ejecuta una interfaz escrita en HTML, CSS y JavaScript, y el servidor es un proceso que corre en una computadora remota.

Para este proyecto, la división de responsabilidades entre ambos extremos es una decisión de diseño con consecuencias: dónde se ejecuta la detección de manos determina cuántos datos viajan por la red, cuánto cómputo necesita el servidor y qué tan potente debe ser el dispositivo del usuario. La sección 8.8 analiza las alternativas.

## 6.2 HTTP y el estilo REST

El protocolo **HTTP** es la base de la web: el cliente envía una solicitud —por ejemplo, `GET /health`— y el servidor devuelve una respuesta. Cada intercambio es independiente y lo inicia siempre el cliente. Fielding (2000) formalizó el estilo arquitectónico **REST**, que organiza un servicio como un conjunto de recursos identificados por direcciones y manipulados mediante los métodos estándar de HTTP, sin estado entre solicitudes. Es el estilo dominante para las interfaces de programación web (API), y el sistema lo utiliza para sus servicios de consulta (sección 11.2).

REST es adecuado para operaciones aisladas, pero poco eficiente para un flujo continuo como el video. Enviar cada cuadro como una solicitud HTTP independiente implica repetir en cada envío las cabeceras del protocolo y, si no se reutilizan las conexiones, el costo de establecerlas. Con ocho cuadros por segundo, ese costo se multiplica.

## 6.3 WebSocket

El protocolo **WebSocket** (Fette y Melnikov, 2011) establece, a partir de una solicitud HTTP inicial, un canal persistente y bidireccional entre cliente y servidor. Una vez abierto, cualquiera de los dos extremos puede enviar mensajes —de texto o binarios— en cualquier momento, con una sobrecarga mínima por mensaje. Es el protocolo estándar para aplicaciones con comunicación continua, como chats, juegos o tableros en vivo.

La @tab:protocolos compara las alternativas consideradas para transportar el video al servidor.

Tabla: Alternativas de comunicación para enviar video al servidor. {#tab:protocolos}

| Criterio | HTTP por cuadro (REST) | WebSocket | WebRTC |
|---|---|---|---|
| Conexión | Una solicitud por cuadro | Persistente | Persistente, entre pares |
| Sobrecarga por cuadro | Alta (cabeceras completas) | Baja | Baja |
| Respuesta del servidor | En la misma solicitud | Mensaje independiente en el mismo canal | Canal de datos aparte |
| Estado por cliente | Hay que reconstruirlo en cada solicitud | Natural: una conexión, una sesión | Natural |
| Complejidad | Baja | Baja | Alta (señalización, NAT, códecs) |
| Soporte en FastAPI | Nativo | Nativo | Requiere bibliotecas externas |

Se eligió WebSocket por su equilibrio entre eficiencia y simplicidad. Además, la persistencia de la conexión resuelve naturalmente un requisito del sistema: cada usuario necesita su propio estado —el historial reciente de predicciones y la ventana de cuadros acumulados—, y una conexión WebSocket delimita exactamente la vida de ese estado (sección 11.3). WebRTC, diseñado para videollamadas, ofrecería mejor manejo del video pero agrega una complejidad que no se justifica para el alcance del proyecto.

## 6.4 Acceso a la cámara desde el navegador

Los navegadores modernos permiten acceder a la cámara mediante la interfaz `MediaDevices.getUserMedia()`, que solicita el permiso del usuario y devuelve un flujo de video que puede mostrarse en un elemento `<video>` (Mozilla, s.f.). Por razones de seguridad, el acceso sólo está permitido en contextos seguros: páginas servidas por HTTPS o desde la propia máquina (`localhost`). Este requisito condiciona el despliegue, porque la aplicación publicada debe servirse con HTTPS para que la cámara funcione.

Para enviar un cuadro al servidor, el cliente lo dibuja sobre un elemento `<canvas>` y lo codifica como imagen JPEG. El canvas permite además transformar la imagen antes de enviarla —por ejemplo, reducir su resolución o espejarla horizontalmente—, operaciones que el sistema utiliza para disminuir el volumen de datos y para respetar la convención de orientación de la sección 4.4.

## 6.5 El backend en Python

### 6.5.1 FastAPI

Python es el lenguaje dominante en aprendizaje automático: PyTorch, scikit-learn (Pedregosa et al., 2011), OpenCV y MediaPipe ofrecen sus interfaces principales en Python. Escribir el backend en el mismo lenguaje permite que el servidor reutilice exactamente el mismo código de preprocesamiento que se usa para entrenar, que es un requisito central del diseño (sección 8.5).

**FastAPI** es un marco de trabajo web para Python orientado a la construcción de API (Ramírez, s.f.). Se basa en el estándar **ASGI**, que a diferencia de su antecesor WSGI soporta programación asincrónica y conexiones de larga duración como WebSocket. Entre sus características se destacan la validación automática de datos a partir de las anotaciones de tipos de Python y la generación automática de documentación de la API en formato OpenAPI, navegable desde el propio servidor. El servidor ASGI utilizado para ejecutarlo es **Uvicorn**.

### 6.5.2 Concurrencia y trabajo bloqueante

FastAPI atiende muchas conexiones simultáneas con un único hilo mediante un **bucle de eventos**: mientras una conexión espera datos de la red, el bucle atiende a otra. El modelo funciona bien mientras cada tarea ceda el control rápidamente. Sin embargo, el procesamiento de un cuadro —decodificar el JPEG, ejecutar MediaPipe y la red neuronal— es cómputo intensivo y sincrónico: si se ejecuta directamente dentro del bucle, lo bloquea, y todas las demás conexiones esperan. Con dos usuarios simultáneos, las traducciones se serializan y la latencia de cada uno se duplica.

La solución estándar es delegar el trabajo bloqueante a un conjunto de hilos auxiliares (*thread pool*), dejando el bucle de eventos libre para la comunicación. El sistema aplica esta técnica (sección 11.3), con la salvedad de que el detector de MediaPipe no admite ser usado por varios hilos a la vez, por lo que su acceso se protege con un cerrojo.

## 6.6 Contenedores y despliegue en la nube

### 6.6.1 Docker

Una aplicación de aprendizaje automático depende de muchas bibliotecas con versiones específicas, algunas con componentes nativos. Reproducir ese entorno en otra máquina es una fuente habitual de errores. **Docker** empaqueta la aplicación junto con su sistema operativo base, sus dependencias y su configuración en una **imagen**, que se ejecuta como un **contenedor** aislado de forma idéntica en cualquier máquina que tenga Docker instalado (Merkel, 2014; Docker Inc., s.f.). La imagen se describe en un archivo de texto, el `Dockerfile`, que se versiona junto con el código.

### 6.6.2 Plataformas como servicio y capa gratuita

Las **plataformas como servicio** (PaaS), como Render, Railway o Google Cloud Run, ejecutan contenedores sin que el desarrollador administre servidores. Varias ofrecen una **capa gratuita** con restricciones típicas: poca memoria y CPU compartida, suspensión del servicio tras un período de inactividad —con un arranque lento en la siguiente solicitud— y un número limitado de servicios. Desplegar dentro de esas restricciones, como exige el anteproyecto, obliga a tomar decisiones concretas: servir el cliente y la API desde un único servicio, mantener la imagen liviana y no descargar recursos al arrancar (sección 13.4).

## 6.7 Accesibilidad web

Una herramienta cuyo propósito es la inclusión debe ser, ella misma, accesible. Las **Pautas de Accesibilidad para el Contenido Web** (WCAG) del World Wide Web Consortium (2023) organizan los requisitos en cuatro principios: el contenido debe ser **perceptible**, **operable**, **comprensible** y **robusto**. Cada pauta tiene criterios verificables clasificados en tres niveles de conformidad (A, AA y AAA). Algunos criterios de nivel AA relevantes para el proyecto son:

- Contraste mínimo de 4,5:1 entre el texto y su fondo.
- Tamaño mínimo de 24 × 24 píxeles para los objetivos táctiles.
- Foco del teclado visible en todos los controles.
- Información que cambia dinámicamente anunciada a las tecnologías de asistencia, como los lectores de pantalla, mediante regiones vivas (`aria-live`).

Cabe una observación: los usuarios directos de la herramienta —una persona sorda que seña y una persona oyente que lee la traducción— no necesariamente usan lectores de pantalla, pero la accesibilidad no se agota en un perfil de usuario. Una persona con baja visión, o con una discapacidad motriz, puede ser cualquiera de los dos interlocutores. La sección 12.6 detalla cómo se aplicaron estas pautas.

