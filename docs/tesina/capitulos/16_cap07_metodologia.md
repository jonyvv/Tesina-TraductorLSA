# Capítulo 7 — Metodología de desarrollo

Este capítulo describe cómo se organizó el trabajo: el marco de gestión adoptado, la división de tareas, el manejo del código fuente y las prácticas que se incorporaron para que los resultados experimentales fueran confiables y reproducibles.

## 7.1 Enfoque ágil

El proyecto combinaba dos frentes con incertidumbre alta: un modelo de aprendizaje automático cuyo desempeño no podía conocerse de antemano, y una aplicación cuyos requisitos de latencia y usabilidad sólo podían validarse con el sistema funcionando. En ese contexto, planificar todo el desarrollo al comienzo era poco realista. Se adoptó por eso un enfoque ágil, basado en los valores del Manifiesto Ágil (Beck et al., 2001): priorizar el software funcionando, responder al cambio y trabajar en ciclos cortos con entregas incrementales.

### 7.1.1 Scrum adaptado

El anteproyecto tomó como marco de referencia **Scrum** (Schwaber y Sutherland, 2020), que organiza el trabajo en iteraciones de duración fija llamadas **sprints**, cada una con un objetivo y un incremento verificable, y define eventos de planificación, revisión y retrospectiva. Preveía sprints de dos semanas. En la práctica, con dos personas que trabajaban en paralelo sobre frentes distintos y con tiempos condicionados por el cursado, el ritmo no fue el de un calendario fijo. Lo que se mantuvo de Scrum fue:

- **Iteraciones cerradas por una integración** de los dos frentes, que el historial del repositorio registra el 14 de agosto, el 1 de septiembre, el 18 de octubre y el 17 de noviembre, con una revisión al cierre para decidir qué se mantenía y qué se descartaba.
- **Sin roles formales.** Los tres roles de la guía (dueño del producto, desarrolladores y *Scrum Master*) no tienen sentido separados en un equipo de dos. Las decisiones de alcance se tomaron en conjunto, y los profesores coordinador y tutor actuaron como referentes externos en las revisiones de avance.

### 7.1.2 Tablero Kanban

Las tareas se gestionaron en un tablero **Kanban** en Trello, con tres columnas (por hacer, en progreso y terminado), que hace visible el estado del trabajo y limita la cantidad de tareas abiertas al mismo tiempo. [[PENDIENTE: incluir una captura del tablero de Trello.]]

## 7.2 División del trabajo

El trabajo se organizó en dos frentes, cada uno con un responsable principal:

- **Modelo de reconocimiento** (Jonatan Videla): preparación de los datos de LSA64, extracción de características, entrenamiento de la red, validación leave-one-subject-out y experimentos de mejora.
- **Aplicación** (Leandro Provenzano): arquitectura del sistema, backend de inferencia, cliente web, accesibilidad, pruebas automatizadas, contenedor de despliegue y documentación.

Esta división se definió al comenzar el desarrollo, es la que figura en la sección de recursos humanos del anteproyecto y se mantuvo sin cambios durante todo el proyecto.

Ambos frentes compartían un punto de contacto crítico: el módulo de extracción de características, que debe producir exactamente los mismos vectores durante el entrenamiento y durante el uso. Ese módulo se diseñó desde el comienzo como código compartido entre los dos frentes (sección 8.5), y su contrato se documentó en el propio código.

## 7.3 Control de versiones

El código se gestionó con **Git** y se alojó en un repositorio de GitHub (Chacon y Straub, 2014). Se trabajó con ramas por línea de trabajo, que se integraban mediante fusiones (*merge*) explícitas:

- `main`: versión estable del proyecto.
- `test`: rama de integración, donde se combinan los avances de los dos frentes y se verifica que el sistema funcione de punta a punta.
- `lean`: desarrollo de la aplicación (backend y cliente).
- `Modulo-LSA64`: desarrollo del pipeline de datos y del modelo sobre LSA64.

A partir de agosto de 2026, los mensajes de los *commits* siguieron una convención con prefijos que indican el tipo de cambio (`feat` para funcionalidad nueva, `fix` para correcciones, `docs` para documentación, `chore` para mantenimiento) y se redactaron explicando qué se cambió y por qué; los primeros registros del repositorio son anteriores a esa convención. El historial registra el trabajo en pocos *commits* grandes, cada uno al cerrar una etapa, y no cambio por cambio. Aun así, resultó valioso como registro de las decisiones: varios apartados de este documento, en particular los de la sección 9.7 y el Capítulo 14, se reconstruyeron a partir de él.

## 7.4 Pruebas automatizadas

Desde la reorganización del backend y del pipeline de datos, en agosto de 2026, se escribieron **pruebas automatizadas** con pytest para los componentes con lógica no trivial: la normalización de las características, la carga y validación de modelos, el suavizado temporal y el pipeline de datos de LSA64. Una práctica que resultó especialmente útil fue escribir una **prueba de regresión** por cada error real encontrado: una prueba que falla con el código defectuoso y pasa con el corregido, documentada con la etiqueta `REGRESIÓN` y una explicación del problema. Así, cada error corregido queda protegido contra su reaparición. El Capítulo 13 describe la suite completa.

## 7.5 Rigor experimental

El desarrollo del modelo se trató como un proceso experimental, con prácticas tomadas de la investigación científica:

- **Un protocolo de evaluación fijo.** Todos los resultados que se informan provienen del mismo protocolo (LOSO) y del mismo conjunto de características, de modo que son comparables entre sí.
- **Reproducibilidad.** Todas las fuentes de aleatoriedad —el reparto de los datos, la inicialización de los pesos, el dropout y el orden de los lotes— se controlan con una semilla. Dos ejecuciones con la misma configuración en el mismo equipo producen el mismo resultado, lo que permite atribuir cualquier diferencia a los cambios introducidos y no al azar. Entre equipos distintos la reproducción no es exacta, por lo que toda comparación se hace entre ejecuciones del mismo equipo (sección 10.4).
- **Línea base inmutable.** Cada nueva opción de entrenamiento se incorporó desactivada por defecto, de modo que una ejecución sin opciones reprodujera exactamente el resultado de referencia.
- **Hipótesis explícitas.** Cada experimento se planteó como una hipótesis con un resultado esperado, y los resultados negativos se documentaron con el mismo detalle que los positivos (sección 14.6).
- **Registro de resultados.** Cada ejecución guarda un reporte en formato JSON con sus hiperparámetros, su reparto de datos y sus métricas, versionado en el repositorio.

## 7.6 Herramientas

La @tab:herramientas resume las herramientas utilizadas.

Tabla: Herramientas utilizadas en el desarrollo. {#tab:herramientas}

| Área | Herramienta | Uso |
|---|---|---|
| Lenguajes | Python 3.12, JavaScript, HTML5, CSS3 | Backend y modelo; cliente web |
| Aprendizaje automático | PyTorch, scikit-learn, NumPy | Red BiLSTM, bosque aleatorio y MLP, cálculo numérico |
| Visión por computadora | MediaPipe (API Tasks), OpenCV | Detección de landmarks, decodificación de video e imágenes |
| Backend | FastAPI, Uvicorn | API REST, WebSocket, servidor ASGI |
| Pruebas | pytest | Pruebas unitarias, de regresión y de integración |
| Despliegue | Docker | Imagen del servicio |
| Gestión | Git, GitHub, Trello | Control de versiones, trabajo colaborativo y tablero Kanban |
| Documentación | Markdown, matplotlib | Documentación técnica y figuras de resultados |
| Edición | Visual Studio Code | Editor principal |

Durante el desarrollo y la redacción de este documento se utilizaron asistentes de programación basados en inteligencia artificial (Claude, de Anthropic) como apoyo en la escritura y la revisión de código, en la búsqueda de errores y en la redacción y corrección de textos. Todo resultado obtenido con su ayuda fue revisado y probado por los autores, que son responsables del contenido del trabajo. Las contribuciones de estos asistentes quedan registradas en el historial del repositorio.

## 7.7 Cronograma

El Anexo D presenta el cronograma del proyecto en forma de diagrama de Gantt, con las etapas cumplidas y las restantes hasta la entrega.

