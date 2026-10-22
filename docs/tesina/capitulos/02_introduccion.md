# Introducción

## Contexto

Según la Organización Mundial de la Salud (2021), más de 1500 millones de personas en el mundo viven con algún grado de pérdida auditiva, y alrededor de 430 millones presentan una pérdida discapacitante. Para una parte de ellas, la lengua de señas es su primera lengua y su principal medio de comunicación. La Federación Mundial de Sordos (s.f.) estima que más de 70 millones de personas sordas utilizan alguna de las más de 300 lenguas de señas que existen en el mundo. En la Argentina, la comunidad sorda se comunica en **Lengua de Señas Argentina** (LSA), reconocida por la Ley 27.710 (2023) como una lengua natural y originaria, con una estructura gramatical completa y distinta de la del castellano.

A pesar de ese reconocimiento, la mayor parte de las tecnologías de comunicación fueron diseñadas para la lengua hablada o escrita. En la atención de un comercio, en un trámite administrativo, en un consultorio o en una guardia, la comunicación entre una persona sorda y una persona oyente que no conoce la LSA depende de la presencia de un intérprete, de la escritura o de gestos improvisados. Los intérpretes son escasos y no siempre están disponibles, y la escritura no resuelve el problema para quienes tienen el castellano como segunda lengua.

En la última década, los avances en visión por computadora y aprendizaje profundo hicieron posible reconocer automáticamente gestos y señas a partir de video. Existen sistemas de reconocimiento con alta precisión para lenguas como la Lengua de Señas Americana o la Lengua de Señas Alemana (Koller, 2020), pero los trabajos sobre LSA son pocos y, en su mayoría, se limitan a experimentos de laboratorio sin una interfaz utilizable por el público (ver el Capítulo 5).

## Planteamiento del problema

**Estado actual.** La comunicación entre personas sordas y oyentes que no conocen la LSA es difícil en la vida cotidiana. No existen herramientas accesibles, gratuitas y adaptadas a la LSA que permitan una traducción automática, aunque sea parcial, sin equipamiento especial.

**Estado deseado.** Disponer de una aplicación accesible desde cualquier navegador, que utilice únicamente la cámara del dispositivo, capaz de reconocer un vocabulario acotado de señas de la LSA y mostrar su traducción al castellano escrito en tiempo real, como apoyo a la comunicación en situaciones cotidianas.

## Objetivos

### Objetivo general

Desarrollar una aplicación web basada en inteligencia artificial que permita la traducción en tiempo real de señas aisladas de la Lengua de Señas Argentina a texto.

### Objetivos específicos

1. Diseñar la arquitectura del sistema y de la red neuronal de reconocimiento.
2. Entrenar una red neuronal sobre un conjunto de datos de LSA, con un preprocesamiento idéntico en el entrenamiento y en la inferencia.
3. Evaluar el modelo mediante un protocolo independiente del sujeto (que mida el desempeño sobre personas no vistas durante el entrenamiento) y analizar sus errores.
4. Desarrollar la aplicación web que capture video en tiempo real, se comunique con el modelo y presente la traducción de forma accesible.
5. Desplegar el sistema en una plataforma en la nube dentro de los límites de una capa gratuita.

El anteproyecto preveía medir la exactitud del modelo sobre datos no utilizados en el entrenamiento. El tercer objetivo amplía esa medición a un protocolo independiente del sujeto y se incorporó durante el desarrollo, al comprobar que las mediciones de exactitud obtenidas con un único reparto de los datos variaban varios puntos según qué personas quedaban en el conjunto de prueba (ver la sección 14.1). Sin una evaluación rigurosa, ninguna cifra de desempeño del modelo podía sostenerse.

## Alcances y limitaciones

### Alcances

- Aplicación web con traducción en tiempo real desde la cámara del dispositivo, utilizable desde computadoras y dispositivos móviles con navegador moderno.
- Reconocimiento de las 64 señas aisladas del conjunto LSA64 (Ronchetti et al., 2016b), que incluyen colores, alimentos, verbos y sustantivos de uso cotidiano (ver el Anexo A).
- Reconocimiento del abecedario de la LSA (alfabeto manual), con un pipeline propio de captura y entrenamiento de configuraciones manuales estáticas. [[PENDIENTE: en desarrollo. El pipeline está implementado y probado con dos letras (sección 9.8); resta capturar y evaluar el abecedario completo.]]
- Despliegue del sistema como un único servicio en contenedor.

### Limitaciones

- El sistema reconoce **señas aisladas**, no lengua de señas continua. No interpreta estructuras gramaticales, el uso del espacio ni los componentes no manuales (expresión facial, movimiento corporal), que en la LSA forman parte del significado.
- El vocabulario está restringido a las señas presentes en los datos de entrenamiento.
- Los resultados cuantitativos se midieron sobre LSA64, grabado en condiciones controladas (fondo uniforme, iluminación constante, vestimenta oscura). El desempeño con cámaras web domésticas y en entornos reales puede ser menor.
- El reconocimiento requiere buena iluminación y que la persona esté de frente a la cámara, con las manos dentro del encuadre.
- El despliegue debe mantenerse dentro de los límites de la capa gratuita de servicios en la nube, lo que restringe la cantidad de usuarios simultáneos.

## Justificación

El proyecto responde a una necesidad concreta de inclusión. Una herramienta de traducción automática, aun con un vocabulario limitado, puede facilitar interacciones breves y frecuentes —saludos, consultas, indicaciones— y reducir la dependencia de terceros en situaciones donde no hay intérprete disponible. No pretende reemplazar a los intérpretes humanos, cuya tarea abarca dimensiones lingüísticas y culturales que ningún sistema actual reproduce, sino complementar su labor.

Desde el punto de vista disciplinar, el trabajo integra áreas centrales de la Licenciatura en Sistemas de Información: análisis y diseño de sistemas, desarrollo de aplicaciones web, arquitectura cliente-servidor, aprendizaje automático, pruebas de software y despliegue en la nube. Además, el proyecto aporta un antecedente local poco frecuente: la evaluación independiente del sujeto de un modelo para LSA integrado en una aplicación funcional, con el código y los resultados documentados de forma reproducible.

## Metodología de trabajo

El desarrollo se organizó con una adaptación de Scrum para un equipo de dos personas, con iteraciones de dos semanas y un tablero Kanban (ver el Capítulo 7). Las tareas se dividieron en dos frentes de trabajo paralelos: el modelo de reconocimiento (datos, entrenamiento y evaluación) y la aplicación (backend, cliente web, pruebas y despliegue), con integraciones frecuentes a través de ramas de Git.

## Estructura del documento

El documento se divide en dos partes. La **Parte I** presenta el marco teórico: la comunidad sorda y la LSA (Capítulo 1), los fundamentos del aprendizaje automático y de su evaluación (Capítulo 2), las redes neuronales y en particular las recurrentes (Capítulo 3), la visión por computadora y la estimación de la pose de la mano (Capítulo 4), el estado del arte en reconocimiento de lengua de señas (Capítulo 5), las tecnologías para aplicaciones web en tiempo real (Capítulo 6) y la metodología de desarrollo (Capítulo 7).

La **Parte II** describe la implementación: el análisis y el diseño del sistema (Capítulo 8), la preparación de los datos y la extracción de características (Capítulo 9), el modelo y su entrenamiento (Capítulo 10), el backend de inferencia (Capítulo 11), el cliente web (Capítulo 12), el aseguramiento de la calidad y el despliegue (Capítulo 13) y los resultados de la evaluación (Capítulo 14). Cierran el trabajo las conclusiones, los anexos, la bibliografía y un glosario de términos técnicos, que aparecen en **negrita** en el lugar del texto donde se definen.
