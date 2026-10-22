# Conclusiones

## Conclusiones generales

Esta tesina se propuso desarrollar una aplicación web basada en inteligencia artificial que traduzca a texto, en tiempo real, señas aisladas de la Lengua de Señas Argentina. El resultado es un sistema funcional de tres capas: un cliente web que captura el video de la cámara y lo envía por WebSocket; un backend en Python que detecta las manos con MediaPipe, construye un vector de 138 características por cuadro y clasifica la secuencia con una red BiLSTM; y un pipeline de entrenamiento y evaluación sobre el conjunto de datos LSA64, que comparte con el backend el código de extracción de características.

Respecto de los objetivos específicos:

1. **Diseño de la arquitectura.** Se diseñó e implementó una arquitectura cliente-backend-modelo con un módulo de código compartido, que garantiza por construcción que el preprocesamiento sea el mismo en el entrenamiento y en la inferencia. El diseño separa el estado de la aplicación del estado de cada conexión, y admite modelos de distinto formato mediante adaptadores (Capítulos 8 y 11).
2. **Entrenamiento de la red.** Se entrenó una BiLSTM de 290 880 parámetros sobre LSA64, con un proceso determinista y repartos de datos siempre por persona (Capítulos 9 y 10).
3. **Evaluación independiente del sujeto.** El modelo alcanzó una exactitud de 79,8 % ± 8,9 % en una validación leave-one-subject-out de 10 particiones, sobre 64 clases. Se analizaron los errores por persona y por clase, y se evaluaron hipótesis de mejora (Capítulo 14). El anteproyecto sólo preveía medir la exactitud sobre datos no utilizados en el entrenamiento; el protocolo independiente del sujeto se incorporó al comprobar que un reparto único producía cifras que variaban hasta 7,5 puntos entre ejecuciones casi idénticas.
4. **Aplicación web.** Se desarrolló un cliente accesible, que cumple los criterios de nivel AA de WCAG 2.2 aplicables, con dibujo del esqueleto de las manos, métricas de latencia y reconexión automática (Capítulo 12).
5. **Despliegue.** El sistema está empaquetado en una imagen Docker pensada para la capa gratuita de una plataforma en la nube. [[PENDIENTE: completar con el resultado del despliegue, todavía no realizado.]]

## Inconvenientes resueltos y camino recorrido

El proyecto partió de un prototipo de escritorio de terceros y de un diagrama de arquitectura. El camino hasta el sistema actual estuvo marcado, más que por la construcción de componentes nuevos, por la detección y corrección de errores que no producían ningún mensaje de error y que, sin embargo, invalidaban los resultados:

- La **extracción de características duplicada** en el prototipo, que se resolvió con el módulo compartido y la versión del esquema verificada al cargar cada modelo.
- La **orientación inconsistente de la imagen** entre captura, entrenamiento e inferencia, que intercambiaba las manos en el vector sin que ningún control lo advirtiera, y que se resolvió con una convención explícita y verificable.
- El **etiquetado incorrecto** de LSA64, que entrenaba con 50 clases artificiales en lugar de las 64 señas reales, y la consiguiente **fuga de datos** entre entrenamiento y prueba.
- El **entrenamiento no reproducible**, que impedía distinguir una mejora real del ruido.
- El **estado compartido entre usuarios** en el backend, invisible en el diagrama de clases.
- Un **flag de presentación que alteraba el resultado** del entrenamiento, al cambiar el orden de las clases.

El aprendizaje común a todos estos casos es que, en un sistema de aprendizaje automático, los errores más graves no se manifiestan como fallas, sino como números verosímiles. Un modelo entrenado con etiquetas incorrectas o con fuga de datos produce una exactitud que parece razonable. La defensa contra este tipo de error no es la revisión del código en abstracto, sino un conjunto de prácticas: verificaciones explícitas de los supuestos (que no haya fuga, que el esquema coincida, que el caché sea compatible), pruebas de regresión para cada error encontrado, reproducibilidad estricta y un protocolo de evaluación fijo y exigente.

## Contribuciones y aportes

**A la comunidad sorda y a la sociedad.** El trabajo ofrece una herramienta de código abierto que demuestra la viabilidad de reconocer señas de la LSA en tiempo real con una cámara común y un navegador, y que puede servir como base para herramientas de apoyo a la comunicación en ámbitos concretos —atención al público, salud, educación—, con un vocabulario adaptado a cada uno.

**A otros estudiantes e investigadores.**

- Un sistema completo, documentado y con pruebas, que integra un modelo de reconocimiento en una aplicación web, algo poco frecuente en los antecedentes sobre LSA.
- Una evaluación independiente del sujeto sobre LSA64 con un modelo liviano basado en landmarks, con resultados por persona y por clase, y las predicciones disponibles para análisis posteriores.
- La documentación de un caso concreto, con evidencia, de cómo el protocolo de evaluación cambia las conclusiones, incluida la revisión del protocolo del principal antecedente citado.
- Las glosas de LSA64 en castellano, con la corrección de la seña 008 y criterios explícitos para las traducciones ambiguas.
- Un registro de errores metodológicos frecuentes en proyectos de reconocimiento de señas y de cómo detectarlos, útil para quien encare un trabajo similar.

## Futuro del proyecto

Los siguientes puntos quedan abiertos para continuar el trabajo, ordenados aproximadamente según su impacto esperado:

1. **Cerrar la brecha con el uso real.** Segmentar las señas según la presencia de las manos en lugar de usar una ventana fija, alinear la frecuencia de muestreo entre entrenamiento y uso, y medir el desempeño con personas reales frente a la cámara (sección 14.8).
2. **Mejorar la detección de las manos.** Evaluar el modo de seguimiento de video de MediaPipe, otros umbrales de confianza y la recuperación de los videos descartados, que es el factor más asociado con los errores (sección 14.5).
3. **Representar la ubicación respecto del cuerpo.** Incorporar puntos del cuerpo con el modelo de pose de MediaPipe para expresar la posición de las manos de forma independiente del encuadre, y evaluar si reduce las confusiones entre señas que difieren en la ubicación.
4. **Ampliar el vocabulario y los datos.** Capturar un conjunto de datos propio con más personas, en condiciones reales y con consentimiento informado, orientado a un ámbito de uso concreto; incorporar el alfabeto manual completo.
5. **Otras arquitecturas.** Probar modelos basados en atención (Transformers) y redes convolucionales sobre grafos del esqueleto, que son el estado del arte en reconocimiento basado en landmarks.
6. **Detección en el cliente.** Ejecutar MediaPipe en el navegador para reducir el cómputo del servidor y el ancho de banda, con una verificación automática de que produce los mismos vectores que la implementación en Python (sección 8.8.1).
7. **Reconocimiento continuo y componentes no manuales.** A más largo plazo, avanzar hacia el reconocimiento de frases y la incorporación de la expresión facial, que en la LSA forma parte del significado.
8. **Participación de la comunidad sorda.** Cualquier evolución del sistema debería diseñarse y evaluarse junto con personas sordas usuarias de la LSA, que son quienes pueden determinar si la herramienta resulta útil y en qué situaciones.
