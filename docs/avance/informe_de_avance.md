# Informe de avance de la tesina

**Tesina:** Sistema de traducción de Lengua de Señas Argentina a texto en tiempo real mediante inteligencia artificial.

**Alumnos:** Leandro Provenzano y Jonatan Videla. **Profesor de tesina:** Mario Alberto Mirallas. **Profesor tutor:** Marcelo Palma.

**Fecha:** 21 de octubre de 2026.

## 1. Documentos que se entregan

Tabla: Documentos de esta revisión.

| Documento | Archivo | Contenido |
|---|---|---|
| Anteproyecto, versión corregida | AnteproyectoProvenzanoVidela20261021v02.pdf | 17 páginas. Corrige la forma, el cumplimiento de la Guía de Anteproyecto y la redacción del anteproyecto aprobado, sin cambiar el plan. |
| Tesina, borrador v02 | ProvenzanoVidela20261021v02.pdf | 115 páginas (el rango pedido es de 90 a 120). Documento completo; los apartados que faltan están resaltados en amarillo. |

## 2. Estado del desarrollo

**Implementado y probado:**

- Backend en Python con FastAPI, que recibe los cuadros de la cámara por WebSocket, detecta las manos con MediaPipe y clasifica la seña con la red neuronal.
- Cliente web en HTML, CSS y JavaScript, que cumple los criterios aplicables de nivel AA de las pautas de accesibilidad WCAG 2.2.
- Red neuronal recurrente bidireccional (BiLSTM) de 290 880 parámetros, entrenada sobre las 64 señas del conjunto de datos LSA64.
- Pipeline de captura y entrenamiento de configuraciones manuales estáticas, probado con dos letras del abecedario.
- Imagen Docker del servicio y 116 pruebas automatizadas.

**Resultado principal:** 79,8 % ± 8,9 % de exactitud sobre 64 señas, medida con validación *leave-one-subject-out* de 10 particiones, es decir, siempre sobre personas que el modelo no vio durante el entrenamiento. El 99,4 % que reporta Mindlin (2021) sobre el mismo conjunto de datos se obtuvo con las mismas personas en el entrenamiento y en la prueba, por lo que no es comparable (sección 5.3.1 de la tesina).

**Pendiente:** la captura y evaluación del abecedario completo, el despliegue en una plataforma en la nube, la línea base con un bosque aleatorio, la medición de la latencia y de la diferencia entre el desempeño medido y el uso en tiempo real, y una prueba con personas usuarias de LSA. Las fechas tentativas, entre noviembre y diciembre de 2026, figuran en el Anexo D de la tesina.

## 3. Estado del documento de tesina

- Sigue la estructura del "Documento general de redacción" y el formato de la facultad: A4, márgenes de 4, 2, 3 y 3 cm, Times New Roman 11, cabecera por capítulo y pie con "Página X de Y".
- Contiene la Parte I, de marco teórico (7 capítulos), la Parte II, de implementación (7 capítulos), las conclusiones, 4 anexos, la bibliografía en formato APA y un glosario, con 19 figuras y 21 tablas.
- Se aplicaron las correcciones que recibió la tesina aprobada en 2025: toda la bibliografía está citada en el texto, los términos del glosario están en negrita donde se definen, y se usan referencias cruzadas y notas al pie.
- Quedan 20 apartados marcados como pendientes. Quince dependen del desarrollo que resta. Los otros cinco corresponden a los autores: los agradecimientos, la captura del tablero de tareas, una captura de la aplicación con la cámara activa, la prueba en un teléfono y la carpeta compartida "Fuentes del proyecto".

## 4. Correcciones del anteproyecto

El anteproyecto aprobado se corrigió sin modificar su plan (alcance, tecnologías, roles y cronograma):

- Se numeraron las secciones del 1 al 12, como enumera la guía, y se agregaron los recursos materiales y financieros (10.3 y 10.4), con el monto total del proyecto, y las horas-hombre de los recursos humanos.
- Los objetivos específicos se redactaron de forma medible, y se incorporó entre ellos el despliegue, que el aprobado ya preveía en los alcances, los resultados esperados y el cronograma.
- La justificación incluye los beneficios económicos y académicos, además de los sociales; la metodología se describe como etapas secuenciales, y el planteamiento del problema sigue el esquema de la caja negra que propone la guía.
- El diagrama de Gantt se redibujó con las mismas tareas y semanas, porque en el aprobado los nombres de las tareas no se leían, y se agregaron los hitos de evaluación académica.
- Los anexos, que antes sólo se enumeraban, se incluyeron y se referencian desde el cuerpo.
- Se corrigieron datos (el informe de la OMS es de 2021 y da 430 millones de personas con pérdida auditiva discapacitante) y terminología: "lengua" de señas en lugar de "lenguaje", y "personas sordas" en lugar de "sordomudas".

## 5. Consultas para el profesor

1. **Interlineado.** El formato de la facultad indica interlineado doble, pero las tesinas aprobadas recientes usan 1,5. Con doble, el documento llega a 144 páginas y supera el máximo de 120. ¿Se acepta el interlineado de 1,5?
2. **Cambios respecto del anteproyecto.** El cliente se desarrolló sin React, con HTML, CSS y JavaScript; el modelo se entrenó con PyTorch, en scripts de Python y en forma local, en lugar de cuadernos de Jupyter y Google Colab; y la división de roles resultó inversa a la prevista. Los cambios se justifican en las secciones 7.2 y 8.8 de la tesina. ¿Alcanza con justificarlos allí?
3. **Anteproyecto corregido.** ¿Corresponde presentar la versión corregida en reemplazo de la aprobada, o como un documento adicional?
4. **Profesor de Metodología.** El formato de la carátula pide el nombre del profesor o la profesora de Metodología. ¿Quién corresponde?
5. **Uso de asistentes de inteligencia artificial.** Su uso en el desarrollo y en la redacción se declara en la sección 7.6 de la tesina. ¿La facultad tiene una pauta sobre cómo hacerlo?
