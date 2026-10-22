# Capítulo 1 — La comunidad sorda y la Lengua de Señas Argentina

Antes de abordar los aspectos técnicos, conviene precisar qué se intenta reconocer. Un sistema de traducción de lengua de señas no reconoce "gestos" en general, sino unidades de una lengua con su propia fonología, gramática y comunidad de hablantes. Las decisiones de diseño de los capítulos siguientes —qué información extraer de la imagen, qué tipo de modelo usar y cómo evaluarlo— dependen de entender cómo está estructurada una seña.

## 1.1 Discapacidad auditiva y comunidad sorda

La pérdida auditiva abarca un espectro amplio, desde dificultades leves hasta la sordera profunda, y puede ser congénita o adquirida. La Organización Mundial de la Salud (2021) estima que más de 1500 millones de personas viven con algún grado de pérdida auditiva, de las cuales unos 430 millones presentan una pérdida discapacitante.

Desde el modelo social de la discapacidad, adoptado por la Convención sobre los Derechos de las Personas con Discapacidad (Naciones Unidas, 2006) e incorporado al derecho argentino por la Ley 26.378 (2008), la discapacidad no reside en la persona sino en la interacción entre sus características y las barreras del entorno. Para una persona sorda, la barrera no es la sordera en sí, sino un entorno que presupone que todos oyen y hablan.

Dentro de ese colectivo, la **comunidad sorda** está formada por las personas que comparten la lengua de señas como lengua principal, junto con una identidad y una cultura propias. Esta distinción es relevante para el proyecto: los destinatarios de la herramienta no son todas las personas con pérdida auditiva —muchas de las cuales se comunican oralmente o con audífonos e implantes— sino quienes se comunican en LSA.

### 1.1.1 Terminología

El término "sordomudo", todavía frecuente en el lenguaje cotidiano y presente en la versión inicial del anteproyecto de esta tesina, es considerado inadecuado por la propia comunidad sorda[^mudo]. La forma aceptada es **persona sorda**, y en este documento se utiliza exclusivamente esa expresión. En el mismo sentido, se habla de "lengua" de señas y no de "lenguaje", porque se trata de lenguas naturales completas y no de un sistema de comunicación artificial.

[^mudo]: La mayoría de las personas sordas tienen el aparato fonador intacto y pueden vocalizar; lo que no adquieren de forma natural es la lengua oral, por no acceder a ella auditivamente. "Mudo" describe incorrectamente esa situación y arrastra una connotación de incapacidad que la comunidad rechaza.

## 1.2 Las lenguas de señas como lenguas naturales

Durante mucho tiempo se consideró que las lenguas de señas eran pantomima o una versión gestual de la lengua oral del entorno. El trabajo de Stokoe (1960) sobre la Lengua de Señas Americana fue el primero en demostrar que las señas no son unidades indivisibles, sino que se componen de un conjunto finito de elementos que se combinan de manera sistemática, del mismo modo en que las palabras de una lengua oral se componen de fonemas. Desde entonces, la lingüística reconoce a las lenguas de señas como lenguas naturales, con todos los niveles de análisis de cualquier lengua: fonología, morfología, sintaxis y pragmática.

Dos propiedades tienen consecuencias directas para el reconocimiento automático:

- **No son universales.** Cada comunidad desarrolló su propia lengua de señas, y estas no se corresponden con las lenguas orales del entorno: la Lengua de Señas Americana y la Lengua de Señas Británica son lenguas distintas, aunque en ambos países se hable inglés. Un modelo entrenado para una lengua de señas no sirve para otra.
- **Son lenguas visoespaciales.** La información se transmite simultáneamente por varios canales: las dos manos, su ubicación respecto del cuerpo, el movimiento, la orientación, la expresión facial y la postura. Mientras una lengua oral es esencialmente secuencial, una seña combina varios elementos en el mismo instante.

## 1.3 La Lengua de Señas Argentina

La LSA es la lengua de la comunidad sorda argentina. Massone y Machado (1994) realizaron la primera descripción lingüística sistemática de la LSA junto con un vocabulario bilingüe, y establecieron que su estructura es independiente de la del castellano. La Ley 27.710 (2023) reconoció a la LSA como "lengua natural y originaria" que conforma el patrimonio lingüístico y cultural de las personas sordas en todo el territorio nacional, y facultó al Estado a promover su acceso y uso en los distintos ámbitos.

### 1.3.1 Parámetros formales de una seña

Siguiendo el análisis iniciado por Stokoe (1960) y aplicado a la LSA por Massone y Machado (1994), una seña puede describirse a partir de un conjunto de parámetros formales. La @tab:parametros los resume e indica cuáles de ellos son observables por el sistema desarrollado en esta tesina, que trabaja sobre los puntos de referencia de las manos (ver el Capítulo 9).

Tabla: Parámetros formales de una seña y su tratamiento en el sistema. {#tab:parametros}

| Parámetro | Descripción | ¿Lo captura el sistema? |
|---|---|---|
| Configuración | Forma que adopta la mano: dedos extendidos, flexionados, en contacto | Sí: coordenadas normalizadas de los 21 puntos y ángulos de flexión de cada dedo |
| Ubicación | Lugar del espacio o del cuerpo donde se realiza la seña | Parcialmente: la versión v1 de las características la descarta al centrar en la muñeca; la versión experimental v2 la incorpora (sección 9.6) |
| Movimiento | Desplazamiento o cambio de la mano durante la seña | Sí, de forma implícita: el modelo recibe la secuencia completa de cuadros |
| Orientación | Hacia dónde apunta la palma | Sí, de forma indirecta, a través de la coordenada de profundidad y la geometría de los puntos |
| Rasgos no manuales | Expresión facial, movimiento de la cabeza y del torso | No: el sistema sólo observa las manos |

La segunda fila de la tabla anticipa uno de los hallazgos del Capítulo 14: buena parte de los errores del modelo se concentran en pares de señas que comparten configuración y difieren en la ubicación, precisamente el parámetro que la representación principal del sistema no conserva.

### 1.3.2 Señas de una y dos manos

En la LSA, como en otras lenguas de señas, existen señas realizadas con una sola mano —la **mano dominante**— y señas bimanuales, en las que ambas manos se mueven de forma simétrica o una actúa como base de la otra. En el conjunto LSA64, 42 de las 64 señas se realizan con una mano y 22 con las dos (Ronchetti et al., 2016b). Un sistema que sólo considere "la primera mano detectada" en cada cuadro no puede distinguir correctamente las señas bimanuales; por eso el diseño de las características reserva una posición fija para cada mano (sección 9.4).

### 1.3.3 El alfabeto manual

Además del léxico de señas, la LSA cuenta con un **alfabeto manual** o dactilológico, que asigna una configuración de la mano a cada letra del alfabeto castellano. Se usa para deletrear nombres propios, palabras sin seña establecida o términos técnicos. Desde el punto de vista del reconocimiento, la mayoría de las letras son configuraciones estáticas —basta con un cuadro para identificarlas—, aunque algunas incorporan movimiento. Esta diferencia justifica tratar por separado el reconocimiento de letras (clasificación de un vector) y el de señas léxicas (clasificación de una secuencia), como se discute en el Capítulo 8.

## 1.4 Barreras de comunicación y herramientas existentes

En la práctica, la comunicación entre una persona sorda y una oyente que no sabe LSA se resuelve con alguna de las siguientes alternativas, ninguna de ellas plenamente satisfactoria:

- **Intérpretes de LSA.** Es la solución de mayor calidad, pero los intérpretes son pocos, se concentran en las grandes ciudades y su presencia debe coordinarse con anticipación. No están disponibles para interacciones espontáneas y breves.
- **Videointerpretación remota.** Algunos organismos ofrecen servicios de interpretación a distancia por videollamada. Reducen el problema de la presencia física, pero siguen dependiendo de la disponibilidad de un intérprete.
- **Escritura.** Escribir en papel o en el teléfono es la alternativa más común, pero el castellano escrito es para muchas personas sordas una segunda lengua, aprendida en condiciones desfavorables, y la comunicación resulta lenta y parcial.
- **Aplicaciones de aprendizaje y diccionarios.** Existen diccionarios y aplicaciones para aprender señas, útiles para las personas oyentes interesadas, pero no traducen en tiempo real.

La herramienta propuesta en esta tesina se ubica en un espacio distinto: un apoyo automático, inmediato y sin costo para interacciones acotadas, que no requiere coordinar con terceros ni equipamiento especial. Su principal limitación —el vocabulario reducido y el reconocimiento de señas aisladas— está explicitada desde los alcances del proyecto.

## 1.5 Consideraciones éticas

Trabajar con una lengua minoritaria y con imágenes de personas implica responsabilidades que se tuvieron en cuenta durante el desarrollo:

- **Representación honesta de las capacidades.** Un sistema que "traduce" sólo 64 señas aisladas no traduce la LSA. Presentarlo como un traductor general generaría expectativas falsas y podría perjudicar a quienes confíen en él en una situación importante. Por eso la interfaz muestra la confianza de cada predicción y el documento declara explícitamente las limitaciones.
- **Datos biométricos.** El video de una persona y la geometría de sus manos son datos personales. El sistema no almacena el video recibido: cada cuadro se procesa en memoria y se descarta. Si se capturan datos propios para entrenar (sección 9.8), es necesario el consentimiento informado de los participantes.
- **Participación de la comunidad.** La evaluación de una herramienta destinada a la comunidad sorda debe incluir a personas sordas. [[PENDIENTE: está prevista una prueba con personas usuarias de LSA, con consentimiento informado; su protocolo y sus resultados se describirán en la sección 14.8.2.]]

