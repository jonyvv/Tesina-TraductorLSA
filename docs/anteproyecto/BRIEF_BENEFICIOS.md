# Brief: completar los beneficios de la Justificación

Documento autosuficiente para pasar a Claude Cowork. Contiene todo el contexto
necesario; no hace falta leer el repositorio.

---

## 1. Qué hay que escribir

La sección **JUSTIFICACIÓN DEL PROYECTO** del anteproyecto tiene un apartado
"Beneficios" que está incompleto: solo desarrolla el beneficio social, en una
línea. La Guía de Anteproyecto de la facultad pide clasificar los beneficios en
**económicos, sociales, académicos y otros**.

**Entregable:** entre 2 y 4 viñetas por categoría, en el mismo formato que la
viñeta que ya existe (`Categoría: texto.`). Texto plano, listo para pegar.

---

## 2. Cómo está hoy esa sección (texto literal del documento)

> **JUSTIFICACIÓN DEL PROYECTO**
>
> El proyecto surge como respuesta a las dificultades de comunicación que
> enfrenta la comunidad sorda en distintos ámbitos de la vida cotidiana.
>
> La solución propuesta permitirá:
>
> - Facilitar la comunicación en tiempo real entre personas sordas y oyentes.
> - Reducir la dependencia de intérpretes humanos.
> - Promover la inclusión social y la igualdad de oportunidades.
>
> Beneficios:
>
> - Sociales: mejora de la calidad de vida e inclusión.

---

## 3. Qué pide exactamente la guía de la facultad

De la *Guía de Anteproyecto* de la Facultad de Informática de la Universidad
Champagnat, punto 4:

> Las justificaciones del proyecto son los argumentos que se establecen para
> demostrar el por qué vale la pena realizar la investigación del tema
> propuesto. Por norma se establece que los Trabajos de Graduación son estudios
> técnicos que tienen función social, en el sentido que buscan el beneficio de
> una colectividad.
>
> En este apartado se debe responder a las preguntas siguientes:
> - ¿Qué aportes proporcionará la investigación propuesta?
> - ¿Qué solución pretende obtener dicha investigación?
> - ¿Qué beneficios traerá al país, a la sociedad, a las Empresas e
>   Instituciones, a la Universidad, a los Profesionales, a los Estudiantes?
>
> Los beneficios pueden ser:
> - **Económicos**: incrementar la rentabilidad, mejorar la productividad, etc.
> - **Sociales**: creación de empleos, mejorar la calidad de vida, mejorar los
>   servicios de educación, salud, etc.
> - **Académicos**: facilitar la investigación científico-tecnológica, mejorar
>   métodos y condiciones de trabajo, mejorar métodos de enseñanza, etc.
> - **Otros**

---

## 4. Qué es el proyecto

**Título:** Sistema de traducción de lenguaje de señas LSA con inteligencia
artificial.

**Institución:** Universidad Champagnat, Licenciatura en Sistemas de
Información, 2026. Director de tesina: Mario Alberto Mirallas. Tutor: Marcelo
Palma. Alumnos: Jonatan Videla (red neuronal) y Leandro Provenzano (aplicación
web, backend, infraestructura y despliegue).

**Qué hace:** una aplicación web que, desde la cámara de una computadora o un
celular y sin instalar nada, reconoce en tiempo real señas de Lengua de Señas
Argentina y las muestra como texto en pantalla.

**Cómo funciona:** el navegador captura el video y envía los cuadros por
WebSocket a un backend en Python (FastAPI). El backend usa MediaPipe para
extraer 21 puntos característicos por mano, arma un vector de 138 valores por
cuadro y clasifica la secuencia con una red LSTM bidireccional entrenada con
PyTorch. La respuesta vuelve al navegador con la seña reconocida, su confianza
y los puntos para dibujar el esqueleto sobre el video.

---

## 5. Datos verificados que se pueden usar

Todos estos números están medidos y documentados. **No inventar ni redondear
otros.**

### Del sistema construido
| Dato | Valor |
|---|---|
| Vocabulario reconocido | 64 señas de palabras de LSA |
| Exactitud del modelo servido, validación independiente de sujeto (LOSO) | 81,7 % ± 7,9 % |
| Línea base no neuronal (Random Forest), mismo protocolo | 78,2 % ± 7,5 % |
| Tamaño del modelo entrenado | ~280.000 parámetros, archivo de 1,4 MB |
| Requisitos de ejecución | corre en CPU, sin GPU |
| Frontend | HTML, CSS y JavaScript nativos, sin framework |
| Tests automatizados que pasan | 132 |
| Dataset usado | LSA64 (Ronchetti et al., 2016, UNLP–LIDI): 3.200 videos, 64 señas, 10 personas, público y gratuito |
| Costo de software y servicios | cero: todo licencia libre o plan gratuito |
| Repositorio | público: https://github.com/jonyvv/Tesina-TraductorLSA |

### Del contexto (ya citados en el anteproyecto)
| Dato | Fuente |
|---|---|
| Más de 466 millones de personas con discapacidad auditiva en el mundo | Organización Mundial de la Salud (2020), *World Report on Hearing* |
| Más de 70 millones de personas usan alguna de las más de 300 lenguas de señas existentes | Federación Mundial de Sordos |
| LSA64 es el primer dataset público de Lengua de Señas Argentina | Ronchetti et al. (2016) |
| Ningún trabajo revisado aborda LSA en tiempo real mediante una aplicación web accesible | relevamiento del estado del arte del propio anteproyecto |

### Lo que el proyecto aporta y no existía
- Es, según el relevamiento hecho, el **primer sistema de reconocimiento de LSA
  en tiempo real accesible desde un navegador**. Los antecedentes sobre LSA
  (Mindlin, 2021; Marais et al., 2022; Akdag y Baykan, 2024) se quedan en el
  laboratorio: no tienen interfaz de usuario ni despliegue.
- El código es **abierto y reproducible**: el pipeline completo —extracción de
  características, entrenamiento, evaluación y servicio— está versionado, y
  los resultados se regeneran corriendo los scripts.
- Se midió con **protocolo independiente de sujeto** y contra una **línea base
  no neuronal**, dos cosas que la mayoría de los trabajos sobre LSA64 no hace
  (los valores cercanos al 99 % que se publican corresponden a un régimen donde
  la misma persona aparece en entrenamiento y prueba).

---

## 6. Límites honestos — qué NO se puede afirmar

Esto es lo más importante del brief. El texto tiene que ser defendible ante un
tribunal que conoce el trabajo.

1. **No reemplaza a un intérprete.** El sistema reconoce 64 señas aisladas. No
   hace lengua de señas continua, no maneja gramática espacial, no interpreta
   expresión facial ni estructuras complejas. La viñeta que hoy dice "reducir la
   dependencia de intérpretes humanos" es más fuerte de lo que el sistema
   sostiene, y además es una afirmación sensible para la comunidad sorda, que
   reivindica la figura del intérprete. **Sugerencia: proponer una redacción
   alternativa para esa viñeta**, del tipo "cubrir intercambios breves y
   cotidianos en los que hoy no hay ningún intérprete disponible".
2. **No está validado con usuarios reales.** Todavía no se hicieron pruebas con
   personas usuarias de LSA. Los resultados salen del dataset LSA64, grabado en
   condiciones controladas de laboratorio.
3. **No hay medición en condiciones reales.** El 81,7 % se midió sobre
   secuencias completas de video; en vivo el sistema recibe ventanas
   deslizantes y ese número no predice directamente su desempeño.
4. **No está desplegado todavía.** El esquema previsto es frontend en Vercel y
   backend corriendo en la computadora del equipo, expuesto con ngrok.
5. **No hay abecedario.** Solo palabras.

Por lo tanto: los beneficios se redactan como **potencial del enfoque y aporte
del trabajo**, no como resultados ya conseguidos. Nada de "mejora la
empleabilidad" o "reduce costos de interpretación en X %": no hay dato que lo
sostenga.

---

## 7. Reglas de redacción

- **Español rioplatense, registro impersonal**: "se desarrolló", "el sistema
  permite". Sin voseo en el cuerpo del documento.
- **Números con coma decimal y espacio antes del signo de porcentaje**:
  "81,7 %", no "81.7%".
- **Citas en APA 7**: (Apellido, año). Solo se puede citar lo que ya está en la
  bibliografía del anteproyecto: Mindlin (2021); Ronchetti et al. (2016);
  Goodfellow et al. (2016); Géron (2022); Akdag y Baykan (2024); Marais et al.
  (2022); Koller (2020); Organización Mundial de la Salud (2020); Federación
  Mundial de Sordos. Si hace falta una fuente nueva, hay que avisarlo aparte
  para agregarla a la bibliografía.
- **Nada inventado.** Si un beneficio necesita un dato que no está en la
  sección 5, no se escribe: se deja anotado como pregunta al final.
- Cada viñeta, de una a tres líneas. Es un anteproyecto, no la tesina.

---

## 8. Formato de la respuesta

Texto plano, así:

```
Beneficios:

Económicos: <viñeta>
Económicos: <viñeta>

Sociales: mejora de la calidad de vida e inclusión.
Sociales: <viñeta nueva>
...

Académicos: <viñeta>
...

Otros: <viñeta>
...
```

Y al final, por separado:

- **Redacción alternativa propuesta** para la viñeta "Reducir la dependencia de
  intérpretes humanos" (ver sección 6, punto 1).
- **Dudas o datos faltantes**: lo que haga falta confirmar con el tutor antes
  de dar el texto por cerrado.
