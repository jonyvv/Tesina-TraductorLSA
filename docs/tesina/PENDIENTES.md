# Qué falta completar en la tesina

_Al 24/11/2026, sobre `salida/Tesina-TraductorLSA.docx` (120 páginas, 34.679 palabras)._

Hay **16 huecos** en los capítulos y **3 decisiones** abiertas. Los huecos aparecen **resaltados en
amarillo** en el Word, así que se los puede recorrer abriendo el archivo y buscando el resaltado;
`python docs/tesina/build/verificar.py` lista los 16 en orden.

Este archivo los agrupa por la acción que los cierra. Las decisiones van primero porque cambian
qué hay que escribir.

---

## 1. Decisiones

| # | Decisión | Estado | Por qué importa |
|---|---|---|---|
| **1** | **Extensión: el documento quedó en 120 páginas**, el máximo del rango 90–120 | Resuelta, pero justa | Se llegó a 120 el 24/11 sacando el Abstract y recortando frases redundantes en los capítulos 4, 5, 6, 10, 12, 14 y en las conclusiones, que derramaban 2 a 7 renglones a una página nueva. **Cualquier texto que se agregue puede volver a pasarse**: después de cada cambio, mirar el número que imprime `postproceso.ps1`. Sigue sin respuesta la consulta del interlineado (con doble, el documento da unas 144 páginas). |
| **2** | **El título difiere del anteproyecto** | Abierta | Anteproyecto: "Sistema de traducción de lenguaje de señas LSA con inteligencia artificial". Tesina: "Sistema de traducción de Lengua de Señas Argentina a texto en tiempo real mediante inteligencia artificial". Hay que unificarlo con el tutor. |
| **3** | **¿v2 pasa a ser el resultado principal?** | Abierta | La tesina ya informa v2 en la sección 14.6.4 como resultado medido: 83,4 % ± 7,2 % contra 81,0 % ± 6,1 % de v1 **en el mismo equipo con GPU** (`comparacion_v2_v1.json`), y explica por qué el principal sigue siendo v1 (el backend no genera v2 y la ubicación depende del encuadre). Adoptarla implica cambiar el preprocesamiento del backend, reentrenar el `.pt` y reescribir 14.2, 14.10, el resumen y las conclusiones. |
| **4** | **Cliente sin React** | Resuelta en el texto, sin validar | 8.8.2 justifica la decisión y el anteproyecto v3 la respalda en su B.6. Falta el visto del tutor. |

---

## 2. Trabajo ordenado, con lo que cierra cada cosa

### A. Desplegar el sistema — **Leandro**

| Hueco | Qué dato hay que traer |
|---|---|
| **13.4 Despliegue** | La dirección pública de la interfaz, los pasos seguidos y la latencia medida contra el servicio expuesto por el túnel. |
| **Conclusiones, punto 5** | Una línea con el resultado del despliegue. |

La imagen ya está construida y probada: `docker run --rm -p 8000:8000 traductor-lsa` levanta el
servicio que va detrás del túnel. La dirección configurable del backend (`frontend/config.js`) y la
validación de orígenes (`LSA_CORS_ORIGINS`) ya están hechas.

Tres cosas de ngrok a tener en cuenta (documentación de la capa gratuita, consultada el 17/11):

- **El dominio es fijo**, así que `config.js` se configura una sola vez.
- **Página de aviso para navegadores.** Si los `fetch` a `/health` y `/model/info` reciben ese HTML
  en lugar del JSON, hay que mandarles el encabezado `ngrok-skip-browser-warning`.
- **Cupos.** 1 GB de transferencia y 20 000 pedidos HTTP por mes.

### B. Tiempo real — **Leandro**, después de A

| Hueco | Qué dato hay que traer |
|---|---|
| **14.8.1 Latencia** | Tabla con latencia total, cómputo del servidor y tiempo de red, en tres escenarios (local, red local, desplegado) y a 8 y 15 fps. Primera medición del 17/11, local sin Docker, a 8 fps: **23 ms total, 21 ms servidor, 2 ms red**. |
| **14.8.2 Brecha con el uso en tiempo real** | Implementar las dos acciones previstas: (a) alinear la frecuencia de muestreo entre entrenamiento y uso y (b) segmentar según la presencia de las manos en lugar de la ventana fija. |

### C. Capturas — **Leandro**

| Hueco | Qué falta |
|---|---|
| **12.1 Estructura** | Captura con la cámara activa, el esqueleto dibujado y una seña reconocida, sin otras personas de fondo (`figuras/cap_interfaz_camara.png`). Señas que el modelo reconoce mejor: Verde, Dónde, Dulce de leche, Agua y Rojo. |
| **12.7 Diseño adaptable** | Probar en un teléfono real con la cámara, **después del despliegue** (sin HTTPS el teléfono no da acceso a la cámara). |
| **7.1.2 Tablero Kanban** | Captura del tablero de Trello. La herramienta ya figura en el texto. |

### D. Capítulo de resultados sobre una sola ejecución — **Jonatan**

| Hueco | Qué dato hay que traer |
|---|---|
| **14.2 Resultado principal** | Repetir sobre la ejecución regularizada el análisis por clase (14.4), la correlación con la retención (14.5) y la comparación con la línea base (14.7.1). |
| **14.4.2 Matriz de confusión** | Mirar los videos de LSA64 par por par y confirmar si las señas más confundidas comparten configuración y difieren en ubicación; ilustrarlo con una figura. |
| **14.7.2** | Describir en qué se parecen las señas que el sistema confunde. |
| **9.10 Calidad de la detección** | Probar umbrales más bajos y el modo VIDEO de MediaPipe y medir cuántas secuencias se recuperan. |

Si se adopta v2 (decisión 3), este bloque hay que hacerlo sobre v2.

### E. Trámites y forma — **los dos**

| Hueco | Qué falta | Quién |
|---|---|---|
| **Agradecimientos** | Redacción final con palabras propias; hoy hay un borrador. | Los dos |
| **8.4 Prototipo de escritorio** | Citar de dónde salió (autor y repositorio) y agregarlo a la bibliografía. Si es código de terceros y no se cita, es lo primero que puede preguntar un jurado. | Los dos |
| **Anexo C** | La dirección de la carpeta compartida "Fuentes del proyecto". | Leandro |
| **Anexo A** | Figura con señas de ejemplo de LSA64 (CC BY-NC-SA 4.0, citando la fuente). | Jonatan |
| **13.1.3** | Ejecución automática de las pruebas con GitHub Actions. | Leandro |

Fuera del documento, pero visible en el repositorio:

- **Jonatan tiene que configurar su nombre en git** (`git config --global user.name "Jonatan Videla"`
  y su mail). 26 de los 39 commits figuran como "Your Name".
- **Licencia.** El repositorio no tiene `LICENSE`. La tesina dice "código publicado en un
  repositorio público", no "código abierto"; si se agrega una licencia (por ejemplo MIT), se puede
  volver a decir "código abierto".

---

## 3. Orden sugerido

1. **Decisiones 2 y 3** con el tutor, en la misma consulta.
2. **A → B → C.** El despliegue habilita la latencia "desplegado" y las capturas conviene sacarlas
   del sistema ya publicado.
3. **D**, una vez decidido si es sobre v1 regularizado o sobre v2.
4. **E** en los huecos muertos, y la regeneración final del documento.

Después de cualquier cambio en los capítulos:

```bash
python docs/tesina/build/verificar.py
cd docs/tesina/build && node build.js && powershell -ExecutionPolicy Bypass -File postproceso.ps1
```

`postproceso.ps1` imprime la cantidad de páginas: **hay que mirar ese número** (decisión 1).

El Google Doc del Drive es una copia del `.docx`: los cambios se hacen en los `.md`, se regenera el
Word y se sube como nueva versión del mismo archivo en Drive. Lo que se edite a mano en el Drive se
pierde en la próxima regeneración si no se pasa a los `.md`.

---

## 4. Lo que ya NO es un hueco

- **El Abstract** se sacó (decisión del 24/11): el documento lleva sólo el Resumen en castellano.
- **El agradecimiento al LIDI** se sacó de Agradecimientos (24/11); LSA64 se sigue citando en el
  texto.
- **La prueba con personas usuarias de LSA** quedó fuera del alcance (24/11): no se hace ni antes
  de la entrega ni en la mesa final. La tesina lo declara como limitación en 1.5 y 14.8.2, y como
  trabajo futuro en las conclusiones.
- **La profesora de Metodología**: Elisa Galdame, en la carátula (`build/build.js`).
- **La herramienta del tablero Kanban**: Trello (7.1.2 y tabla de herramientas). Falta sólo la
  captura.
- **Las inconsistencias con el repositorio** corregidas el 24/11: 132 pruebas (no 116), MediaPipe
  0.10.35, reproducibilidad sólo dentro del mismo equipo, capítulo 10 y Anexo B con la configuración
  del modelo servido, tiempos de entrenamiento ("pocos minutos", no "segundos"), S10 = 81,9 %,
  convención de commits desde agosto, pruebas desde agosto, Scrum descripto como se aplicó, citas
  de Mindlin parafraseadas (sin comillas, no necesitan página).
- **La dirección configurable del backend**: `frontend/config.js` y `frontend/app.js:37`.
- **La imagen de Docker (13.3.1)**: 1,9 GB sin comprimir, 571 MB comprimida, responde a `/health` a
  los ~15 s.
- **La línea base no neuronal**: 78,2 % ± 7,5 % con el mismo protocolo (14.7.1).
- **El abecedario**: salió del alcance; el pipeline queda documentado en 9.8.
