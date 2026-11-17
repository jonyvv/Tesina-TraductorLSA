# Qué falta completar en la tesina

_Al 17/11/2026, sobre `salida/Tesina-TraductorLSA.docx` (121 páginas, 34.370 palabras)._

Hay **17 huecos** en los capítulos, más el nombre del profesor de Metodología en la carátula
(decisión 4), y **5 decisiones** que no dependen de ningún dato. Los huecos aparecen **resaltados en
amarillo** en el Word, así que se los puede recorrer abriendo el archivo y buscando el resaltado;
`python docs/tesina/build/verificar.py` lista los 17 de los capítulos en orden.

Este archivo los agrupa de otra manera: **por la acción que los cierra**. Casi todos los huecos no
son 17 tareas distintas, son 6 tareas que tapan varios huecos cada una. Las decisiones van primero
porque dos de ellas cambian qué hay que escribir.

---

## 1. Decisiones (no hace falta medir nada)

| # | Decisión | Estado | Por qué importa |
|---|---|---|---|
| **1** | **El documento quedó en 121 páginas** y el rango pedido es 90–120 | Abierta | Antes de la alineación con el anteproyecto v3 estaba exactamente en 120. Para bajar una página hay que recortar unas 250 palabras de texto existente, o preguntarle a Mirallas si acepta 121. Va junto con la consulta del interlineado (D2), que sigue sin respuesta: con interlineado doble el documento da 144 páginas. |
| **2** | **El título difiere del anteproyecto** | Abierta | Anteproyecto: "Sistema de traducción de lenguaje de señas LSA con inteligencia artificial". Tesina: "Sistema de traducción de Lengua de Señas Argentina a texto en tiempo real mediante inteligencia artificial". Hay que unificarlo con el tutor; cambiar el título de la tesina no es una decisión del equipo. |
| **3** | **¿Las *features* v2 pasan a ser el resultado principal?** | Abierta | Las corridas del 12/11 dan **83,4 % ± 7,2 %** con v2 (`ml/reports/loso_v2_gpu.json`) contra 81,7 % ± 7,9 % con v1, y la ventaja sobre el bosque aleatorio sube de 1,6 a 2,8 puntos, ganando en 8 de 10 sujetos (`comparacion_v2_rf2.json`). El documento informa v1 porque **el modelo servido sigue en `feature_version: v1`**. Si se adopta v2 hay que reentrenar el `.pt`, cambiar el modelo servido y reescribir 14.2, 14.7.1, 14.10 y las conclusiones. Si no, conviene al menos mencionar v2 en 14.6 como resultado medido. |
| **4** | **Nombre del/la profesor/a de Metodología** | Abierta | Va en la carátula. Se completa en `build/build.js`, constante `META.metodologia`. Es el único hueco que no está en los capítulos. |
| **5** | **Cliente sin React** (D3 del plan) | Resuelta en el texto, sin validar | 8.8.2 justifica la decisión y el anteproyecto v3 la respalda en su B.6. Falta el visto del tutor. |

---

## 2. Trabajo ordenado, con lo que cierra cada cosa

### A. Desplegar el sistema — **Leandro**

Cierra 3 huecos y es lo que más desbloquea.

| Hueco | Qué dato hay que traer |
|---|---|
| **13.4 Despliegue** | La dirección pública de la interfaz, los pasos seguidos y la latencia medida contra el servicio expuesto por el túnel. |
| **Conclusiones, punto 5** | Una línea con el resultado del despliegue. |
| **13.3.1 Tamaño de la imagen** | Construir la imagen y anotar su tamaño final (el `Dockerfile` ya instala PyTorch-CPU). |

El esquema ya está decidido y documentado: interfaz estática + servicio de inferencia local expuesto
con un túnel. En el código ya están resueltas las dos piezas que hacían falta: la dirección
configurable del backend (`frontend/config.js`, `frontend/app.js:37`) y la validación de orígenes
tanto en HTTP como en el WebSocket (`LSA_CORS_ORIGINS`).

Tres cosas de ngrok a tener en cuenta (según su documentación de la capa gratuita, consultada el
17/11):

- **El dominio es fijo.** Cada cuenta gratuita tiene un dominio asignado que no cambia al reiniciar
  el túnel, así que `config.js` se configura una sola vez. La tesina (2 y 13.4) ya lo dice así.
- **Página de aviso para navegadores.** La capa gratuita intercala una página de advertencia delante
  del tráfico que viene de un navegador. Si los `fetch` del cliente a `/health` y `/model/info`
  reciben ese HTML en lugar del JSON, hay que mandarles el encabezado `ngrok-skip-browser-warning`
  (el backend ya acepta cualquier encabezado en CORS, `backend/app/main.py:153`). Probarlo al
  desplegar; con el WebSocket la documentación no dice nada.
- **Cupos.** 1 GB de transferencia y 20 000 pedidos HTTP por mes.

### B. Medir la latencia real — **Leandro**, después de A

| Hueco | Qué dato hay que traer |
|---|---|
| **14.8.1 Latencia** | Una tabla con latencia total, cómputo del servidor y tiempo de red, en tres escenarios (servidor local, red local, desplegado) y a 8 y 15 cuadros por segundo. Los números salen del panel de rendimiento del cliente. |

### C. Capturas de la aplicación funcionando — **Leandro**

Cuatro capturas tapan 2 huecos de la tesina y los 4 espacios reservados del Anexo B.8 del
anteproyecto. Es el hueco que más se nota en una defensa: las tesinas aprobadas de la carpeta tienen
muchas capturas en la parte de implementación.

| Hueco | Qué falta |
|---|---|
| **12.1 Estructura** | Captura con la cámara activa, el esqueleto dibujado y una seña reconocida. |
| **12.7 Diseño adaptable** | Verificar la interfaz en un teléfono real. En la prueba con navegador sin interfaz gráfica, a 400 px de ancho, algunos paneles parecían desbordar a la derecha: si desborda, es un arreglo de CSS, no sólo una captura. |

### D. Cerrar el capítulo de resultados sobre una sola ejecución — **Jonatan**

Son 4 huecos y todos tienen la misma raíz: parte de los análisis se hicieron sobre la ejecución de
referencia (sin regularizar, 79,8 %) porque era la que estaba disponible en ese momento, y el
resultado principal es el de la ejecución regularizada (81,7 %).

| Hueco | Qué dato hay que traer |
|---|---|
| **14.2 Resultado principal** | Repetir sobre la ejecución regularizada: el análisis por clase (14.4), la correlación con la retención (14.5) y la comparación con la línea base (14.7.1), para que todo el capítulo informe la misma configuración. |
| **14.4.2 Matriz de confusión** | Mirar los videos de LSA64 par por par y confirmar si las señas más confundidas comparten configuración manual y difieren en ubicación; ilustrarlo con una figura. |
| **14.7.2** (tarea M4) | Describir en qué se parecen las señas que el sistema confunde. Ojo: **no** es para confirmar la hipótesis de la ventaja temporal, que quedó descartada. |
| **9.10 Calidad de la detección** | Probar umbrales de detección más bajos y el **modo VIDEO** de MediaPipe (que sigue la mano entre cuadros) y medir cuántas secuencias se recuperan. Es la mejora con más potencial según el propio capítulo 14. |

Si se adopta v2 (decisión 3), este bloque hay que hacerlo sobre v2 y no sobre la ejecución
regularizada de v1. Conviene decidir 3 antes de empezar D.

### E. Evaluación con personas reales — **los dos**

Dos huecos, y es el único que necesita gente ajena al equipo. Tiene un costo de coordinación que
conviene arrancar temprano, no al final.

| Hueco | Qué falta |
|---|---|
| **14.8.2 Brecha con el uso en tiempo real** | Implementar las tres acciones previstas: (a) alinear la frecuencia de muestreo entre entrenamiento y uso, (b) reemplazar la ventana fija por una segmentación según la presencia de las manos, (c) medir el desempeño con señas hechas frente a la cámara por personas que no estén en LSA64, registrando aciertos, errores y señas no detectadas. |
| **1.5 Consideraciones éticas** | La prueba con personas usuarias de LSA, con consentimiento informado. El protocolo y los resultados se describen en 14.8.2, así que este hueco se cierra solo cuando se cierra el otro. |

### F. Trámites y forma — **los dos**

Cinco huecos sueltos, ninguno depende de nada. Son los más rápidos de cerrar.

| Hueco | Qué falta | Quién |
|---|---|---|
| **Agradecimientos** | Redacción final; hoy hay un borrador. | Los dos |
| **Anexo C** | La dirección de la carpeta compartida "Fuentes del proyecto" que exige la facultad. | Leandro |
| **Anexo A** | Una figura con señas de ejemplo tomadas de los videos de LSA64 (licencia CC BY-NC-SA 4.0, citando la fuente), en particular de los pares más confundidos. | Jonatan |
| **7.1.2 Tablero Kanban** | Indicar qué herramienta se usó para el tablero e incluir una captura. | Los dos |
| **13.1.3** | Configurar la ejecución automática de las pruebas en cada envío al repositorio (GitHub Actions). No es imprescindible para la entrega, pero es una línea que suma en calidad. | Leandro |

---

## 3. Orden sugerido

1. **Decisiones 1 a 4.** La 3 (v2) cambia el bloque D y parte del capítulo 14; la 1 y la 2 van en la
   misma consulta al tutor.
2. **E (coordinación de la prueba con personas)** en paralelo con todo lo demás, porque depende de
   terceros.
3. **A → B → C.** El despliegue habilita la latencia "desplegado" y las capturas conviene sacarlas
   del sistema ya publicado.
4. **D**, una vez decidido si es sobre v1 regularizado o sobre v2.
5. **F** en los huecos muertos, y la regeneración final del documento.

Después de cualquier cambio en los capítulos:

```bash
python docs/tesina/build/verificar.py
cd docs/tesina/build && node build.js && powershell -ExecutionPolicy Bypass -File postproceso.ps1
```

`postproceso.ps1` imprime la cantidad de páginas: **hay que mirar ese número** (decisión 1).

---

## 4. Lo que ya NO es un hueco

Para que no se vuelva a escribir como pendiente algo que está hecho:

- **La dirección configurable del backend.** El anteproyecto v3 (B.5) la declara pendiente, pero ya
  está implementada: `frontend/config.js` y `frontend/app.js:37`.
- **PyTorch sólo para CPU.** Está en el `Dockerfile:38` de esta rama; lo único que falta es anotar el
  tamaño de la imagen.
- **La línea base no neuronal.** Medida: 78,2 % ± 7,5 % con el mismo protocolo (14.7.1). El Gantt la
  tenía como pendiente y se corrigió.
- **El abecedario.** Salió del alcance (anteproyecto v3), así que ya no es algo que falte: el pipeline
  queda documentado en 9.8 y el abecedario completo, en "Futuro del proyecto".
- **Los beneficios económicos y académicos del anteproyecto.** Están escritos; la marca de pendiente
  que quedó arriba en `anteproyecto-v3.docx` es una marca vencida y hay que borrarla de ese `.docx`.
