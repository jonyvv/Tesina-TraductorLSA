# Plan de trabajo — tesina Traductor LSA

_Armado el 30/9/2026 a la noche, junto con el primer borrador completo del documento._

Este archivo tiene tres partes: qué quedó hecho, qué decisiones tienen que tomar ustedes antes de seguir, y las tareas concretas separadas por persona. Los códigos (D1, F3, M2…) son los mismos que aparecen resaltados en amarillo como `PENDIENTE` dentro del documento, así cada hueco del texto apunta a una tarea de acá.

---

## 1. Qué quedó hecho

### Documento (rama `docs/tesina`)

- **Borrador completo de la tesina: 115 páginas** (rango pedido: 90–120), en `docs/tesina/salida/Tesina-TraductorLSA.pdf` y `.docx`.
- Estructura según el "Documento general de redacción" del profesor (2019) y el formato de la facultad: carátula en dos hojas, agradecimientos, resumen (+ abstract), introducción, índices general/figuras/tablas, Parte I marco teórico (7 capítulos), Parte II implementación (7 capítulos), conclusiones con contribuciones y futuro, 4 anexos, bibliografía APA y glosario.
- Formato: A4, márgenes 4/2/3/3 cm, Times New Roman 11, títulos 18/14/12, capítulos en hoja nueva y centrados, cabecera "(capítulo) — (título) - (alumnos)" sin cabecera en la primera hoja de cada capítulo, pie "Universidad Champagnat · año · Página X de Y", notas al pie, referencias cruzadas a figuras/tablas/secciones.
- **19 figuras** (16 generadas desde los reportes reales del repo, nada inventado, y 3 capturas de la app) y **21 tablas**.
- Se aplicaron de entrada las correcciones que le hicieron a la tesina aprobada de 2025 (archivo "Sugerencias y Correcciones" del Drive): marco teórico ilustrado, conceptos de la carrera (REST, WebSocket, Python, IA) explicados antes de usarlos, referencias cruzadas y notas al pie, **toda la bibliografía citada al menos una vez**, **términos del glosario en negrita en el cuerpo**, cabeceras que respetan el capítulo actual. `build/verificar.py` chequea los dos primeros automáticamente.
- Todo el contenido está en Markdown (`docs/tesina/capitulos/`), uno por capítulo, para que lo editen los dos con git y se pueda regenerar el Word cuando quieran (ver `docs/tesina/README.md`).

### Tesinas del Drive

Revisé las 25 tesinas/seminarios en PDF de la carpeta, con foco en las más recientes (Álvarez 2025, las seis de 2022, Romero y Bajda 2021). Lo que tomé como guía:

- **Álvarez 2025** (la más reciente, también IA + API): estructura Marco teórico → Desarrollo de la implementación → Conclusiones → Glosario → Bibliografía, interlineado 1,5, ~158 páginas. Sus correcciones están aplicadas (ver arriba).
- **Veisaga 2022** (ML + web): separa "primera implementación" y "segunda implementación" y documenta el proceso con datos; acá eso aparece como la historia de errores corregidos y experimentos, que es lo más fuerte del trabajo de ustedes.
- **Videla 2022** (reconocimiento facial con OpenCV): marco teórico que arranca de lo general (lenguajes, historia del reconocimiento) a lo particular; lo repliqué con IA → ML → redes → visión → estado del arte.
- Patrón común en todas: muchas capturas de pantalla e ilustraciones en la parte de implementación. Ese es el principal hueco que les queda (ver F1).

### Hallazgos sobre el antecedente principal (importante para la defensa)

- **El 99,4 % de Mindlin (2021) es *signer-dependent*.** Lo leí: usó 5-fold con videos al azar y él mismo escribe que "esta separación no es independiente del sujeto". No es comparable con el 79,8 % de ustedes. Ya está explicado en la sección 5.3.1. Si el tribunal saca el 99,4 %, esa es la respuesta.
- **Ojo con el rango que figuraba en `docs/PROXIMOS_PASOS.md`** ("74,2 % a ~91,7 % signer-independent"): Akdag y Baykan (2024, *Electronics*) reportan **98,53 % signer-independent** sobre LSA64 con una 3D-CNN sobre video RGB. El 79,8 % queda por encima de Marais 2022 (74,22 %) pero lejos de ese techo. El documento lo dice y lo justifica (modelo liviano sobre landmarks, viable en capa gratuita, vs. 3D-CNN que necesita GPU). Conviene corregir PROXIMOS_PASOS.

---

## 2. Decisiones que tienen que tomar ustedes

| # | Decisión | Opciones | Mi recomendación |
|---|---|---|---|
| **D1** | Alcance del **abecedario** (el anteproyecto lo promete, LSA64 no tiene letras y hoy sólo hay A y C capturadas por Leandro) | (a) LSA16: 16 configuraciones, público, sin capturar nada; (b) capturar el abecedario con varias personas y sesiones (+ consentimiento informado); (c) sacarlo del alcance | (a) o (c). Con el tiempo que queda, (b) es lo más caro y lo más riesgoso. Hay que hablarlo con el tutor porque cambia un alcance del anteproyecto. |
| **D2** | **Interlineado** | El formato escaneado de la facultad dice "doble"; las tesinas aprobadas recientes usan 1,5 | 1,5 (es lo que está). Con doble el documento da **144 páginas** y se pasa de 120. Confirmar con Mirallas. Para probar: `INTERLINEADO=doble node build.js`. |
| **D3** | Cliente **sin React** (el anteproyecto decía React) | Mantener HTML/JS o migrar | Mantener y justificarlo (ya está en 8.8.2). Validar con el tutor. |
| **D4** | **Reparto de roles**: el anteproyecto dice Leandro = red neuronal, Jonatan = web; en la práctica es al revés | Unificar | Escribir lo real. El texto (7.2) ya lo pone así, con un pendiente para confirmar. |
| **D5** | **Declarar el uso de asistentes de IA** en el desarrollo y la redacción | Declararlo o no | Declararlo en 7.6 con una frase. Varios commits tienen `Co-Authored-By: Claude`, así que es visible en el repo de todas formas. |
| **D6** | Nombre del/la **profesor/a de Metodología** (va en la carátula) | — | Completar en `build/build.js` (constante `META`). |

---

## 3. Hallazgos técnicos de la revisión del código

Ordenados por gravedad. Ninguno rompe los tests (116/116 pasan), pero varios afectan lo que se puede afirmar en la defensa.

1. **Desajuste entrenamiento / tiempo real (el más importante).** El modelo se entrenó con señas **completas** a **30 fps** (LSA64 a 60 fps con `frame_step=2`). En la app recibe **ventanas deslizantes de 8 frames** (= `min_seq_len`) a **8 fps** (default del frontend). O sea: ve las señas ~4 veces "aceleradas" y recortadas en cualquier punto. El 79,8 % del LOSO no dice nada sobre cómo anda en vivo. Está documentado como limitación en 14.8.2; las tareas M5 y F3 lo atacan.
2. **Dos manos con la misma lateralidad se pisan** (`common/features.py`, `build_feature_vector`): si MediaPipe etiqueta las dos manos como `Right` (pasa con oclusiones), la segunda sobreescribe a la primera y se pierde una mano sin aviso. Tocarlo cambia las features, así que va con reentrenamiento (M3).
3. **MediaPipe en modo IMAGE**: el `HandLandmarker` corre en `RunningMode.IMAGE`, que detecta la palma de cero en cada frame y no hace seguimiento. El modo VIDEO usa tracking entre frames: probablemente suba la retención (la variable que más explica los errores, r = 0,68) y baje la latencia. Para LSA64 es directo; para el backend hay que pasar timestamps por sesión (M3).
4. **Imagen Docker con PyTorch + CUDA**: `backend/requirements.txt` instala `torch` genérico (~2,5 GB de CUDA que no se usa). **Arreglado en la rama `mejoras/despliegue`** (ver abajo).
5. **CORS en `"*"`** en producción. **Arreglado en la rama `mejoras/despliegue`**: ahora se configura con `LSA_CORS_ORIGINS`.
6. **README desactualizado**: decía que los `.pt` no se versionan (sí se versionan) y que las glosas eran `Opaque, Learn, Candy`. **Corregido en `docs/tesina`.**
7. **Responsive en móvil**: en una captura con Edge headless a 400 px algunos paneles parecían cortarse a la derecha. Puede ser un artefacto del modo headless (tiene ancho mínimo), hay que mirarlo en un teléfono real (F2).
8. Los tests e2e se saltean solos si no está instalado `httpx2` (`pytest.importorskip("httpx2")`); en tu `.venv-ml` está, pero en una CI limpia podrían saltearse sin que nadie se entere (Q1).

---

## 4. Tareas

### Leandro — frontend, backend, infraestructura, documento

| # | Tarea | Por qué | Estimado |
|---|---|---|---|
| **R1** | Revisar la rama `mejoras/despliegue` y mergearla a `test` | Imagen Docker liviana + CORS configurable; precondición del deploy | 15 min |
| **D1** | **Deploy** en Render (o Railway) con el Dockerfile. Servir con HTTPS (si no, la cámara no anda). Setear `LSA_CORS_ORIGINS` al dominio. Medir tiempo de arranque en frío. | Objetivo específico 5 del anteproyecto; hoy no está cumplido | 2–3 h |
| **F1** | **Capturas con la cámara andando**: seña reconocida, esqueleto dibujado, panel de métricas con valores, texto acumulado; idealmente también en un celular. Guardarlas en `docs/tesina/figuras/` y agregarlas en el cap. 12 | Las tesinas aprobadas están llenas de capturas; es lo que más le falta al documento | 1 h |
| **F3** | **Medición de latencia**: con el panel de rendimiento, anotar latencia total / cómputo servidor / red a 8 y 15 fps, en local, en LAN y contra el deploy. Llenar la tabla de 14.8.1 | Hoy la sección de tiempo real no tiene números | 1 h |
| **F2** | Verificar el responsive en un teléfono real; corregir si desborda | 12.7 | 30 min |
| **Q1** | GitHub Actions que corra `pytest tests ml/tests` en cada push (y que falle si se saltean los e2e) | 13.1.3; es una línea más de "aseguramiento de la calidad" | 45 min |
| **F4** | (Si llegan) Prueba con 2–3 personas que sepan LSA: lista de 10 señas de LSA64, contar aciertos / errores / no detectadas. Con consentimiento | Cubre 1.5 y 14.8.2(c); el tribunal seguro lo pregunta | 2 h + coordinación |
| **DOC** | Completar los pendientes del documento que caen de tu lado (agradecimientos, herramienta Kanban + captura del tablero, URL del repo, fechas reales del Gantt, carpeta "Fuentes del proyecto" en Drive) | Ver `python docs/tesina/build/verificar.py`, que lista los 19 pendientes | 1–2 h |

### Jonatan — modelo

| # | Tarea | Por qué | Estimado |
|---|---|---|---|
| **M5** | **Cerrar la brecha con el tiempo real** (junto con Leandro, porque toca backend): (a) alinear fps — o reentrenar con `frame_step` equivalente al del cliente, o subir el cliente a 15 fps y remuestrear; (b) en vez de ventana fija de 8, acumular mientras haya manos y clasificar la secuencia completa al bajarlas (que es como se entrenó) | Hallazgo técnico 1. Sin esto, la demo en vivo puede andar bastante peor que el 79,8 % | 3–4 h |
| **M1** | **LOSO completo** con `--dropout 0.4 --aug-frame-drop 0.30` (combinados, y si hay tiempo cada uno por separado). Si mejora, reentrenar el modelo servido con esa config | El screening dio +1,9 y +1,7 en 2 de 10 folds; sin LOSO completo no se puede afirmar | ~1 h de cómputo cada uno |
| **M2** | **Baseline Random Forest** sobre el mismo caché (media/desvío/mín/máx por feature) evaluado con LOSO | El planteo promete "RF vs. red neuronal"; sin baseline no hay forma de justificar la BiLSTM (14.7.1) | 1 h |
| **M3** | **Retención**: probar `min_hand_detection_confidence` 0,4–0,5 y `RunningMode.VIDEO`; contar cuántas secuencias de las 338 descartadas se recuperan y correr LOSO. De paso, arreglar lo de las dos manos con la misma lateralidad | Es lo que más explica los errores (14.5) | 2 h + 47 min de extracción |
| **M4** | Mirar los videos de los pares más confundidos (Aprender↔Caramelo, Ayudar→Agradecer, Espumadera→Brillante, Hijo→Mujer) y escribir en una línea por par en qué se parecen. Sacar un frame de cada uno para una figura | Hoy el texto dice "comparten configuración y difieren en ubicación" pero sin verificar (14.4.2) | 1 h |

### Orden sugerido para mañana

1. **Leer el PDF** (aunque sea en diagonal) y tomar D1–D6. Lo que no les cierre del texto, anotarlo: es un borrador, la voz y el criterio finales son de ustedes.
2. Leandro: R1 → D1 (deploy) → F1/F3 (capturas y latencia, ya sobre el deploy).
3. Jonatan: M5 primero (es lo que más riesgo tiene para la demo), después M2 y M1 corriendo en paralelo mientras escribe M4.
4. Ir regenerando el documento a medida que se completan los pendientes (`node build.js` + `postproceso.ps1`) y subir el PDF al Drive con el nombre que pide la facultad: `ProvenzanoVidela20261001v01.pdf` (apellidos + fecha + versión).

---

## 5. Lo que hay en cada rama

- **`docs/tesina`** (desde `test`): `docs/tesina/` completo (capítulos, figuras, script de figuras, build a Word/PDF, verificador, este plan) + corrección del README.
- **`mejoras/despliegue`** (desde `test`): Dockerfile con PyTorch CPU y CORS configurable por variable de entorno. Con tests pasando. Chico y revisable en 5 minutos.

Ninguna de las dos está pusheada ni mergeada: quedan locales para que las revisen.
