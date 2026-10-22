# Anteproyecto corregido

Versión corregida del anteproyecto aprobado (`FACULTAD/TESINA/anteproyecto.pdf`, fuera del repo). Se corrigieron la forma, el cumplimiento de la "Guía de Anteproyecto" de la facultad y la redacción, **sin cambiar el plan**: el alcance (abecedario y palabras clave), las tecnologías previstas (React, Jupyter, Colab, TensorFlow o PyTorch), los roles y el cronograma son los del aprobado. Las diferencias entre ese plan y lo que finalmente se hizo se justifican en la tesina (sección 8.8).

```
docs/anteproyecto/
├── anteproyecto.md          # el texto, secciones 1 a 12 de la guía
├── build.js                 # Markdown -> .docx con el formato del aprobado
├── figuras/                 # logo, Gantt y bocetos + generar_figuras.py
└── salida/                  # Anteproyecto-TraductorLSA.docx / .pdf
```

## Regenerar

```bash
.venv-ml/Scripts/python docs/anteproyecto/figuras/generar_figuras.py   # sólo si cambia el Gantt o los bocetos
cd docs/anteproyecto
node build.js
powershell -ExecutionPolicy Bypass -File ../tesina/build/postproceso.ps1 -Docx salida/Anteproyecto-TraductorLSA.docx -Entrega AnteproyectoProvenzanoVidela
```

`build.js` usa la biblioteca `docx` de `docs/tesina/build` (`npm install` allí una vez). El Markdown es el mismo subconjunto que el de la tesina, sin notas al pie.

## Qué se corrigió respecto del aprobado

**Contra la guía de la facultad**

- Secciones numeradas del 1 al 12, como enumera la guía, más la hoja de seguimiento.
- Objetivos específicos medibles. Se agregó el de despliegue, que el aprobado ya preveía en los alcances, los resultados esperados y el cronograma, pero no entre los objetivos.
- Alcances con la delimitación del marco de acción y lo que queda fuera. Limitaciones como factores que restringen el resultado: tiempo, dinero, acceso a datos y condiciones de uso.
- Justificación con el aporte, la solución y los beneficios sociales, económicos y académicos (antes, sólo los sociales).
- Resultados esperados enumerados y verificables.
- Planteamiento del problema en una tabla con el esquema de la caja negra (A → B) que propone la guía, citando a Krick (1995).
- Metodología como etapas secuenciales y técnicas (la guía lo pide así); la gestión con Scrum y Kanban queda como subsección.
- Gantt redibujado: en el PDF aprobado los nombres de las tareas no se leían (gris claro sobre blanco). Las mismas tareas y semanas, más los hitos de evaluación académica que pide la guía (revisiones con el tutor y entrega).
- Recursos humanos con horas-hombre. **Se agregaron 10.3 Recursos materiales y 10.4 Recursos financieros**, con el monto total que pide la guía.
- Anexos incluidos y referenciados desde el cuerpo (en el aprobado sólo se enumeraban como sugeridos).
- Bibliografía en APA 7, agrupada por tipo, con todas las entradas citadas en el texto.

**Contenido**

- Organización Mundial de la Salud: el *World report on hearing* es de 2021 y da 430 millones de personas con pérdida auditiva discapacitante (el aprobado decía 2020 y 466 millones).
- Mindlin (2021): título correcto de la tesina, y aclaración de que su 99,4 % se midió con las mismas personas en entrenamiento y prueba.
- Se agregó la Ley 27.710 (2023), que reconoce la LSA.

**Redacción**

- "Lengua" de señas en lugar de "lenguaje", también en el título, y Lengua de Señas Argentina en lugar de "Lenguaje de señas argentino".
- "Personas sordas" y "comunidad sorda" en lugar de "sordomudas" (término que la comunidad considera inadecuado; ver la sección 1.1 de la tesina).
- Erratas: "la señas", "react y fast api", "Git Hub", "google colab", "CRONOGRAMA(Diagrama de gantt)".

## Valores a confirmar

El presupuesto (10.4) y las horas-hombre (10.1) son estimaciones con supuestos explícitos: 10 h semanales por persona, USD 10 por hora, notebooks de USD 900 amortizadas en 36 meses, Internet a USD 20 por mes y Colab Pro durante 3 meses. Ajustarlos en `anteproyecto.md` si los reales son otros.
