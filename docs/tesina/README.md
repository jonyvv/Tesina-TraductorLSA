# Documento de la tesina

El texto de la tesina vive en Markdown, un archivo por capítulo, y se convierte a Word (y PDF) con el formato de la facultad. Así los dos pueden editar con git, ver los cambios en los diffs y regenerar el documento cuando quieran.

```
docs/tesina/
├── PLAN.md              # decisiones pendientes y tareas por persona
├── capitulos/           # el texto: 00_ a 51_, en el orden del documento
├── figuras/             # PNG + generar_figuras.py (las figuras de resultados salen de ml/reports/)
├── build/
│   ├── build.js         # Markdown -> .docx con el formato de la facultad
│   ├── postproceso.ps1  # Word: actualiza índices, exporta PDF, cuenta páginas
│   └── verificar.py     # bibliografía citada, glosario en negrita, lista de pendientes
└── salida/              # Tesina-TraductorLSA.docx / .pdf
```

## Regenerar

Desde la raíz del repo:

```bash
.venv-ml/Scripts/python docs/tesina/figuras/generar_figuras.py   # sólo si cambiaron los reportes
python docs/tesina/build/verificar.py
cd docs/tesina/build
npm install          # una sola vez
node build.js
powershell -ExecutionPolicy Bypass -File postproceso.ps1
```

`postproceso.ps1` necesita Microsoft Word: actualiza el índice general, el de figuras y el de tablas (que `build.js` deja como campos), guarda, exporta el PDF e imprime la cantidad de páginas. **Hay que mirar ese número: el rango pedido es 90–120.**

Para probar con interlineado doble (el que pide el formato escaneado de la facultad): `INTERLINEADO=doble node build.js`.

## Qué Markdown entiende el build

Es un subconjunto chico:

| Escribís | Sale |
|---|---|
| `# Capítulo N — Título` | Título 1, hoja nueva, centrado. Es también el texto de la cabecera de ese capítulo |
| `## N.M Título` / `### N.M.K Título` | Título 2 / Título 3 (la numeración se escribe a mano) |
| `**negrita**`, `*cursiva*`, `` `código` `` | Igual |
| `[^clave]` + una línea `[^clave]: texto.` | Nota al pie |
| `![Leyenda.](figuras/x.png){ancho=14 #fig:x}` | Figura numerada con leyenda (ancho en cm) |
| `Tabla: Leyenda. {#tab:x}` seguida de una tabla con pipes | Tabla numerada con leyenda arriba |
| `@fig:x` / `@tab:x` en el texto | "Figura N" / "Tabla N", con el número correcto |
| `- item` / `1. item` | Viñetas / lista numerada |
| `> texto` | Cita larga (sangrada, interlineado simple) |
| Bloque entre ` ``` ` | Código en monoespaciado |
| `[[PENDIENTE: ...]]` | Texto resaltado en amarillo, para lo que falta completar |

Las referencias a secciones ("sección 9.7") van escritas a mano: si se renumera un capítulo, buscarlas con `grep -rn "sección 9\." docs/tesina/capitulos`.

## Convenciones del texto

- Registro impersonal ("se eligió", "el sistema utiliza"), como recomienda el documento de formato. Sin voseo en el cuerpo de la tesina.
- Citas en formato APA 7: (Apellido, año), (Apellido y Apellido, año), (Apellido et al., año). Toda entrada de la bibliografía tiene que estar citada, y toda cita tiene que tener entrada: `verificar.py` lo controla.
- Términos del glosario en **negrita** donde se definen en el cuerpo.
- Números con coma decimal y espacio antes del %: "79,8 %".
- Ningún número de resultados se escribe "de memoria": salen de `ml/reports/` o de `backend/models/modelo_lsa64_lstm.json`.
