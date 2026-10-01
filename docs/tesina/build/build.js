// =============================================================================
// docs/tesina/build/build.js
// -----------------------------------------------------------------------------
// Arma el documento de la tesina (.docx) a partir de los capítulos en Markdown
// de docs/tesina/capitulos/, aplicando el formato pedido por la facultad:
//
//   - A4, márgenes izq. 4 cm, der. 2 cm, sup./inf. 3 cm.
//   - Interlineado doble y texto justificado, Times New Roman 11.
//   - Títulos 18 / 14 / 12 pt; cada capítulo arranca en hoja nueva, centrado.
//   - Cabecera "(capítulo actual)   (título) - (alumnos)" en las hojas de texto,
//     nunca en la primera hoja de un capítulo ni en la carátula.
//   - Pie "Universidad Champagnat   año   página/cantidad" en todas las hojas
//     salvo la carátula.
//   - Índice general, de figuras y de tablas como campos de Word: se completan
//     al correr postproceso.ps1 (Word actualiza los campos y exporta el PDF).
//
// Uso (desde docs/tesina/build):
//     npm install          (una sola vez)
//     node build.js        -> ../salida/Tesina-TraductorLSA.docx
//     powershell -File postproceso.ps1   -> actualiza índices y genera el PDF
//
// El Markdown es un subconjunto chico, documentado en docs/tesina/README.md.
// =============================================================================
"use strict";

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, ExternalHyperlink, Footer, FootnoteReferenceRun,
  Header, HeadingLevel, HighlightColor, ImageRun, LevelFormat, Packer, PageBreak, PageNumber,
  Paragraph, SequentialIdentifier, ShadingType, Table, TableCell, TableOfContents, TableRow,
  TabStopType, TextRun, VerticalAlign, WidthType,
} = require("docx");

const RAIZ = path.resolve(__dirname, "..");
const CAPS = path.join(RAIZ, "capitulos");
const SALIDA = path.join(RAIZ, "salida");

// --- Datos de la carátula ----------------------------------------------------
const META = {
  universidad: "Universidad Champagnat",
  facultad: "Facultad de Informática y Diseño",
  carrera: "Licenciatura en Sistemas de Información",
  tipo: "Tesina de Licenciatura en Sistemas de Información",
  titulo: "Sistema de traducción de Lengua de Señas Argentina a texto en tiempo real mediante inteligencia artificial",
  tituloCorto: "Traductor LSA con IA",
  alumnos: ["Leandro Provenzano", "Jonatan Videla"],
  alumnosCorto: "Provenzano, L. - Videla, J.",
  coordinador: "Ing. Mario Alberto Mirallas",
  tutor: "Marcelo Palma",
  metodologia: "[[PENDIENTE: nombre del profesor/a de Metodología]]",
  anio: "2026",
};

// --- Orden del documento -------------------------------------------------------
// tipo: "front" (apartado sin numerar), "indices", "parte" (portadilla),
// "capitulo", "anexos" (portadilla de anexos), "biblio", "glosario".
const ORDEN = [
  { tipo: "front", archivo: "00_agradecimientos.md" },
  { tipo: "front", archivo: "01_resumen.md" },
  { tipo: "front", archivo: "02_introduccion.md" },
  { tipo: "indices" },
  { tipo: "parte", titulo: "PARTE I", subtitulo: "Marco teórico" },
  { tipo: "capitulo", archivo: "10_cap01_comunidad_sorda_lsa.md" },
  { tipo: "capitulo", archivo: "11_cap02_aprendizaje_automatico.md" },
  { tipo: "capitulo", archivo: "12_cap03_redes_neuronales.md" },
  { tipo: "capitulo", archivo: "13_cap04_vision_por_computadora.md" },
  { tipo: "capitulo", archivo: "14_cap05_estado_del_arte.md" },
  { tipo: "capitulo", archivo: "15_cap06_aplicaciones_web_tiempo_real.md" },
  { tipo: "capitulo", archivo: "16_cap07_metodologia.md" },
  { tipo: "parte", titulo: "PARTE II", subtitulo: "Desarrollo de la implementación" },
  { tipo: "capitulo", archivo: "20_cap08_analisis_y_diseno.md" },
  { tipo: "capitulo", archivo: "21_cap09_datos_y_features.md" },
  { tipo: "capitulo", archivo: "22_cap10_modelo.md" },
  { tipo: "capitulo", archivo: "23_cap11_backend.md" },
  { tipo: "capitulo", archivo: "24_cap12_frontend.md" },
  { tipo: "capitulo", archivo: "25_cap13_calidad_y_despliegue.md" },
  { tipo: "capitulo", archivo: "26_cap14_resultados.md" },
  { tipo: "capitulo", archivo: "30_conclusiones.md" },
  { tipo: "anexos" },
  { tipo: "capitulo", archivo: "40_anexo_a_senas.md" },
  { tipo: "capitulo", archivo: "41_anexo_b_red.md" },
  { tipo: "capitulo", archivo: "42_anexo_c_instalacion.md" },
  { tipo: "capitulo", archivo: "43_anexo_d_cronograma.md" },
  { tipo: "capitulo", archivo: "50_bibliografia.md", biblio: true },
  { tipo: "capitulo", archivo: "51_glosario.md" },
];

// --- Medidas (DXA: 1 cm = 567) ----------------------------------------------------
const CM = 567;
const ANCHO_TEXTO = 21 * CM - 4 * CM - 2 * CM; // 15 cm
const FUENTE = "Times New Roman";
const MONO = "Consolas";
// Interlineado del cuerpo. El documento de formato de la facultad (escaneado,
// sin fecha) pide doble; las tesinas aprobadas más recientes (2022-2025) usan
// 1,5. Con doble, esta versión da ~144 páginas y se pasa del máximo de 120.
// Se puede forzar con la variable de entorno INTERLINEADO=doble.
const INTERLINEADO_DOBLE = process.env.INTERLINEADO === "doble" ? 480 : 360;
const INTERLINEADO_SIMPLE = 240;

// --- Estilos -------------------------------------------------------------------
const estilos = {
  default: {
    document: { run: { font: FUENTE, size: 22 }, paragraph: { spacing: { line: INTERLINEADO_DOBLE } } },
  },
  paragraphStyles: [
    {
      id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 36, bold: true, font: FUENTE },
      paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 2 * CM, after: 360, line: 360 }, outlineLevel: 0, keepNext: true },
    },
    {
      id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 28, bold: true, font: FUENTE },
      paragraph: { spacing: { before: 240, after: 120, line: 300 }, outlineLevel: 1, keepNext: true },
    },
    {
      id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 24, bold: true, italics: true, font: FUENTE },
      paragraph: { spacing: { before: 240, after: 60, line: 300 }, indent: { left: 0 }, outlineLevel: 2, keepNext: true },
    },
    {
      id: "Cuerpo", name: "Cuerpo", basedOn: "Normal", quickFormat: true,
      run: { size: 22, font: FUENTE },
      paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: INTERLINEADO_DOBLE }, indent: { firstLine: Math.round(1.25 * CM) } },
    },
    {
      id: "Caption", name: "caption", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 19, italics: true, font: FUENTE },
      paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240, line: INTERLINEADO_SIMPLE } },
    },
    {
      id: "Codigo", name: "Codigo", basedOn: "Normal",
      run: { size: 16, font: MONO },
      paragraph: { spacing: { line: INTERLINEADO_SIMPLE, before: 0, after: 0 }, indent: { left: 284 } },
    },
    {
      id: "Cita", name: "Cita larga", basedOn: "Normal",
      run: { size: 21, italics: true, font: FUENTE },
      paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { before: 240, after: 240, line: INTERLINEADO_SIMPLE }, indent: { left: 1134, right: 1134 } },
    },
    {
      id: "Bibliografia", name: "Bibliografia", basedOn: "Normal",
      run: { size: 22, font: FUENTE },
      paragraph: { alignment: AlignmentType.LEFT, spacing: { after: 160, line: 300 }, indent: { left: 709, hanging: 709 } },
    },
    {
      id: "CeldaTabla", name: "Celda tabla", basedOn: "Normal",
      run: { size: 18, font: FUENTE },
      paragraph: { spacing: { line: INTERLINEADO_SIMPLE, before: 20, after: 20 } },
    },
  ],
};

const numeracion = {
  config: [
    {
      reference: "viñetas",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
      ],
    },
  ],
};
// Cada lista numerada necesita su propia referencia para reiniciar en 1.
let contadorListas = 0;
function nuevaListaNumerada() {
  const ref = `num-${++contadorListas}`;
  numeracion.config.push({
    reference: ref,
    levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
  });
  return ref;
}

// --- Notas al pie ----------------------------------------------------------------
const notas = {};
let contadorNotas = 0;

// --- Texto en línea ----------------------------------------------------------------
// Soporta **negrita**, *cursiva*, `código`, [texto](url), [^nota] y
// [[PENDIENTE: ...]] (resaltado amarillo, para lo que falta completar).
function runs(texto, base = {}, notasArchivo = {}) {
  const salida = [];
  const patron = /(\[\[PENDIENTE:[^\]]*\]\])|(\*\*[^*]+\*\*)|(\*[^*\s][^*]*\*)|(`[^`]+`)|(\[\^[^\]]+\])|(\[[^\]]+\]\([^)]+\))/g;
  let ultimo = 0;
  let m;
  while ((m = patron.exec(texto)) !== null) {
    if (m.index > ultimo) salida.push(new TextRun({ text: texto.slice(ultimo, m.index), ...base }));
    const t = m[0];
    if (m[1]) {
      salida.push(new TextRun({ text: t.slice(2, -2), ...base, highlight: HighlightColor.YELLOW, bold: true }));
    } else if (m[2]) {
      salida.push(...runs(t.slice(2, -2), { ...base, bold: true }, notasArchivo));
    } else if (m[3]) {
      salida.push(...runs(t.slice(1, -1), { ...base, italics: true }, notasArchivo));
    } else if (m[4]) {
      salida.push(new TextRun({ text: t.slice(1, -1), ...base, font: MONO, size: Math.round((base.size || 22) * 0.9) }));
    } else if (m[5]) {
      const clave = t.slice(2, -1);
      const contenido = notasArchivo[clave];
      if (contenido === undefined) throw new Error(`Nota al pie sin definir: ${clave}`);
      const id = ++contadorNotas;
      notas[id] = {
        children: [new Paragraph({ spacing: { line: INTERLINEADO_SIMPLE }, children: runs(contenido, { size: 18 }, notasArchivo) })],
      };
      salida.push(new FootnoteReferenceRun(id));
    } else if (m[6]) {
      const mm = t.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      salida.push(new ExternalHyperlink({ link: mm[2], children: [new TextRun({ text: mm[1], ...base, style: "Hyperlink" })] }));
    }
    ultimo = m.index + t.length;
  }
  if (ultimo < texto.length) salida.push(new TextRun({ text: texto.slice(ultimo), ...base }));
  return salida;
}

// --- Imágenes -----------------------------------------------------------------------
function dimensionesPng(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function figura(ruta, leyenda, anchoCm) {
  const abs = path.resolve(RAIZ, ruta);
  const buf = fs.readFileSync(abs);
  const { w, h } = dimensionesPng(buf);
  const anchoPx = Math.round((anchoCm / 2.54) * 96);
  let altoPx = Math.round((anchoPx * h) / w);
  // Que ninguna figura supere ~19 cm de alto (la caja de texto mide 23,7 cm).
  const maxAlto = Math.round((16 / 2.54) * 96);
  let anchoFinal = anchoPx;
  if (altoPx > maxAlto) {
    anchoFinal = Math.round((anchoPx * maxAlto) / altoPx);
    altoPx = maxAlto;
  }
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0, line: INTERLINEADO_SIMPLE },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: anchoFinal, height: altoPx } })],
    }),
    leyendaParrafo("Figura", leyenda),
  ];
}

function leyendaParrafo(etiqueta, texto, antes = false) {
  return new Paragraph({
    style: "Caption",
    keepNext: antes,
    children: [new TextRun({ text: `${etiqueta} `, bold: true, italics: false }), new SequentialIdentifier(etiqueta),
      new TextRun({ text: ". ", bold: true, italics: false }), ...runs(texto)],
  });
}

// --- Tablas --------------------------------------------------------------------------
function celdas(linea) {
  return linea.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

function tabla(lineas, leyenda, notasArchivo) {
  const filas = lineas.filter((l, i) => i !== 1).map(celdas);
  const n = filas[0].length;
  // Ancho de cada columna proporcional al largo típico de su contenido.
  const largos = Array.from({ length: n }, (_, j) => {
    const vals = filas.map((f) => (f[j] || "").replace(/\*|`/g, "").length);
    return Math.min(Math.max(...vals, 4), 60);
  });
  const total = largos.reduce((a, b) => a + b, 0);
  const anchos = largos.map((l) => Math.floor((l / total) * ANCHO_TEXTO));
  anchos[n - 1] += ANCHO_TEXTO - anchos.reduce((a, b) => a + b, 0);
  const borde = { style: BorderStyle.SINGLE, size: 4, color: "8C8C8C" };
  const bordes = { top: borde, bottom: borde, left: borde, right: borde };
  const rows = filas.map((f, i) => new TableRow({
    tableHeader: i === 0,
    cantSplit: true,
    children: Array.from({ length: n }, (_, j) => new TableCell({
      width: { size: anchos[j], type: WidthType.DXA },
      borders: bordes,
      verticalAlign: VerticalAlign.CENTER,
      shading: i === 0 ? { type: ShadingType.CLEAR, fill: "DCE6F0", color: "auto" } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ style: "CeldaTabla", children: runs(f[j] || "", i === 0 ? { bold: true, size: 18 } : { size: 18 }, notasArchivo) })],
    })),
  }));
  const salida = [];
  if (leyenda) salida.push(leyendaParrafo("Tabla", leyenda, true));
  salida.push(new Table({ width: { size: ANCHO_TEXTO, type: WidthType.DXA }, columnWidths: anchos, rows }));
  salida.push(new Paragraph({ spacing: { after: 120, line: INTERLINEADO_SIMPLE }, children: [] }));
  return salida;
}

// --- Markdown -> bloques ------------------------------------------------------------------
function convertir(md, opciones = {}) {
  const lineas = md.replace(/\r\n/g, "\n").split("\n");
  // Primero se separan las definiciones de notas al pie ([^clave]: texto).
  const notasArchivo = {};
  const cuerpo = [];
  for (const l of lineas) {
    const m = l.match(/^\[\^([^\]]+)\]:\s*(.*)$/);
    if (m) notasArchivo[m[1]] = m[2];
    else cuerpo.push(l);
  }

  const bloques = [];
  let tituloCapitulo = null;
  let parrafo = [];
  let i = 0;
  let leyendaTablaPendiente = null;

  const cerrarParrafo = () => {
    if (!parrafo.length) return;
    const texto = parrafo.join(" ").trim();
    parrafo = [];
    if (!texto) return;
    bloques.push(new Paragraph({ style: opciones.biblio ? "Bibliografia" : "Cuerpo", children: runs(texto, {}, notasArchivo) }));
  };

  while (i < cuerpo.length) {
    const l = cuerpo[i];

    if (/^```/.test(l)) {
      cerrarParrafo();
      const codigo = [];
      i++;
      while (i < cuerpo.length && !/^```/.test(cuerpo[i])) codigo.push(cuerpo[i++]);
      i++;
      codigo.forEach((c, k) => bloques.push(new Paragraph({
        style: "Codigo",
        shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
        spacing: { before: k === 0 ? 120 : 0, after: k === codigo.length - 1 ? 200 : 0, line: INTERLINEADO_SIMPLE },
        children: [new TextRun({ text: c.length ? c : " ", font: MONO, size: 16 })],
      })));
      continue;
    }

    let m;
    if ((m = l.match(/^(#{1,3})\s+(.*)$/))) {
      cerrarParrafo();
      const nivel = m[1].length;
      const texto = m[2].trim();
      if (nivel === 1) tituloCapitulo = texto;
      bloques.push(new Paragraph({
        heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][nivel - 1],
        children: runs(texto),
      }));
      i++;
      continue;
    }

    if ((m = l.match(/^!\[([^\]]*)\]\(([^)]+)\)(\{[^}]*\})?\s*$/))) {
      cerrarParrafo();
      const ancho = m[3] && m[3].match(/ancho=([\d.]+)/);
      bloques.push(...figura(m[2], m[1], (ancho ? parseFloat(ancho[1]) : 14) * 0.88));
      i++;
      continue;
    }

    if ((m = l.match(/^Tabla:\s*(.*)$/))) {
      cerrarParrafo();
      leyendaTablaPendiente = m[1].replace(/\s*\{#tab:[\w-]+\}\s*$/, "");
      i++;
      continue;
    }

    if (/^\|/.test(l)) {
      cerrarParrafo();
      const t = [];
      while (i < cuerpo.length && /^\|/.test(cuerpo[i])) t.push(cuerpo[i++]);
      bloques.push(...tabla(t, leyendaTablaPendiente, notasArchivo));
      leyendaTablaPendiente = null;
      continue;
    }

    if ((m = l.match(/^(\s*)[-*]\s+(.*)$/))) {
      cerrarParrafo();
      const nivel = m[1].length >= 2 ? 1 : 0;
      let texto = m[2];
      i++;
      while (i < cuerpo.length && /^\s{2,}\S/.test(cuerpo[i]) && !/^\s*[-*]\s/.test(cuerpo[i])) texto += " " + cuerpo[i++].trim();
      bloques.push(new Paragraph({
        numbering: { reference: "viñetas", level: nivel },
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 40, line: 300 },
        children: runs(texto, {}, notasArchivo),
      }));
      continue;
    }

    if (/^\d+\.\s+/.test(l)) {
      cerrarParrafo();
      const ref = nuevaListaNumerada();
      while (i < cuerpo.length && /^\d+\.\s+/.test(cuerpo[i])) {
        let texto = cuerpo[i++].replace(/^\d+\.\s+/, "");
        while (i < cuerpo.length && /^\s{2,}\S/.test(cuerpo[i])) texto += " " + cuerpo[i++].trim();
        bloques.push(new Paragraph({
          numbering: { reference: ref, level: 0 },
          alignment: AlignmentType.JUSTIFIED,
          spacing: { after: 40, line: 300 },
          children: runs(texto, {}, notasArchivo),
        }));
      }
      continue;
    }

    if ((m = l.match(/^>\s?(.*)$/))) {
      cerrarParrafo();
      const cita = [];
      while (i < cuerpo.length && /^>/.test(cuerpo[i])) cita.push(cuerpo[i++].replace(/^>\s?/, ""));
      bloques.push(new Paragraph({ style: "Cita", children: runs(cita.join(" "), {}, notasArchivo) }));
      continue;
    }

    if (/^\s*$/.test(l)) {
      cerrarParrafo();
      i++;
      continue;
    }

    if (/^---\s*$/.test(l)) {
      cerrarParrafo();
      bloques.push(new Paragraph({ children: [new PageBreak()] }));
      i++;
      continue;
    }

    parrafo.push(l.trim());
    i++;
  }
  cerrarParrafo();
  return { bloques, tituloCapitulo };
}

// --- Cabeceras y pies --------------------------------------------------------------------
function cabecera(capitulo) {
  return new Header({
    children: [new Paragraph({
      spacing: { line: INTERLINEADO_SIMPLE },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "808080", space: 2 } },
      tabStops: [{ type: TabStopType.RIGHT, position: ANCHO_TEXTO }],
      children: [
        new TextRun({ text: capitulo, size: 17, italics: true }),
        new TextRun({ text: `\t${META.tituloCorto} - ${META.alumnosCorto}`, size: 17 }),
      ],
    })],
  });
}

function pie() {
  return new Footer({
    children: [new Paragraph({
      spacing: { line: INTERLINEADO_SIMPLE },
      border: { top: { style: BorderStyle.SINGLE, size: 6, color: "808080", space: 2 } },
      tabStops: [
        { type: TabStopType.CENTER, position: Math.round(ANCHO_TEXTO / 2) },
        { type: TabStopType.RIGHT, position: ANCHO_TEXTO },
      ],
      children: [
        new TextRun({ text: META.universidad, size: 17 }),
        new TextRun({ text: `\t${META.anio}`, size: 17 }),
        new TextRun({ children: ["\tPágina ", PageNumber.CURRENT, " de ", PageNumber.TOTAL_PAGES], size: 17 }),
      ],
    })],
  });
}

const PAGINA = {
  size: { width: 11906, height: 16838 },
  margin: { top: 3 * CM, bottom: 3 * CM, left: 4 * CM, right: 2 * CM, header: Math.round(1.5 * CM), footer: Math.round(1.5 * CM) },
};

function seccion(children, capitulo, conCabecera = true) {
  const vacia = new Header({ children: [new Paragraph({ children: [] })] });
  return {
    properties: { page: PAGINA, titlePage: true },
    headers: { default: conCabecera && capitulo ? cabecera(capitulo) : vacia, first: vacia },
    footers: { default: pie(), first: pie() },
    children,
  };
}

// --- Carátula -----------------------------------------------------------------------------
function caratula() {
  const centrado = (texto, size, extra = {}) => new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { line: 360, after: 120, ...(extra.spacing || {}) },
    children: [new TextRun({ text: texto, size, bold: extra.bold, font: FUENTE, allCaps: extra.caps })],
  });
  const derecha = (texto, opts = {}) => new Paragraph({
    alignment: AlignmentType.RIGHT, spacing: { line: 300, after: 60 },
    children: runs(texto, { size: 22, ...opts }),
  });
  const hoja1 = [
    centrado(META.universidad, 32, { bold: true, spacing: { before: 1 * CM } }),
    centrado(META.facultad, 26),
    centrado(META.carrera, 26),
    centrado("", 22, { spacing: { before: 3 * CM } }),
    centrado(META.tipo, 24, { caps: true }),
    centrado("", 22, { spacing: { before: 1.5 * CM } }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { line: 400 },
      children: [new TextRun({ text: META.titulo, size: 36, bold: true })],
    }),
  ];
  const hoja2 = [
    derecha(`**Alumnos:** ${META.alumnos.join(" y ")}`),
    derecha(`**Año:** ${META.anio}`),
    derecha(`**Profesor Coordinador:** ${META.coordinador}`),
    derecha(`**Profesor Tutor:** ${META.tutor}`),
    derecha(`**Profesor/a de Metodología:** ${META.metodologia}`),
  ];
  const vacia = new Header({ children: [new Paragraph({ children: [] })] });
  const pieVacio = new Footer({ children: [new Paragraph({ children: [] })] });
  // La segunda hoja va en su propia sección con alineación vertical inferior:
  // los datos tienen que quedar en el margen inferior derecho.
  return [
    { properties: { page: PAGINA }, headers: { default: vacia }, footers: { default: pieVacio }, children: hoja1 },
    { properties: { page: PAGINA, verticalAlign: VerticalAlign.BOTTOM }, headers: { default: vacia }, footers: { default: pieVacio }, children: hoja2 },
  ];
}

function portadilla(titulo, subtitulo) {
  return seccion([
    new Paragraph({ spacing: { before: 8 * CM }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: titulo, size: 40, bold: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, heading: HeadingLevel.HEADING_1, spacing: { before: 240 }, children: [new TextRun({ text: subtitulo })] }),
  ], null, false);
}

function indices() {
  const titulo = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
  const nota = new Paragraph({
    style: "Cuerpo",
    children: [new TextRun({ text: "(Si este índice aparece vacío, abrir el documento en Word y presionar F9 sobre él, o correr build/postproceso.ps1.)", italics: true, size: 18, color: "808080" })],
  });
  return [
    seccion([titulo("Índice general"), new TableOfContents("Índice general", { hyperlink: true, headingStyleRange: "1-2" }), nota], "Índice general", false),
    seccion([titulo("Índice de figuras"), new TableOfContents("Índice de figuras", { hyperlink: true, captionLabelIncludingNumbers: "Figura" })], "Índice de figuras", false),
    seccion([titulo("Índice de tablas"), new TableOfContents("Índice de tablas", { hyperlink: true, captionLabelIncludingNumbers: "Tabla" })], "Índice de tablas", false),
  ];
}

// --- Referencias cruzadas ------------------------------------------------------------------
// Las figuras y tablas se numeran en el orden en que aparecen en el documento,
// igual que lo hace Word con los campos SEQ de las leyendas. Esta pre-pasada
// recorre todos los capítulos en orden, asigna el número a cada etiqueta
// ({#fig:x} / {#tab:x}) y después @fig:x y @tab:x se reemplazan en el texto por
// "Figura N" / "Tabla N".
const REFS = {};
function numerarReferencias() {
  let nf = 0;
  let nt = 0;
  for (const item of ORDEN) {
    if (!item.archivo) continue;
    const ruta = path.join(CAPS, item.archivo);
    if (!fs.existsSync(ruta)) continue;
    for (const l of fs.readFileSync(ruta, "utf8").split(/\r?\n/)) {
      if (/^!\[/.test(l)) {
        nf++;
        const m = l.match(/#(fig:[\w-]+)/);
        if (m) REFS[m[1]] = `Figura ${nf}`;
      } else if (/^Tabla:/.test(l)) {
        nt++;
        const m = l.match(/#(tab:[\w-]+)/);
        if (m) REFS[m[1]] = `Tabla ${nt}`;
      }
    }
  }
}

function resolverReferencias(md, archivo) {
  return md.replace(/@((?:fig|tab):[\w-]+)/g, (_, clave) => {
    if (!REFS[clave]) throw new Error(`Referencia sin destino: ${clave} (en ${archivo})`);
    return REFS[clave];
  });
}

// --- Armado ------------------------------------------------------------------------------
function main() {
  numerarReferencias();
  const secciones = [...caratula()];
  let faltantes = 0;
  for (const item of ORDEN) {
    if (item.tipo === "indices") {
      secciones.push(...indices());
      continue;
    }
    if (item.tipo === "parte") {
      secciones.push(portadilla(item.titulo, item.subtitulo));
      continue;
    }
    if (item.tipo === "anexos") {
      secciones.push(portadilla("ANEXOS", "Apéndices del trabajo"));
      continue;
    }
    const ruta = path.join(CAPS, item.archivo);
    if (!fs.existsSync(ruta)) {
      console.warn(`  (falta ${item.archivo}, se omite)`);
      faltantes++;
      continue;
    }
    const md = resolverReferencias(fs.readFileSync(ruta, "utf8"), item.archivo);
    const { bloques, tituloCapitulo } = convertir(md, { biblio: !!item.biblio });
    secciones.push(seccion(bloques, tituloCapitulo || ""));
  }

  const doc = new Document({
    creator: META.alumnos.join(", "),
    title: META.titulo,
    description: META.tipo,
    features: { updateFields: true },
    styles: estilos,
    numbering: numeracion,
    footnotes: notas,
    sections: secciones,
  });

  fs.mkdirSync(SALIDA, { recursive: true });
  const destino = path.join(SALIDA, "Tesina-TraductorLSA.docx");
  return Packer.toBuffer(doc).then((buf) => {
    fs.writeFileSync(destino, buf);
    console.log(`OK -> ${destino}${faltantes ? ` (${faltantes} capítulos faltantes)` : ""}`);
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
