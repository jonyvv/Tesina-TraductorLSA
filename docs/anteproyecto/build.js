// =============================================================================
// docs/anteproyecto/build.js
// -----------------------------------------------------------------------------
// Arma el anteproyecto corregido (.docx) a partir de anteproyecto.md, con el
// formato del anteproyecto aprobado: logo de la Universidad y "Lic. en
// Sistemas de Información" en la cabecera, Arial 11, títulos centrados y
// subrayados, número de página abajo a la derecha. Los componentes y su orden
// siguen la "Guía de Anteproyecto" de la facultad (1 a 12 + hoja de seguimiento).
//
// Uso (desde docs/anteproyecto):
//     node build.js            -> salida/Anteproyecto-TraductorLSA.docx
//     powershell -ExecutionPolicy Bypass -File ../tesina/build/postproceso.ps1 `
//         -Docx salida/Anteproyecto-TraductorLSA.docx -Entrega AnteproyectoProvenzanoVidela
//
// Usa la biblioteca docx instalada en docs/tesina/build (npm install allí).
// El Markdown es el mismo subconjunto que el de la tesina (docs/tesina/README.md),
// sin notas al pie ni marcas de pendiente.
// =============================================================================
"use strict";

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel, ImageRun, LevelFormat,
  Packer, PageBreak, PageNumber, Paragraph, SequentialIdentifier, ShadingType, Table, TableCell,
  TableOfContents, TableRow, TabStopType, TextRun, VerticalAlign, WidthType, HeightRule,
} = require(path.join(__dirname, "..", "tesina", "build", "node_modules", "docx"));

const RAIZ = __dirname;
const SALIDA = path.join(RAIZ, "salida");

const META = {
  titulo: "Sistema de traducción de Lengua de Señas Argentina (LSA) con inteligencia artificial",
  carrera: "Lic. en Sistemas de Información",
  alumnos: "VIDELA JONATAN - PROVENZANO LEANDRO",
  profesorTesina: "MARIO ALBERTO MIRALLAS",
  profesorTutor: "MARCELO PALMA",
  version: "Versión corregida · octubre de 2026",
};

// --- Página -------------------------------------------------------------------
const CM = 567; // twips por centímetro
const MARGEN = { top: Math.round(2.5 * CM), bottom: Math.round(2.5 * CM), left: Math.round(2.5 * CM), right: Math.round(2.5 * CM) };
const ANCHO_TEXTO = Math.round((21 - 5) * CM);
const FUENTE = "Arial";
const LINEA = 276; // 1,15

const estilos = {
  default: { document: { run: { font: FUENTE, size: 22 }, paragraph: { spacing: { line: LINEA } } } },
  paragraphStyles: [
    {
      id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 30, underline: {}, color: "000000" },
      paragraph: { alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 360, after: 200 }, outlineLevel: 0 },
    },
    {
      id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 24, bold: true, color: "000000" },
      paragraph: { keepNext: true, spacing: { before: 240, after: 120 }, outlineLevel: 1 },
    },
    {
      id: "Cuerpo", name: "Cuerpo", basedOn: "Normal", quickFormat: true,
      paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: LINEA } },
    },
    {
      id: "Caption", name: "caption", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 18, italics: true },
      paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240, line: 240 } },
    },
    {
      id: "CeldaTabla", name: "Celda de tabla", basedOn: "Normal",
      paragraph: { spacing: { line: 240, before: 20, after: 20 } },
    },
    {
      id: "Bibliografia", name: "Bibliografía", basedOn: "Normal",
      paragraph: { alignment: AlignmentType.LEFT, spacing: { after: 160, line: LINEA }, indent: { left: 709, hanging: 709 } },
    },
  ],
};

const numeracion = {
  config: [{
    reference: "vinetas",
    levels: [{ level: 0, format: LevelFormat.BULLET, text: "●", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
  }],
};
let contadorListas = 0;
function nuevaListaNumerada() {
  const ref = `num-${++contadorListas}`;
  numeracion.config.push({
    reference: ref,
    levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
  });
  return ref;
}

// --- Texto en línea -------------------------------------------------------------
function runs(texto, base = {}) {
  const salida = [];
  const patron = /(\*\*[^*]+\*\*)|(\*[^*\s][^*]*\*)|(`[^`]+`)/g;
  let ultimo = 0;
  let m;
  while ((m = patron.exec(texto)) !== null) {
    if (m.index > ultimo) salida.push(new TextRun({ text: texto.slice(ultimo, m.index), ...base }));
    const t = m[0];
    if (m[1]) salida.push(...runs(t.slice(2, -2), { ...base, bold: true }));
    else if (m[2]) salida.push(...runs(t.slice(1, -1), { ...base, italics: true }));
    else salida.push(new TextRun({ text: t.slice(1, -1), ...base, font: "Consolas" }));
    ultimo = m.index + t.length;
  }
  if (ultimo < texto.length) salida.push(new TextRun({ text: texto.slice(ultimo), ...base }));
  return salida;
}

// --- Referencias @fig:x / @tab:x ----------------------------------------------------
function numerarReferencias(md) {
  const numeros = {};
  let fig = 0;
  let tab = 0;
  for (const linea of md.split("\n")) {
    const f = linea.match(/^!\[.*\]\(.*\)\{.*#(fig:[\w-]+).*\}/);
    if (f) numeros[f[1]] = `Figura ${++fig}`;
    else if (/^!\[/.test(linea)) fig++;
    const t = linea.match(/^Tabla: .*\{#(tab:[\w-]+)\}/);
    if (t) numeros[t[1]] = `Tabla ${++tab}`;
    else if (/^Tabla: /.test(linea)) tab++;
  }
  return md.replace(/@((?:fig|tab):[\w-]+)/g, (_, clave) => {
    if (!numeros[clave]) throw new Error(`Referencia sin destino: ${clave}`);
    return numeros[clave];
  });
}

// --- Bloques --------------------------------------------------------------------------
function leyenda(etiqueta, texto, antes) {
  return new Paragraph({
    style: "Caption",
    keepNext: antes,
    children: [new TextRun({ text: `${etiqueta} `, bold: true, italics: false }), new SequentialIdentifier(etiqueta),
      new TextRun({ text: ". ", bold: true, italics: false }), ...runs(texto)],
  });
}

function figura(ruta, texto, anchoCm) {
  const buf = fs.readFileSync(path.resolve(RAIZ, ruta));
  const w = buf.readUInt32BE(16);
  const h = buf.readUInt32BE(20);
  let ancho = Math.round((anchoCm / 2.54) * 96);
  let alto = Math.round((ancho * h) / w);
  const maxAlto = Math.round((21 / 2.54) * 96);
  if (alto > maxAlto) {
    ancho = Math.round((ancho * maxAlto) / alto);
    alto = maxAlto;
  }
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0, line: 240 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: ancho, height: alto } })],
    }),
    leyenda("Figura", texto, false),
  ];
}

function celdas(linea) {
  return linea.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

const borde = { style: BorderStyle.SINGLE, size: 4, color: "8C8C8C" };
const bordes = { top: borde, bottom: borde, left: borde, right: borde };

function tabla(lineas, texto) {
  const filas = lineas.filter((_, i) => i !== 1).map(celdas);
  const n = filas[0].length;
  const largos = Array.from({ length: n }, (_, j) => Math.min(Math.max(...filas.map((f) => (f[j] || "").replace(/\*/g, "").length), 4), 60));
  const total = largos.reduce((a, b) => a + b, 0);
  const anchos = largos.map((l) => Math.floor((l / total) * ANCHO_TEXTO));
  anchos[n - 1] += ANCHO_TEXTO - anchos.reduce((a, b) => a + b, 0);
  const rows = filas.map((f, i) => new TableRow({
    tableHeader: i === 0,
    cantSplit: true,
    children: Array.from({ length: n }, (_, j) => new TableCell({
      width: { size: anchos[j], type: WidthType.DXA },
      borders: bordes,
      verticalAlign: VerticalAlign.CENTER,
      shading: i === 0 ? { type: ShadingType.CLEAR, fill: "DCE3F0", color: "auto" } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ style: "CeldaTabla", children: runs(f[j] || "", i === 0 ? { bold: true, size: 19 } : { size: 19 }) })],
    })),
  }));
  const salida = [];
  if (texto) salida.push(leyenda("Tabla", texto, true));
  salida.push(new Table({ width: { size: ANCHO_TEXTO, type: WidthType.DXA }, columnWidths: anchos, rows }));
  salida.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
  return salida;
}

function convertir(md) {
  const lineas = md.replace(/\r\n/g, "\n").split("\n");
  const bloques = [];
  let enBibliografia = false;
  let refLista = null;
  let i = 0;
  while (i < lineas.length) {
    const l = lineas[i];
    if (!/^\d+\. /.test(l)) refLista = null;
    if (!l.trim()) { i++; continue; }
    let m;
    if ((m = l.match(/^# (.+)$/))) {
      enBibliografia = /^11\. /.test(m[1]);
      // Cronograma y anexos arrancan en hoja nueva.
      const hojaNueva = /^(9|12)\. /.test(m[1]);
      bloques.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: hojaNueva, children: [new TextRun(m[1].toUpperCase())] }));
      i++;
    } else if ((m = l.match(/^## (.+)$/))) {
      const anexo = /^Anexo [B-E] /.test(m[1]);
      bloques.push(new Paragraph({ heading: HeadingLevel.HEADING_2, pageBreakBefore: anexo, children: [new TextRun(m[1])] }));
      i++;
    } else if ((m = l.match(/^!\[(.*)\]\((.*)\)\{ancho=([\d.]+).*\}$/))) {
      bloques.push(...figura(m[2], m[1], parseFloat(m[3])));
      i++;
    } else if ((m = l.match(/^Tabla: (.*?)(?: \{#tab:[\w-]+\})?$/))) {
      const filas = [];
      i++;
      while (i < lineas.length && !lineas[i].trim()) i++;
      while (i < lineas.length && lineas[i].trim().startsWith("|")) filas.push(lineas[i++]);
      bloques.push(...tabla(filas, m[1]));
    } else if (l.trim().startsWith("|")) {
      const filas = [];
      while (i < lineas.length && lineas[i].trim().startsWith("|")) filas.push(lineas[i++]);
      bloques.push(...tabla(filas, null));
    } else if ((m = l.match(/^- (.+)$/))) {
      bloques.push(new Paragraph({ style: "Cuerpo", numbering: { reference: "vinetas", level: 0 }, spacing: { after: 80 }, children: runs(m[1]) }));
      i++;
    } else if ((m = l.match(/^\d+\. (.+)$/))) {
      if (!refLista) refLista = nuevaListaNumerada();
      bloques.push(new Paragraph({ style: "Cuerpo", numbering: { reference: refLista, level: 0 }, spacing: { after: 80 }, children: runs(m[1]) }));
      i++;
    } else {
      // Párrafo: junta líneas consecutivas.
      const partes = [];
      while (i < lineas.length && lineas[i].trim() && !/^(#|- |\d+\. |\||!\[|Tabla: )/.test(lineas[i])) partes.push(lineas[i++].trim());
      const texto = partes.join(" ");
      const soloTitulo = /^\*\*[^*]+\*\*$/.test(texto);
      const estilo = enBibliografia && !soloTitulo ? "Bibliografia" : "Cuerpo";
      // Un título en negrita o un párrafo que presenta una lista no queda solo al pie de la hoja.
      bloques.push(new Paragraph({ style: estilo, keepNext: soloTitulo || texto.endsWith(":"), children: runs(texto) }));
    }
  }
  return bloques;
}

// --- Cabecera, pie, carátula, índice, hoja de seguimiento ---------------------------------
const logo = fs.readFileSync(path.join(RAIZ, "figuras", "logo_uch.png"));

function cabecera() {
  const altoLogo = Math.round((1.6 / 2.54) * 96);
  const anchoLogo = Math.round((altoLogo * logo.readUInt32BE(16)) / logo.readUInt32BE(20));
  return new Header({
    children: [new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: ANCHO_TEXTO }],
      children: [
        new ImageRun({ type: "png", data: logo, transformation: { width: anchoLogo, height: altoLogo } }),
        new TextRun({ text: `\t${META.carrera}`, bold: true, size: 22 }),
      ],
    })],
  });
}

function pie() {
  return new Footer({
    children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ children: [PageNumber.CURRENT] })] })],
  });
}

function caratula() {
  const derecha = (etiqueta, valor) => new Paragraph({
    alignment: AlignmentType.RIGHT, spacing: { after: 0 },
    children: [new TextRun({ text: `${etiqueta} `, bold: true }), new TextRun(valor)],
  });
  return [
    new Paragraph({ spacing: { before: Math.round(6 * CM) }, children: [] }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { line: 360 },
      children: [new TextRun({ text: META.titulo, italics: true, size: 52 })],
    }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 480 }, children: [new TextRun({ text: "Anteproyecto de tesina", size: 26 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: META.version, size: 20, color: "555555" })] }),
    new Paragraph({ spacing: { before: Math.round(7 * CM) }, children: [] }),
    derecha("ALUMNOS:", META.alumnos),
    derecha("PROFESOR TESINA:", META.profesorTesina),
    derecha("PROFESOR TUTOR:", META.profesorTutor),
  ];
}

function indice() {
  return [
    new Paragraph({ pageBreakBefore: true, alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "ÍNDICE", size: 30, underline: {} })] }),
    new TableOfContents("Índice", { hyperlink: true, headingStyleRange: "1-2" }),
  ];
}

function hojaSeguimiento() {
  const anchos = [Math.round(ANCHO_TEXTO * 0.18), Math.round(ANCHO_TEXTO * 0.57)];
  anchos.push(ANCHO_TEXTO - anchos[0] - anchos[1]);
  const fila = (textos, encabezado) => new TableRow({
    tableHeader: encabezado,
    height: encabezado ? undefined : { value: Math.round(1.25 * CM), rule: HeightRule.ATLEAST },
    children: textos.map((t, j) => new TableCell({
      width: { size: anchos[j], type: WidthType.DXA },
      borders: bordes,
      verticalAlign: VerticalAlign.CENTER,
      shading: encabezado ? { type: ShadingType.CLEAR, fill: "DCE3F0", color: "auto" } : undefined,
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: t, bold: encabezado })] })],
    })),
  });
  return [
    new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("HOJA DE SEGUIMIENTO")] }),
    new Paragraph({ style: "Cuerpo", children: runs("Registro de las devoluciones del profesor hasta la aprobación del anteproyecto.") }),
    new Table({
      width: { size: ANCHO_TEXTO, type: WidthType.DXA },
      columnWidths: anchos,
      rows: [fila(["Fecha", "Comentario", "Firma"], true), ...Array.from({ length: 12 }, () => fila(["", "", ""], false))],
    }),
  ];
}

// --- Principal -------------------------------------------------------------------------
// Sin argumentos arma el anteproyecto. Con la ruta de otro Markdown (p. ej. el
// informe de avance) arma un documento simple con la misma cabecera y el mismo
// formato, sin carátula, índice ni hoja de seguimiento, al lado del original.
function main() {
  const otro = process.argv[2] ? path.resolve(process.argv[2]) : null;
  const md = numerarReferencias(fs.readFileSync(otro || path.join(RAIZ, "anteproyecto.md"), "utf8"));
  const pagina = { size: { width: 11906, height: 16838 }, margin: MARGEN };
  const children = otro
    ? convertir(md)
    : [...caratula(), ...indice(), new Paragraph({ children: [new PageBreak()] }), ...convertir(md), ...hojaSeguimiento()];
  const doc = new Document({
    creator: "Leandro Provenzano, Jonatan Videla",
    title: otro ? md.match(/^# (.+)$/m)[1] : META.titulo,
    description: otro ? "" : "Anteproyecto de tesina",
    features: { updateFields: true },
    styles: estilos,
    numbering: numeracion,
    sections: [{
      properties: { page: pagina, titlePage: !otro },
      headers: { default: cabecera(), first: cabecera() },
      footers: { default: pie(), first: new Footer({ children: [new Paragraph({ children: [] })] }) },
      children,
    }],
  });
  const carpeta = otro ? path.dirname(otro) : SALIDA;
  fs.mkdirSync(carpeta, { recursive: true });
  const destino = otro ? otro.replace(/\.md$/, ".docx") : path.join(SALIDA, "Anteproyecto-TraductorLSA.docx");
  return Packer.toBuffer(doc).then((buf) => {
    fs.writeFileSync(destino, buf);
    console.log("Generado:", destino);
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
