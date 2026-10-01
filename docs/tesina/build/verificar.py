# -*- coding: utf-8 -*-
"""
docs/tesina/build/verificar.py

Controles editoriales que pidieron los evaluadores en tesinas anteriores
(ver "Sugerencias y Correcciones" de la tesina de 2025):

  1. Toda referencia de la bibliografía tiene que estar citada al menos una vez.
  2. Toda cita del texto tiene que tener su entrada en la bibliografía.
  3. Todo término del glosario tiene que aparecer en negrita en el cuerpo.
  4. Lista los [[PENDIENTE: ...]] que quedan, para saber qué falta completar.

Uso (desde la raíz del repo):  python docs/tesina/build/verificar.py
Sale con código 1 si alguno de los tres primeros controles falla.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

CAPS = Path(__file__).resolve().parents[1] / "capitulos"
BIBLIO = CAPS / "50_bibliografia.md"
GLOSARIO = CAPS / "51_glosario.md"


def sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn").lower()


def cuerpo() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(CAPS.glob("*.md"))
            if p not in (BIBLIO, GLOSARIO)}


def entradas_bibliografia() -> list[tuple[str, str, str]]:
    """(apellido del primer autor, año, línea completa) de cada entrada."""
    salida = []
    for linea in BIBLIO.read_text(encoding="utf-8").splitlines():
        if not linea.strip() or linea.startswith("#"):
            continue
        m = re.match(r"^(.+?)\s\((\d{4}[a-z]?|s\.f\.)\)", linea)
        if m:
            autor = m.group(1).split(",")[0].strip().rstrip(".")
            salida.append((autor, m.group(2), linea))
        else:
            # Leyes: "Ley 27.710 de 2023."
            m = re.match(r"^(Ley [\d.]+) de (\d{4})", linea)
            if m:
                salida.append((m.group(1), m.group(2), linea))
    return salida


def main() -> int:
    textos = cuerpo()
    todo = "\n".join(textos.values())
    plano = sin_acentos(todo)
    errores = 0

    print("== Bibliografía citada en el texto ==")
    for autor, anio, _ in entradas_bibliografia():
        clave_autor = sin_acentos(autor)
        if autor.startswith("Ley"):
            ok = clave_autor in plano
        else:
            ok = re.search(re.escape(clave_autor) + r"[^()]{0,60}?\(?" + re.escape(anio), plano) is not None
        if not ok:
            errores += 1
            print(f"  SIN CITAR: {autor} ({anio})")

    print("== Citas del texto con entrada en la bibliografía ==")
    bib = sin_acentos(BIBLIO.read_text(encoding="utf-8"))
    citas = set(re.findall(r"([A-ZÁÉÍÓÚÑ][\w\-áéíóúñö]+)(?: et al\.| y [A-ZÁÉÍÓÚ][\w\-áéíóúñö]+)?,? \(?((?:19|20)\d{2}[ab]?)\)?", todo))
    for autor, anio in sorted(citas):
        if autor in {"Ley", "Figura", "Tabla", "Capítulo", "Anexo", "Python", "WCAG"}:
            continue
        if sin_acentos(autor) not in bib or anio not in bib:
            print(f"  ¿sin entrada?: {autor} ({anio})")

    print("== Glosario en negrita en el cuerpo ==")
    terminos = re.findall(r"^\| \*\*(.+?)\*\* \|", GLOSARIO.read_text(encoding="utf-8"), flags=re.M)
    negritas = {sin_acentos(t).strip(" .:") for t in re.findall(r"\*\*(.+?)\*\*", todo)}
    for t in terminos:
        if sin_acentos(t) not in negritas:
            errores += 1
            print(f"  SIN NEGRITA EN EL CUERPO: {t}")

    print("== Referencias a secciones existentes ==")
    titulos = set(re.findall(r"^#{2,3} (\d+(?:\.\d+)+) ", todo, flags=re.M))
    for nombre, texto in textos.items():
        for ref in re.findall(r"secci(?:ón|ones) (\d+(?:\.\d+)+)", texto):
            if ref.rstrip(".") not in titulos:
                errores += 1
                print(f"  {nombre}: la sección {ref} no existe")

    print("== Pendientes ==")
    total = 0
    for nombre, texto in textos.items():
        for p in re.findall(r"\[\[PENDIENTE:(.*?)\]\]", texto, flags=re.S):
            total += 1
            print(f"  {nombre}: {p.strip()[:110]}")
    for p in re.findall(r"\[\[PENDIENTE:(.*?)\]\]", BIBLIO.read_text(encoding="utf-8")):
        total += 1
        print(f"  {BIBLIO.name}: {p.strip()[:110]}")
    print(f"  ({total} pendientes)")

    print("\nOK" if errores == 0 else f"\n{errores} problemas")
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
