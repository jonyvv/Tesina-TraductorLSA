# -*- coding: utf-8 -*-
"""
Tests de `origenes_permitidos`: la lista de orígenes CORS se arma desde la
variable de entorno LSA_CORS_ORIGINS (ver backend/app/main.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.app.main import origen_ws_permitido, origenes_permitidos  # noqa: E402


def test_sin_variable_acepta_cualquier_origen():
    assert origenes_permitidos(None) == ["*"]


def test_variable_vacia_es_como_no_tenerla():
    assert origenes_permitidos("   ") == ["*"]


def test_un_dominio():
    assert origenes_permitidos("https://traductor.onrender.com") == ["https://traductor.onrender.com"]


def test_varios_dominios_con_espacios_y_barra_final():
    """La barra final y los espacios son errores típicos al pegar la URL desde el
    navegador; el header Origin nunca la trae, así que con la barra no matchearía."""
    valor = " https://a.com/ ,https://b.com,, "
    assert origenes_permitidos(valor) == ["https://a.com", "https://b.com"]


# --- Origen del WebSocket -----------------------------------------------------
# CORSMiddleware no interviene en los WebSocket, así que /ws/translate —que es
# todo el sistema— necesita su propio chequeo. Ver origen_ws_permitido().


def test_ws_sin_restriccion_acepta_cualquier_origen():
    """Con LSA_CORS_ORIGINS sin setear, en local nada cambia."""
    assert origen_ws_permitido("https://cualquiera.com", ["*"]) is True


def test_ws_acepta_el_origen_configurado():
    assert origen_ws_permitido("https://traductor.onrender.com", ["https://traductor.onrender.com"]) is True


def test_ws_rechaza_un_origen_ajeno():
    """El caso que motiva el chequeo: otra página usando el modelo desde el
    navegador de un visitante, que CORS no impide."""
    assert origen_ws_permitido("https://sitio-ajeno.com", ["https://traductor.onrender.com"]) is False


def test_ws_tolera_la_barra_final():
    assert origen_ws_permitido("https://traductor.onrender.com/", ["https://traductor.onrender.com"]) is True


def test_ws_sin_header_origin_se_acepta():
    """Ningún navegador puede omitir Origin, así que aceptarlo no debilita la
    protección contra otros sitios; y es lo que mandan los clientes que no son
    navegadores (los tests, curl, un script)."""
    assert origen_ws_permitido(None, ["https://traductor.onrender.com"]) is True
