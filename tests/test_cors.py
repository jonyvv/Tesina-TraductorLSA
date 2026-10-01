# -*- coding: utf-8 -*-
"""
Tests de `origenes_permitidos`: la lista de orígenes CORS se arma desde la
variable de entorno LSA_CORS_ORIGINS (ver backend/app/main.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.app.main import origenes_permitidos  # noqa: E402


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
