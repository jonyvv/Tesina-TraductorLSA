# -*- coding: utf-8 -*-
"""
Tests de punta a punta del backend: arranque de la app y round-trip real por
WebSocket (frame JPEG -> respuesta JSON).

Se saltean solos si falta `common/models/hand_landmarker.task` (que no se
versiona: se baja con `python common/download_model.py`), así que un checkout
nuevo no falla por esto.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from common.features import DEFAULT_MODEL_PATH  # noqa: E402

pytest.importorskip("httpx2", reason="starlette.testclient necesita httpx2")

requiere_mediapipe = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="falta hand_landmarker.task (correr: python common/download_model.py)",
)

FRAME_JPEG = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))[1].tobytes()


@pytest.fixture(scope="module")
def cliente():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as c:
        yield c


class TestArranque:
    def test_health_responde(self, cliente):
        cuerpo = cliente.get("/health").json()
        assert cuerpo["status"] == "ok"
        # El backend tiene que arrancar aunque falte el modelo entrenado.
        assert set(cuerpo) >= {"preprocesador_listo", "modelo_cargado", "listo_para_traducir"}

    def test_health_es_coherente(self, cliente):
        """Si el modelo no está cargado, el backend NO puede decir que está listo.

        REGRESIÓN: `cargar()` dejaba el modelo asignado aunque la validación
        fallara, así que /health informaba `modelo_cargado: true` para un modelo
        que había sido rechazado."""
        cuerpo = cliente.get("/health").json()
        if not cuerpo["modelo_cargado"]:
            assert cuerpo["listo_para_traducir"] is False
            assert cuerpo["clases"] == []

    def test_model_info_responde(self, cliente):
        cuerpo = cliente.get("/model/info").json()
        assert set(cuerpo) >= {"clases", "umbral_confianza", "feature_version"}

    def test_model_info_sirve_el_esqueleto_de_la_mano(self, cliente):
        """El frontend pide las conexiones acá en vez de tener su propia copia,
        para que no puedan quedar desincronizadas con common/features.py."""
        from common.features import HAND_CONNECTIONS, NUM_LANDMARKS

        conexiones = cliente.get("/model/info").json()["conexiones_mano"]
        assert len(conexiones) == len(HAND_CONNECTIONS)
        for a, b in conexiones:
            assert 0 <= a < NUM_LANDMARKS and 0 <= b < NUM_LANDMARKS

    def test_sirve_los_archivos_del_frontend(self, cliente):
        for ruta, esperado in [("/styles.css", "--acento"), ("/app.js", "ESPEJADO_CANONICO")]:
            r = cliente.get(ruta)
            assert r.status_code == 200, ruta
            assert esperado in r.text, f"{ruta} no tiene el contenido esperado"

    def test_sirve_el_frontend(self, cliente):
        r = cliente.get("/")
        assert r.status_code == 200
        assert "Traductor" in r.text


@requiere_mediapipe
class TestWebSocket:
    def test_round_trip_de_un_frame(self, cliente):
        """Un frame entra como binario y sale una respuesta JSON. Sin modelo
        entrenado la respuesta es un error explícito — nunca una desconexión
        ni un crash del servidor."""
        with cliente.websocket_connect("/ws/translate") as ws:
            ws.send_bytes(FRAME_JPEG)
            data = ws.receive_json()

        assert isinstance(data, dict)
        if "error" not in data:
            assert set(data) >= {"seña", "confianza", "estabilidad", "valida"}

    def test_frame_corrupto_no_tumba_la_conexion(self, cliente):
        """Bytes que no son una imagen tienen que devolver un error y dejar la
        conexión viva para el frame siguiente."""
        with cliente.websocket_connect("/ws/translate") as ws:
            ws.send_bytes(b"esto no es un JPEG")
            primera = ws.receive_json()
            assert "error" in primera

            ws.send_bytes(FRAME_JPEG)
            assert isinstance(ws.receive_json(), dict)

    def test_dos_conexiones_no_se_pisan(self, cliente):
        with cliente.websocket_connect("/ws/translate") as ws_a:
            with cliente.websocket_connect("/ws/translate") as ws_b:
                ws_a.send_bytes(FRAME_JPEG)
                ws_b.send_bytes(FRAME_JPEG)
                assert isinstance(ws_a.receive_json(), dict)
                assert isinstance(ws_b.receive_json(), dict)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
