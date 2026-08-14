# -*- coding: utf-8 -*-
"""
Tests de backend/app/traductor_service.py (suavizado temporal y aislamiento
entre conexiones).

Usa dobles de prueba para `Preprocesador` y `ModeloLSE`: así se puede testear
toda la lógica de suavizado sin cámara, sin MediaPipe y sin modelo entrenado.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.app.prediccion import Prediccion  # noqa: E402
from backend.app.traductor_service import TraductorService  # noqa: E402
from common.features import FEATURE_VECTOR_LENGTH, FeatureExtractionResult  # noqa: E402

# Frame JPEG real y mínimo: `Frame.a_array()` lo decodifica con cv2.
FRAME_JPEG = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))[1].tobytes()


def _resultado_mediapipe_falso(hay_mano: bool):
    """Imita el HandLandmarkerResult que necesita `landmarks_para_overlay`."""
    if not hay_mano:
        return SimpleNamespace(hand_landmarks=[], handedness=[])
    puntos = [SimpleNamespace(x=i / 21, y=i / 21, z=0.0) for i in range(21)]
    return SimpleNamespace(
        hand_landmarks=[puntos],
        handedness=[[SimpleNamespace(category_name="Left", score=0.99)]],
    )


class PreprocesadorFalso:
    """Devuelve manos presentes/ausentes según se le indique."""

    def __init__(self):
        self.hay_mano = True

    def extraer_mano(self, _arr):
        return FeatureExtractionResult(
            vector=np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32),
            left_present=self.hay_mano,
            right_present=False,
            raw_result=_resultado_mediapipe_falso(self.hay_mano),
        )


class ModeloFalso:
    """Modelo estático: predice frame a frame (Random Forest / MLP)."""

    requiere_secuencia = False
    ventana_inferencia = 1

    def __init__(self, umbral=0.6):
        self.umbral_confianza = umbral
        self.etiqueta = "a"
        self.confianza = 0.9

    def predecir(self, _features):
        return Prediccion(etiqueta=self.etiqueta, confianza=self.confianza,
                          umbral=self.umbral_confianza)


class ModeloSecuenciaFalso(ModeloFalso):
    """Modelo dinámico: necesita una ventana de N frames (BiLSTM de LSA64)."""

    requiere_secuencia = True

    def __init__(self, umbral=0.6, ventana=4):
        super().__init__(umbral)
        self.ventana_inferencia = ventana
        self.ventanas_recibidas: list[np.ndarray] = []

    def predecir(self, _features):
        raise AssertionError("un modelo de secuencias no se llama frame a frame")

    def predecir_secuencia(self, secuencia):
        self.ventanas_recibidas.append(secuencia)
        return Prediccion(etiqueta=self.etiqueta, confianza=self.confianza,
                          umbral=self.umbral_confianza)


@pytest.fixture
def entorno():
    pre, mod = PreprocesadorFalso(), ModeloFalso()
    servicio = TraductorService(preprocesador=pre, modelo=mod, tam_buffer_suavizado=10)
    return servicio, pre, mod


def _alimentar(sesion, n):
    for _ in range(n):
        resultado = sesion.traducir(FRAME_JPEG)
    return resultado


class TestSuavizado:
    def test_no_arriesga_etiqueta_con_pocos_frames(self, entorno):
        servicio, _, _ = entorno
        sesion = servicio.nueva_sesion()
        assert _alimentar(sesion, 2)["valida"] is False

    def test_confirma_tras_frames_consistentes(self, entorno):
        servicio, _, _ = entorno
        sesion = servicio.nueva_sesion()
        resultado = _alimentar(sesion, 10)
        assert resultado["valida"] is True
        assert resultado["seña"] == "a"

    def test_el_buffer_decae_al_sacar_las_manos(self, entorno):
        """REGRESIÓN: antes solo se agregaban al buffer las predicciones válidas,
        así que nunca se vaciaba. Bajabas las manos y el backend seguía
        reportando la última seña con valida=true indefinidamente."""
        servicio, pre, _ = entorno
        sesion = servicio.nueva_sesion()

        assert _alimentar(sesion, 10)["valida"] is True

        pre.hay_mano = False  # el usuario baja las manos
        resultado = _alimentar(sesion, 10)

        assert resultado["valida"] is False
        assert resultado["seña"] is None

    def test_cambio_de_seña_se_refleja(self, entorno):
        servicio, _, mod = entorno
        sesion = servicio.nueva_sesion()
        assert _alimentar(sesion, 10)["seña"] == "a"

        mod.etiqueta = "b"
        assert _alimentar(sesion, 10)["seña"] == "b"

    def test_confianza_baja_del_modelo_no_confirma(self, entorno):
        servicio, _, mod = entorno
        mod.confianza = 0.3  # por debajo del umbral
        sesion = servicio.nueva_sesion()
        assert _alimentar(sesion, 10)["valida"] is False


class TestMetricas:
    def test_confianza_y_estabilidad_son_independientes(self, entorno):
        """Antes ambas viajaban como `confianza` y se comparaban contra el mismo
        umbral, aunque miden cosas distintas (seguridad del modelo vs.
        consistencia temporal)."""
        servicio, _, mod = entorno
        mod.confianza = 0.75
        sesion = servicio.nueva_sesion()
        resultado = _alimentar(sesion, 10)

        assert resultado["confianza"] == pytest.approx(0.75, abs=1e-3)
        assert resultado["estabilidad"] == pytest.approx(1.0, abs=1e-3)

    def test_estabilidad_refleja_deteccion_intermitente(self, entorno):
        servicio, pre, _ = entorno
        sesion = servicio.nueva_sesion()
        for i in range(10):
            pre.hay_mano = i % 2 == 0     # detección intermitente
            resultado = sesion.traducir(FRAME_JPEG)
        assert resultado["estabilidad"] == pytest.approx(0.5, abs=1e-6)


class TestContratoDeRespuesta:
    """El frontend depende de estos campos: dibuja el overlay con `landmarks` y
    calcula "red + codificación" restando `ms_servidor` al round-trip."""

    def test_campos_presentes(self, entorno):
        servicio, _, _ = entorno
        respuesta = servicio.nueva_sesion().traducir(FRAME_JPEG)
        assert set(respuesta) >= {
            "seña", "confianza", "estabilidad", "valida",
            "landmarks", "manos", "ms_servidor",
        }

    def test_landmarks_se_envian_aunque_no_haya_prediccion_valida(self, entorno):
        """El overlay tiene que dibujarse desde el primer frame, sin esperar a
        que el suavizado confirme una seña."""
        servicio, _, _ = entorno
        respuesta = servicio.nueva_sesion().traducir(FRAME_JPEG)
        assert respuesta["valida"] is False
        assert len(respuesta["landmarks"]) == 1
        assert len(respuesta["landmarks"][0]["puntos"]) == 21

    def test_sin_manos_no_hay_landmarks(self, entorno):
        servicio, pre, _ = entorno
        pre.hay_mano = False
        respuesta = servicio.nueva_sesion().traducir(FRAME_JPEG)
        assert respuesta["landmarks"] == []
        assert respuesta["manos"] == 0

    def test_cuenta_de_manos(self, entorno):
        servicio, _, _ = entorno
        assert servicio.nueva_sesion().traducir(FRAME_JPEG)["manos"] == 1

    def test_ms_servidor_es_positivo(self, entorno):
        servicio, _, _ = entorno
        respuesta = servicio.nueva_sesion().traducir(FRAME_JPEG)
        assert isinstance(respuesta["ms_servidor"], float)
        assert respuesta["ms_servidor"] >= 0


class TestAislamientoEntreConexiones:
    def test_dos_sesiones_no_comparten_buffer(self, entorno):
        """REGRESIÓN: el buffer vivía en el TraductorService compartido, así que
        dos clientes simultáneos se mezclaban las predicciones."""
        servicio, _, _ = entorno
        sesion_a, sesion_b = servicio.nueva_sesion(), servicio.nueva_sesion()

        assert _alimentar(sesion_a, 10)["valida"] is True
        # B recién arranca: no puede heredar la confirmación de A.
        assert sesion_b.traducir(FRAME_JPEG)["valida"] is False

    def test_historial_es_por_sesion(self, entorno):
        servicio, _, _ = entorno
        sesion_a, sesion_b = servicio.nueva_sesion(), servicio.nueva_sesion()
        _alimentar(sesion_a, 10)

        assert len(sesion_a.historial()) > 0
        assert sesion_b.historial() == []

    def test_el_servicio_no_guarda_estado_por_cliente(self, entorno):
        servicio, _, _ = entorno
        _alimentar(servicio.nueva_sesion(), 10)
        assert not hasattr(servicio, "_buffer")
        assert not hasattr(servicio, "_historial")


class TestModeloDinamico:
    """La BiLSTM de LSA64 no predice frame a frame: necesita una ventana de
    `ventana_inferencia` frames CON mano detectada."""

    @pytest.fixture
    def entorno_secuencia(self):
        pre, mod = PreprocesadorFalso(), ModeloSecuenciaFalso(ventana=4)
        servicio = TraductorService(preprocesador=pre, modelo=mod, tam_buffer_suavizado=10)
        return servicio, pre, mod

    def test_no_predice_hasta_llenar_la_ventana(self, entorno_secuencia):
        servicio, _, mod = entorno_secuencia
        sesion = servicio.nueva_sesion()

        for _ in range(3):                      # ventana = 4
            respuesta = sesion.traducir(FRAME_JPEG)

        assert mod.ventanas_recibidas == []
        assert respuesta["valida"] is False

    def test_la_ventana_llega_con_la_forma_correcta(self, entorno_secuencia):
        servicio, _, mod = entorno_secuencia
        _alimentar(servicio.nueva_sesion(), 4)

        assert len(mod.ventanas_recibidas) == 1
        assert mod.ventanas_recibidas[0].shape == (4, FEATURE_VECTOR_LENGTH)

    def test_el_overlay_no_espera_a_la_ventana(self, entorno_secuencia):
        """Los landmarks tienen que viajar desde el primer frame, aunque todavía
        no haya predicción posible: si no, el overlay tarda medio segundo en
        aparecer y parece que la cámara no anda."""
        servicio, _, _ = entorno_secuencia
        respuesta = servicio.nueva_sesion().traducir(FRAME_JPEG)

        assert respuesta["valida"] is False
        assert len(respuesta["landmarks"]) == 1

    def test_bajar_las_manos_corta_la_secuencia(self, entorno_secuencia):
        """Sin esto, la ventana pegaría el final de una seña con el principio de
        la siguiente y la BiLSTM clasificaría una secuencia que nunca ocurrió."""
        servicio, pre, mod = entorno_secuencia
        sesion = servicio.nueva_sesion()

        _alimentar(sesion, 4)
        assert len(mod.ventanas_recibidas) == 1

        pre.hay_mano = False
        sesion.traducir(FRAME_JPEG)             # corta la secuencia
        pre.hay_mano = True
        _alimentar(sesion, 3)                   # 3 < 4: todavía no alcanza

        assert len(mod.ventanas_recibidas) == 1, "la ventana no se reinició"

    def test_dos_sesiones_no_comparten_la_ventana(self, entorno_secuencia):
        servicio, _, mod = entorno_secuencia
        sesion_a, sesion_b = servicio.nueva_sesion(), servicio.nueva_sesion()

        _alimentar(sesion_a, 3)
        sesion_b.traducir(FRAME_JPEG)

        assert mod.ventanas_recibidas == [], "B completó la ventana de A"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
