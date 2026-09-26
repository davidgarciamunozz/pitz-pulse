"""Prevent accidental network access throughout the automated test suite."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Network access is forbidden in automated tests.")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def valid_model_output():
    return {
        "categoria": "bug",
        "prioridad": "alta",
        "area_sugerida": "backend",
        "idioma": "es",
        "resumen": "Un cliente reporta un error en el procesamiento del pedido.",
        "requiere_info": True,
        "pregunta_seguimiento": "¿Qué pedido está afectado?",
        "confianza": 0.85,
    }
