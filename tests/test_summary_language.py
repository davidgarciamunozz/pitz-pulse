"""Synthetic, offline checks for the versioned statistical language policy."""

import socket

import pytest
from langdetect import DetectorFactory, LangDetectException

from app import summary_language


@pytest.mark.parametrize(
    ("summary", "accepted"),
    [
        ("El checkout está cobrando dos veces a algunos clientes.", True),
        ("O checkout está cobrando duas vezes alguns clientes.", False),
        ("The checkout is charging some customers twice.", False),
        ("La API devuelve errores al procesar algunos pagos.", True),
        ("HubSpot no está sincronizando correctamente los contactos.", True),
        ("El archivo de Excel contiene registros duplicados.", True),
        ("Salesforce no muestra los registros.", True),
        ("Error en checkout", False),
        ("Erro no checkout", False),
    ],
)
def test_synthetic_language_verdicts_are_content_only_and_offline(summary, accepted):
    assert summary_language.accepts_spanish_summary(summary) is accepted


def test_undetectable_summary_is_rejected_without_leaking_text(monkeypatch):
    def fail_detection(_summary):
        raise LangDetectException(0, "sensitive text from detector")

    monkeypatch.setattr(summary_language, "detect", fail_detection)
    assert summary_language.accepts_spanish_summary("sensitive text") is False


def test_detector_seed_and_policy_provenance_are_stable():
    assert DetectorFactory.seed == 0
    assert summary_language.POLICY_VERSION == "spanish-summary-v1"
    assert summary_language.DETECTOR_PACKAGE == "langdetect"
    assert summary_language.DETECTOR_VERSION == "1.0.9"
    summary = "Necesitamos acceso al panel para revisar la configuración."
    assert [summary_language.accepts_spanish_summary(summary) for _ in range(5)] == [True] * 5


def test_detection_does_not_need_network(monkeypatch):
    def no_network(*_args, **_kwargs):
        raise AssertionError("Language detection attempted network access")

    monkeypatch.setattr(socket, "create_connection", no_network)
    assert summary_language.accepts_spanish_summary(
        "El servicio presenta errores al procesar pagos."
    )
