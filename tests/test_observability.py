import json
import logging

from app.observability import LOGGER_NAME, configure_json_logging, log_attempt


def test_logging_configuration_emits_json_without_duplicate_handlers(monkeypatch, capsys):
    logger = logging.getLogger(LOGGER_NAME)
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    monkeypatch.setattr(logger, "propagate", True)
    configure_json_logging()
    configure_json_logging()
    assert len(logger.handlers) == 1
    assert not logger.propagate
    log_attempt(
        message_id="request-1",
        model="test-model",
        prompt_version="v1",
        attempt=1,
        latency_ms=12.5,
        usage=None,
        cost=None,
        success=False,
        error_type="timeout",
    )
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "model_call"
    assert event["message_id"] == "request-1"
    assert event["error_type"] == "timeout"
    assert event["estimated_cost_usd"] is None
