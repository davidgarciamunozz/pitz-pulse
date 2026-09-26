"""Content-free JSON events emitted through Python logging."""

import json
import logging
from decimal import Decimal

from app.classification.provider import TokenUsage

LOGGER_NAME = "pitz.model_calls"


def configure_json_logging() -> None:
    """Call from an application entry point; imports do not reconfigure logging."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_attempt(
    *,
    message_id: str,
    model: str,
    prompt_version: str,
    attempt: int,
    latency_ms: float,
    usage: TokenUsage | None,
    cost: Decimal | None,
    success: bool,
    error_type: str | None = None,
) -> None:
    event = {
        "event": "model_call",
        "message_id": message_id,
        "model": model,
        "prompt_version": prompt_version,
        "attempt": attempt,
        "latency_ms": round(latency_ms, 3),
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
        "cached_input_tokens": usage.cached_input_tokens if usage else None,
        "estimated_cost_usd": float(cost) if cost is not None else None,
        "success": success,
    }
    if error_type is not None:
        event["error_type"] = error_type
    logging.getLogger(LOGGER_NAME).info(json.dumps(event, allow_nan=False))
