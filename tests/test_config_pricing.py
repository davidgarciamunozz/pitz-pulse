import os
from decimal import Decimal
from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic import ValidationError

from app.classification.pricing import ModelPricing, estimate_cost
from app.classification.provider import TokenUsage
from app.config import Settings


def test_local_dotenv_loads_without_mutating_process_environment(
    tmp_path, monkeypatch, caplog, capsys
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=synthetic-secret-for-test-only\n"
        "OPENAI_MODEL=local-model\n"
        "MODEL_TIMEOUT_SECONDS=9\n"
        "OPENAI_INPUT_PRICE_PER_MILLION=\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.config.PROJECT_ENV_FILE", env_file)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MODEL_TIMEOUT_SECONDS", raising=False)

    settings = Settings.from_env()

    assert settings.api_key.get_secret_value() == "synthetic-secret-for-test-only"
    assert settings.model == "local-model"
    assert settings.timeout_seconds == 9
    assert settings.input_price_per_million is None
    assert "OPENAI_API_KEY" not in os.environ
    assert "synthetic-secret-for-test-only" not in caplog.text
    captured = capsys.readouterr()
    assert "synthetic-secret-for-test-only" not in captured.out + captured.err
    assert "synthetic-secret-for-test-only" not in repr(settings)


def test_process_environment_takes_precedence_over_local_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_MODEL=local-model\nMODEL_TIMEOUT_SECONDS=9\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.config.PROJECT_ENV_FILE", env_file)
    monkeypatch.setenv("OPENAI_MODEL", "process-model")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "7")

    settings = Settings.from_env()

    assert settings.model == "process-model"
    assert settings.timeout_seconds == 7


def test_explicit_mapping_never_reads_local_dotenv(monkeypatch):
    def unexpected_read(*args, **kwargs):
        pytest.fail("Explicit mapping must not read a local .env file.")

    monkeypatch.setattr("app.config.dotenv_values", unexpected_read)
    monkeypatch.setenv("OPENAI_MODEL", "process-model")

    settings = Settings.from_env({"OPENAI_MODEL": "mapped-model"})

    assert settings.model == "mapped-model"
    assert Settings.from_env({}).api_key is None


def test_env_example_matches_supported_configuration_keys():
    example = Path(__file__).resolve().parents[1] / ".env.example"
    expected = {
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "PROMPT_VERSION",
        "MODEL_TIMEOUT_SECONDS",
        "MODEL_MAX_ATTEMPTS",
        "MODEL_CONCURRENCY_LIMIT",
        "MODEL_BACKOFF_BASE_SECONDS",
        "MODEL_BACKOFF_MAX_SECONDS",
        "OPENAI_INPUT_PRICE_PER_MILLION",
        "OPENAI_OUTPUT_PRICE_PER_MILLION",
        "OPENAI_CACHED_INPUT_PRICE_PER_MILLION",
    }
    assert set(dotenv_values(example)) == expected


def test_environment_configuration_and_secret_representation():
    settings = Settings.from_env(
        {
            "OPENAI_API_KEY": "test-secret-not-a-real-key",
            "OPENAI_MODEL": "configured-model",
            "MODEL_TIMEOUT_SECONDS": "7",
            "MODEL_MAX_ATTEMPTS": "2",
            "MODEL_CONCURRENCY_LIMIT": "4",
            "OPENAI_INPUT_PRICE_PER_MILLION": "1",
            "OPENAI_OUTPUT_PRICE_PER_MILLION": "2",
        }
    )
    assert settings.model == "configured-model"
    assert (settings.timeout_seconds, settings.max_attempts, settings.concurrency_limit) == (
        7,
        2,
        4,
    )
    assert settings.input_price_per_million == Decimal("1")
    assert "test-secret" not in repr(settings)
    assert "test-secret" not in settings.model_dump_json()
    assert Settings.from_env({}).api_key is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"max_attempts": 0},
        {"max_attempts": 6},
        {"concurrency_limit": 0},
        {"model": " "},
        {"input_price_per_million": "1"},
        {"input_price_per_million": "NaN", "output_price_per_million": "2"},
        {"backoff_base_seconds": 3, "backoff_max_seconds": 2},
    ],
)
def test_invalid_configuration_fails_early(overrides):
    with pytest.raises(ValidationError):
        Settings(**overrides)


def test_known_explicit_prices_include_cached_tokens():
    pricing = ModelPricing("test-model", Decimal("1"), Decimal("2"), Decimal("0.5"))
    assert estimate_cost("test-model", TokenUsage(100, 20, 40), pricing) == Decimal("0.00012")


@pytest.mark.parametrize(
    "model,usage,pricing",
    [
        ("unknown", TokenUsage(100, 20), ModelPricing("known", Decimal(1), Decimal(2))),
        ("test", TokenUsage(100, 20), None),
        ("test", None, ModelPricing("test", Decimal(1), Decimal(2))),
        ("test", TokenUsage(100, 20, 10), ModelPricing("test", Decimal(1), Decimal(2))),
        ("test", TokenUsage(10, 20, 100), ModelPricing("test", Decimal(1), Decimal(2))),
    ],
)
def test_unknown_or_inconsistent_cost_is_not_invented(model, usage, pricing):
    assert estimate_cost(model, usage, pricing) is None
