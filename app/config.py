"""Environment configuration without clients, network access, or import-time secrets."""

import os
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    api_key: SecretStr | None = Field(default=None, repr=False)
    service_api_key: SecretStr | None = Field(default=None, repr=False)
    model: str = Field(default="gpt-4.1-mini-2025-04-14", min_length=1)
    prompt_version: str = "v3"
    timeout_seconds: float = Field(default=15, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=5)
    concurrency_limit: int = Field(default=3, ge=1)
    backoff_base_seconds: float = Field(default=0.5, ge=0)
    backoff_max_seconds: float = Field(default=8, ge=0)
    input_price_per_million: Decimal | None = Field(default=None, ge=0)
    output_price_per_million: Decimal | None = Field(default=None, ge=0)
    cached_input_price_per_million: Decimal | None = Field(default=None, ge=0)
    database_path: Path = Path("data/pitz-pulse.sqlite3")

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        if not self.model.strip():
            raise ValueError("Model must not be blank.")
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ValueError("Maximum backoff must be at least the initial backoff.")
        if (self.input_price_per_million is None) != (self.output_price_per_million is None):
            raise ValueError("Input and output prices must be configured together.")
        if self.cached_input_price_per_million is not None and self.input_price_per_million is None:
            raise ValueError("Cached input pricing also requires input and output prices.")
        return self

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        # Explicit mappings stay isolated from local developer configuration.
        if environ is None:
            source = {
                **{key: value for key, value in dotenv_values(PROJECT_ENV_FILE).items() if value},
                **os.environ,
            }
        else:
            source = environ
        names = {
            "OPENAI_API_KEY": "api_key",
            "PITZ_API_KEY": "service_api_key",
            "OPENAI_MODEL": "model",
            "PROMPT_VERSION": "prompt_version",
            "MODEL_TIMEOUT_SECONDS": "timeout_seconds",
            "MODEL_MAX_ATTEMPTS": "max_attempts",
            "MODEL_CONCURRENCY_LIMIT": "concurrency_limit",
            "MODEL_BACKOFF_BASE_SECONDS": "backoff_base_seconds",
            "MODEL_BACKOFF_MAX_SECONDS": "backoff_max_seconds",
            "OPENAI_INPUT_PRICE_PER_MILLION": "input_price_per_million",
            "OPENAI_OUTPUT_PRICE_PER_MILLION": "output_price_per_million",
            "OPENAI_CACHED_INPUT_PRICE_PER_MILLION": "cached_input_price_per_million",
            "DATABASE_PATH": "database_path",
        }
        return cls.model_validate(
            {field: source[name] for name, field in names.items() if source.get(name)}
        )
