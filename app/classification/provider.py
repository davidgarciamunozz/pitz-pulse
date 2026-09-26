"""Minimal contract implemented by model providers and test fakes."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.classification.prompts import PromptDefinition


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0


@dataclass(frozen=True)
class ProviderResult:
    classification: str | Mapping[str, object]
    model: str
    latency_ms: float
    usage: TokenUsage | None = None


class ErrorKind(StrEnum):
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    SERVER = "server_error"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    REFUSAL = "refusal"
    INVALID_RESPONSE = "invalid_response"
    PROVIDER = "provider_error"


class ProviderError(RuntimeError):
    """Sanitized provider failure; response metadata can survive a refusal or truncation."""

    def __init__(
        self, kind: ErrorKind, *, retryable: bool, result: ProviderResult | None = None
    ) -> None:
        super().__init__(kind.value)
        self.kind = kind
        self.retryable = retryable
        self.result = result


class Classifier(Protocol):
    model: str

    async def classify(self, *, masked_message: str, prompt: PromptDefinition) -> ProviderResult:
        """Return semantic labels and confidence for an already-masked message."""
        ...
