"""Deterministic, recording classifier for tests; no network client exists here."""

from collections.abc import Mapping
from copy import deepcopy

from app.classification.prompts import PromptDefinition
from app.classification.provider import ProviderResult


class RecordingFakeClassifier:
    model = "test-model"

    def __init__(self, response: Mapping[str, object], *, failure: Exception | None = None) -> None:
        self._response = response
        self._failure = failure
        self.calls: list[tuple[str, str]] = []

    async def classify(self, *, masked_message: str, prompt: PromptDefinition) -> ProviderResult:
        self.calls.append((masked_message, prompt.version))
        if self._failure is not None:
            raise self._failure
        return ProviderResult(
            classification=deepcopy(self._response), model=self.model, latency_ms=0
        )
