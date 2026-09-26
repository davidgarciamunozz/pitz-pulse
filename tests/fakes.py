"""Deterministic, recording classifier for tests; no network client exists here."""

from collections.abc import Mapping
from copy import deepcopy


class RecordingFakeClassifier:
    def __init__(self, response: Mapping[str, object], *, failure: Exception | None = None) -> None:
        self._response = response
        self._failure = failure
        self.calls: list[tuple[str, str]] = []

    async def classify(self, *, masked_message: str, prompt_version: str) -> Mapping[str, object]:
        self.calls.append((masked_message, prompt_version))
        if self._failure is not None:
            raise self._failure
        return deepcopy(self._response)
