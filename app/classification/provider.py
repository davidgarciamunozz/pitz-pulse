"""Minimal contract implemented by model providers and test fakes."""

from collections.abc import Mapping
from typing import Protocol


class Classifier(Protocol):
    async def classify(self, *, masked_message: str, prompt_version: str) -> Mapping[str, object]:
        """Return semantic labels and confidence for an already-masked message."""
        ...
