"""Explicit model-specific token prices; unknown rates never imply free usage."""

from dataclasses import dataclass
from decimal import Decimal

from app.classification.provider import TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    model: str
    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal | None = None


def estimate_cost(
    model: str, usage: TokenUsage | None, pricing: ModelPricing | None
) -> Decimal | None:
    if pricing is None or usage is None or model != pricing.model:
        return None
    if (
        min(usage.input_tokens, usage.output_tokens, usage.cached_input_tokens) < 0
        or usage.cached_input_tokens > usage.input_tokens
    ):
        return None
    if usage.cached_input_tokens and pricing.cached_input_per_million is None:
        return None
    cached_rate = pricing.cached_input_per_million or Decimal(0)
    return (
        (usage.input_tokens - usage.cached_input_tokens) * pricing.input_per_million
        + usage.cached_input_tokens * cached_rate
        + usage.output_tokens * pricing.output_per_million
    ) / Decimal(1_000_000)
