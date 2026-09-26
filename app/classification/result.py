"""Validated classification and successful provider-call metadata."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas import Classification


class SuccessfulCallMetadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)

    model: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    prompt_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_usage(self) -> "SuccessfulCallMetadata":
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("Input and output usage must be supplied together.")
        if self.cached_input_tokens is not None:
            if self.input_tokens is None or self.cached_input_tokens > self.input_tokens:
                raise ValueError("Cached usage must be included in input usage.")
        return self


class ClassificationResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    classification: Classification
    metadata: SuccessfulCallMetadata

    @model_validator(mode="after")
    def validate_prompt_provenance(self) -> "ClassificationResult":
        if self.classification.version_prompt != self.metadata.prompt_version:
            raise ValueError("Classification and provider metadata prompt versions differ.")
        return self
