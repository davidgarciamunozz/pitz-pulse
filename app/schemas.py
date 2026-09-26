"""Strict Pitz classification contracts with deterministic content checks."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClassificationContent(BaseModel):
    """Semantic labels shared by human references and model classifications."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    categoria: Literal["bug", "datos", "acceso", "automatizacion", "consulta", "otro"]
    prioridad: Literal["alta", "media", "baja"]
    area_sugerida: Literal[
        "backend", "frontend", "data", "devops", "producto", "digital_transformation"
    ]
    idioma: Literal["es", "pt"]
    resumen: str = Field(description="A Spanish summary of at most 20 whitespace-separated words.")
    requiere_info: bool
    pregunta_seguimiento: str | None

    @field_validator("resumen")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        """Check length without rewriting prose or attempting language detection."""
        word_count = len(value.split())
        if not 1 <= word_count <= 20:
            raise ValueError("Summary must contain between 1 and 20 words.")
        return value

    @model_validator(mode="after")
    def validate_follow_up(self) -> Self:
        if self.requiere_info:
            if self.pregunta_seguimiento is None or not self.pregunta_seguimiento.strip():
                raise ValueError(
                    "A non-empty follow-up question is required when information is missing."
                )
        elif self.pregunta_seguimiento is not None:
            raise ValueError("Follow-up question must be null when no information is missing.")
        return self


class Classification(ClassificationContent):
    """Complete external contract, including message identity and model metadata."""

    id: str
    confianza: float = Field(ge=0, le=1)
    version_prompt: str

    @field_validator("id", "version_prompt")
    @classmethod
    def validate_non_blank_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Identifier must not be blank.")
        return value
