"""HTTP-only request and response contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.classification.result import SuccessfulCallMetadata
from app.persistence.models import RequestRecord, RequestState
from app.schemas import Classification, ClassificationContent

Category = Literal["bug", "datos", "acceso", "automatizacion", "consulta", "otro"]
Priority = Literal["alta", "media", "baja"]
Area = Literal["backend", "frontend", "data", "devops", "producto", "digital_transformation"]


class SubmitRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    mensaje: str

    @field_validator("id", "mensaje")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value must not be blank.")
        return value


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    categoria: Category | None = None
    prioridad: Priority | None = None
    area_sugerida: Area | None = None
    idioma: Literal["es", "pt"] | None = None
    resumen: str | None = None
    requiere_info: bool | None = None
    pregunta_seguimiento: str | None = None


class RequestSummary(BaseModel):
    id: str
    state: RequestState
    ai_classification: Classification | None
    human_correction: ClassificationContent | None
    effective_classification: ClassificationContent | None
    provider_metadata: SuccessfulCallMetadata | None
    created_at: str
    updated_at: str
    classified_at: str | None
    corrected_at: str | None

    @classmethod
    def from_record(cls, record: RequestRecord) -> "RequestSummary":
        return cls(
            id=record.id,
            state=record.state,
            ai_classification=record.ai_classification,
            human_correction=record.human_correction,
            effective_classification=record.effective_classification,
            provider_metadata=record.provider_metadata,
            created_at=record.created_at,
            updated_at=record.updated_at,
            classified_at=record.classified_at,
            corrected_at=record.corrected_at,
        )


class RequestDetail(RequestSummary):
    mensaje: str

    @classmethod
    def from_record(cls, record: RequestRecord) -> "RequestDetail":
        return cls(**RequestSummary.from_record(record).model_dump(), mensaje=record.raw_message)


class RequestPage(BaseModel):
    items: list[RequestSummary]
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class InProgressResponse(BaseModel):
    id: str
    state: Literal["processing"] = "processing"
    created_at: str
