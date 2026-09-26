"""Typed records and sanitized outcomes at the persistence boundary."""

from dataclasses import dataclass, field
from enum import StrEnum

from app.classification.result import SuccessfulCallMetadata
from app.schemas import Classification, ClassificationContent


class RequestState(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RequestRecord:
    id: str
    raw_message: str = field(repr=False)
    state: RequestState = RequestState.PROCESSING
    ai_classification: Classification | None = field(default=None, repr=False)
    human_correction: ClassificationContent | None = field(default=None, repr=False)
    provider_metadata: SuccessfulCallMetadata | None = None
    effective_category: str | None = None
    effective_priority: str | None = None
    effective_area: str | None = None
    failure_code: str | None = None
    created_at: str = ""
    updated_at: str = ""
    classified_at: str | None = None
    corrected_at: str | None = None

    @property
    def effective_classification(self) -> ClassificationContent | None:
        if self.human_correction is not None:
            return self.human_correction
        if self.ai_classification is not None:
            return ClassificationContent.model_validate(
                self.ai_classification.model_dump(include=set(ClassificationContent.model_fields))
            )
        return None


@dataclass(frozen=True)
class InProgress:
    id: str
    created_at: str


class IdempotencyConflict(Exception):
    """An existing request ID belongs to a different exact raw message."""


class RequestNotFound(Exception):
    """The requested ID has no reservation."""


class RequestNotCompleted(Exception):
    """Corrections require a completed original classification."""


class InvalidCorrection(Exception):
    """Only non-empty partial semantic corrections are accepted."""
