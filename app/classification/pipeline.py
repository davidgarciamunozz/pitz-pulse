"""The only application entry point from raw messages to a classifier."""

from pydantic import Field

from app.classification.provider import Classifier
from app.masking import mask_sensitive_data
from app.schemas import Classification, ClassificationContent


class ProviderClassification(ClassificationContent):
    """Provider-owned fields; message identity and prompt provenance stay in application code."""

    confianza: float = Field(ge=0, le=1)


class ClassificationPipeline:
    def __init__(self, classifier: Classifier) -> None:
        self._classifier = classifier

    async def classify(
        self, *, message_id: str, raw_message: str, prompt_version: str
    ) -> Classification:
        masked_message = mask_sensitive_data(raw_message)
        provider_output = await self._classifier.classify(
            masked_message=masked_message, prompt_version=prompt_version
        )
        validated = ProviderClassification.model_validate(provider_output)
        return Classification.model_validate(
            {"id": message_id, **validated.model_dump(), "version_prompt": prompt_version}
        )
