"""The only application entry point from raw messages to a classifier."""

import asyncio
from collections.abc import Awaitable, Callable
from time import perf_counter

from pydantic import ValidationError

from app.classification.pricing import ModelPricing, estimate_cost
from app.classification.prompts import PromptDefinition, load_prompt
from app.classification.provider import Classifier, ProviderError
from app.config import Settings
from app.masking import mask_sensitive_data
from app.observability import configure_json_logging, log_attempt
from app.schemas import Classification, ProviderClassification


class ClassificationPipeline:
    @classmethod
    def from_env(cls) -> "ClassificationPipeline":
        from app.classification.openai_provider import OpenAIClassifier

        settings = Settings.from_env()
        configure_json_logging()
        return cls(OpenAIClassifier(settings), settings=settings)

    async def aclose(self) -> None:
        close = getattr(self._classifier, "aclose", None)
        if close is not None:
            await close()

    def __init__(
        self,
        classifier: Classifier,
        *,
        prompt: PromptDefinition | None = None,
        settings: Settings | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._classifier = classifier
        self._settings = settings or Settings()
        self._prompt = prompt or load_prompt(self._settings.prompt_version)
        self._limiter = asyncio.Semaphore(self._settings.concurrency_limit)
        self._sleep = sleep
        self._pricing = None
        if self._settings.input_price_per_million is not None:
            self._pricing = ModelPricing(
                model=self._settings.model,
                input_per_million=self._settings.input_price_per_million,
                output_per_million=self._settings.output_price_per_million,
                cached_input_per_million=self._settings.cached_input_price_per_million,
            )

    async def classify(self, *, message_id: str, raw_message: str) -> Classification:
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("Message ID must be a non-empty string.")
        masked_message = mask_sensitive_data(raw_message)
        for attempt in range(1, self._settings.max_attempts + 1):
            # Release the permit before backoff, including on validation/transport failure.
            async with self._limiter:
                result = None
                started = perf_counter()
                try:
                    async with asyncio.timeout(self._settings.timeout_seconds):
                        result = await self._classifier.classify(
                            masked_message=masked_message, prompt=self._prompt
                        )
                    content = result.classification
                    validated = (
                        ProviderClassification.model_validate_json(content)
                        if isinstance(content, str)
                        else ProviderClassification.model_validate(content)
                    )
                    classification = Classification.model_validate(
                        {
                            "id": message_id,
                            **validated.model_dump(),
                            "version_prompt": self._prompt.version,
                        }
                    )
                except asyncio.CancelledError:
                    log_attempt(
                        message_id=message_id,
                        model=self._classifier.model,
                        prompt_version=self._prompt.version,
                        attempt=attempt,
                        latency_ms=(perf_counter() - started) * 1000,
                        usage=None,
                        cost=None,
                        success=False,
                        error_type="cancelled",
                    )
                    raise
                except Exception as error:
                    if isinstance(error, ProviderError):
                        result = error.result
                        error_type, retryable = error.kind.value, error.retryable
                    elif isinstance(error, TimeoutError):
                        error_type, retryable = "timeout", True
                    elif isinstance(error, ValidationError):
                        error_type, retryable = "invalid_classification", True
                    else:
                        error_type, retryable = "unexpected_error", False
                    log_attempt(
                        message_id=message_id,
                        model=result.model if result else self._classifier.model,
                        prompt_version=self._prompt.version,
                        attempt=attempt,
                        latency_ms=result.latency_ms
                        if result
                        else (perf_counter() - started) * 1000,
                        usage=result.usage if result else None,
                        cost=estimate_cost(result.model, result.usage, self._pricing)
                        if result
                        else None,
                        success=False,
                        error_type=error_type,
                    )
                    if not retryable or attempt == self._settings.max_attempts:
                        raise
                else:
                    log_attempt(
                        message_id=message_id,
                        model=result.model,
                        prompt_version=self._prompt.version,
                        attempt=attempt,
                        latency_ms=result.latency_ms,
                        usage=result.usage,
                        cost=estimate_cost(result.model, result.usage, self._pricing),
                        success=True,
                    )
                    return classification
            delay = min(
                self._settings.backoff_base_seconds * 2 ** (attempt - 1),
                self._settings.backoff_max_seconds,
            )
            await self._sleep(delay)
        raise AssertionError("Validated attempt count must permit at least one attempt.")
