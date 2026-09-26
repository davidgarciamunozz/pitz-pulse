"""One schema-constrained OpenAI call; the application pipeline owns retries."""

from time import perf_counter

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.classification.prompts import PromptDefinition
from app.classification.provider import ErrorKind, ProviderError, ProviderResult, TokenUsage
from app.config import Settings
from app.schemas import ProviderClassification


class OpenAIClassifier:
    def __init__(self, settings: Settings, *, client: AsyncOpenAI | None = None) -> None:
        self.model = settings.model
        self._settings = settings
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()
            self._client = None

    async def classify(self, *, masked_message: str, prompt: PromptDefinition) -> ProviderResult:
        if self._client is None:
            key = self._settings.api_key
            if key is None or not key.get_secret_value().strip():
                raise ProviderError(ErrorKind.CONFIGURATION, retryable=False)
            self._client = AsyncOpenAI(
                api_key=key.get_secret_value(),
                max_retries=0,
                timeout=self._settings.timeout_seconds,
            )
        started = perf_counter()
        try:
            response = await self._client.with_options(
                max_retries=0, timeout=self._settings.timeout_seconds
            ).responses.create(
                model=self.model,
                temperature=0,
                store=False,
                instructions=prompt.content,
                input=[{"role": "user", "content": masked_message}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "pitz_classification",
                        "strict": True,
                        "schema": ProviderClassification.model_json_schema(),
                    }
                },
            )
        except APITimeoutError:
            raise ProviderError(ErrorKind.TIMEOUT, retryable=True) from None
        except APIConnectionError:
            raise ProviderError(ErrorKind.CONNECTION, retryable=True) from None
        except APIStatusError as error:
            status = error.status_code
            if status in (401, 403):
                kind, retryable = ErrorKind.AUTHENTICATION, False
            elif status == 429:
                kind, retryable = ErrorKind.RATE_LIMIT, True
            elif status in (408, 409) or status >= 500:
                kind, retryable = ErrorKind.SERVER, True
            elif status in (400, 404, 422):
                kind, retryable = ErrorKind.CONFIGURATION, False
            else:
                kind, retryable = ErrorKind.PROVIDER, False
            raise ProviderError(kind, retryable=retryable) from None
        latency_ms = (perf_counter() - started) * 1000
        usage = None
        if response.usage is not None:
            usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached_input_tokens=getattr(
                    response.usage.input_tokens_details, "cached_tokens", 0
                ),
            )
        result = ProviderResult(
            classification=response.output_text,
            model=response.model,
            latency_ms=latency_ms,
            usage=usage,
        )
        if any(
            part.type == "refusal"
            for item in response.output
            if item.type == "message"
            for part in item.content
        ):
            raise ProviderError(ErrorKind.REFUSAL, retryable=False, result=result)
        if response.status != "completed":
            raise ProviderError(ErrorKind.INVALID_RESPONSE, retryable=True, result=result)
        return result
