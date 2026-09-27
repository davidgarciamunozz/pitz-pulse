"""Thin FastAPI composition and sanitized HTTP error mapping."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.schemas import (
    Area,
    Category,
    CorrectionRequest,
    InProgressResponse,
    Priority,
    RequestDetail,
    RequestPage,
    RequestSummary,
    SubmitRequest,
)
from app.classification.openai_provider import OpenAIClassifier
from app.classification.pipeline import ClassificationPipeline
from app.classification.provider import Classifier
from app.config import Settings
from app.observability import configure_json_logging
from app.persistence.database import initialize_database
from app.persistence.models import (
    IdempotencyConflict,
    InProgress,
    InvalidCorrection,
    RequestNotCompleted,
    RequestNotFound,
    RequestState,
)
from app.persistence.repository import RequestRepository
from app.services.requests import RequestService


def create_app(
    *, settings: Settings | None = None, classifier: Classifier | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings if settings is not None else Settings.from_env()
        key = resolved.service_api_key
        if key is None or not key.get_secret_value().strip():
            raise RuntimeError("PITZ_API_KEY must be configured before starting the API.")
        await initialize_database(resolved.database_path)
        configure_json_logging()
        pipeline = ClassificationPipeline(
            classifier if classifier is not None else OpenAIClassifier(resolved), settings=resolved
        )
        app.state.settings = resolved
        app.state.pipeline = pipeline
        app.state.service = RequestService(RequestRepository(resolved.database_path), pipeline)
        try:
            yield
        finally:
            await pipeline.aclose()
            app.state.service = None
            app.state.pipeline = None

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_http_request(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Invalid request."})

    @app.middleware("http")
    async def sanitize_internal_failure(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception:
            # Consume internal exceptions before the ASGI server can log their text.
            return JSONResponse(status_code=503, content={"detail": "Service unavailable."})

    async def authenticate(
        request: Request, _api_key: str | None = Header(default=None, alias="X-API-Key")
    ) -> None:
        # Keep the header documented, but compare wire bytes to preserve UTF-8.
        configured = request.app.state.settings.service_api_key.get_secret_value().encode("utf-8")
        supplied = [
            value for name, value in request.scope["headers"] if name.lower() == b"x-api-key"
        ]
        if len(supplied) != 1:
            raise HTTPException(status_code=401, detail="Invalid API key.")
        try:
            candidate = supplied[0].decode("utf-8").encode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=401, detail="Invalid API key.") from None
        if not compare_digest(candidate, configured):
            raise HTTPException(status_code=401, detail="Invalid API key.")

    def service(request: Request) -> RequestService:
        return request.app.state.service

    auth = [Depends(authenticate)]
    ServiceDependency = Annotated[RequestService, Depends(service)]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/solicitudes", dependencies=auth, response_model=RequestSummary | InProgressResponse)
    async def submit(
        payload: SubmitRequest, response: Response, request_service: ServiceDependency
    ):
        try:
            submission = await request_service.submit_with_outcome(
                message_id=payload.id, raw_message=payload.mensaje
            )
        except IdempotencyConflict:
            raise HTTPException(
                status_code=409, detail="Request ID conflicts with an existing message."
            ) from None
        outcome = submission.outcome
        if isinstance(outcome, InProgress):
            response.status_code = 202
            response.headers["Retry-After"] = "1"
            return InProgressResponse(id=outcome.id, created_at=outcome.created_at)
        if outcome.state is RequestState.FAILED:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Classification unavailable.",
                    "id": outcome.id,
                    "state": "failed",
                },
            )
        response.status_code = 201 if submission.created else 200
        return RequestSummary.from_record(outcome)

    @app.get("/solicitudes", dependencies=auth, response_model=RequestPage)
    async def list_requests(
        request_service: ServiceDependency,
        categoria: Category | None = None,
        prioridad: Priority | None = None,
        area: Area | None = None,
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> RequestPage:
        records = await request_service.list_completed(
            category=categoria, priority=prioridad, area=area, limit=limit, offset=offset
        )
        return RequestPage(
            items=[RequestSummary.from_record(record) for record in records],
            limit=limit,
            offset=offset,
        )

    @app.get("/solicitudes/{message_id}", dependencies=auth, response_model=RequestDetail)
    async def get_request(message_id: str, request_service: ServiceDependency) -> RequestDetail:
        record = await request_service.get(message_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Request not found.")
        return RequestDetail.from_record(record)

    @app.patch("/solicitudes/{message_id}", dependencies=auth, response_model=RequestDetail)
    async def correct_request(
        message_id: str,
        payload: CorrectionRequest,
        request_service: ServiceDependency,
    ) -> RequestDetail:
        try:
            record = await request_service.correct(
                message_id, payload.model_dump(exclude_unset=True)
            )
        except RequestNotFound:
            raise HTTPException(status_code=404, detail="Request not found.") from None
        except RequestNotCompleted:
            raise HTTPException(status_code=409, detail="Request is not completed.") from None
        except (InvalidCorrection, ValidationError):
            raise HTTPException(status_code=422, detail="Invalid correction.") from None
        return RequestDetail.from_record(record)

    return app


app = create_app()
