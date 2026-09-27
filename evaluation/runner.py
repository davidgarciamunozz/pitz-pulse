"""Bounded Annex execution through the existing classification pipeline."""

import asyncio
import json
import logging
import os
from collections import deque
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from app.classification.openai_provider import OpenAIClassifier
from app.classification.pipeline import ClassificationPipeline
from app.classification.prompts import PROMPT_DIRECTORY, load_prompt
from app.classification.provider import Classifier, ErrorKind, ProviderError
from app.config import Settings
from app.observability import LOGGER_NAME, configure_json_logging
from evaluation.artifacts import (
    RUNS_DIR,
    atomic_json,
    load_messages,
    load_run,
    new_run,
    utc_now,
    validate_run_id,
    verify_resume,
)
from evaluation.scoring import score_run


def failure_category(error: Exception) -> str:
    if isinstance(error, ProviderError):
        return error.kind.value
    if isinstance(error, ValidationError):
        return "invalid_classification"
    if isinstance(error, TimeoutError):
        return "timeout"
    return "execution_error"


def usage_summary(events_path: Path, run: dict) -> dict:
    events = []
    reasons = []
    if events_path.exists():
        contents = events_path.read_text(encoding="utf-8")
        lines = contents.splitlines()
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                reason = (
                    "truncated_final_event"
                    if index == len(lines) - 1 and not contents.endswith("\n")
                    else "invalid_event"
                )
                reasons.append({"reason": reason})
                continue
            if not isinstance(event, dict) or event.get("event") != "model_call":
                reasons.append({"reason": "invalid_event"})
                continue
            events.append(event)

    outcomes = {item["id"]: item for item in run["outcomes"]}
    attempts = {message_id: {} for message_id in outcomes}
    known_events = []
    for event in events:
        message_id = event.get("message_id")
        attempt = event.get("attempt")
        if not isinstance(message_id, str) or message_id not in attempts:
            reasons.append({"reason": "foreign_message_event"})
        elif not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            reasons.append({"message_id": message_id, "reason": "invalid_attempt"})
        elif attempt in attempts[message_id]:
            reasons.append({"message_id": message_id, "reason": "duplicate_attempt"})
        else:
            attempts[message_id][attempt] = event
            known_events.append(event)

    for message_id, outcome in outcomes.items():
        sequence = attempts[message_id]
        state = outcome["state"]
        if state == "pending":
            if sequence:
                reasons.append({"message_id": message_id, "reason": "pending_has_attempts"})
            continue
        if state in {"running", "interrupted_unknown"}:
            reasons.append({"message_id": message_id, "reason": "unknown_outcome"})
            continue
        if not sequence:
            reasons.append({"message_id": message_id, "reason": "missing_attempt"})
            continue
        if state == "failed":
            # A failed outcome has no terminal attempt number in its metadata.
            reasons.append({"message_id": message_id, "reason": "failed_attempt_count_unknown"})
            continue
        metadata = outcome["result"]["metadata"]
        final_attempt = metadata["attempt"]
        max_attempts = run.get("execution", {}).get("max_attempts")
        if (
            not isinstance(final_attempt, int)
            or isinstance(final_attempt, bool)
            or final_attempt < 1
            or (max_attempts is not None and final_attempt > max_attempts)
            or set(sequence) != set(range(1, final_attempt + 1))
        ):
            reasons.append({"message_id": message_id, "reason": "incomplete_attempt_sequence"})
            continue
        if any(sequence[number].get("success") is not False for number in range(1, final_attempt)):
            reasons.append({"message_id": message_id, "reason": "inconsistent_attempt_result"})
        final_event = sequence[final_attempt]
        final_cost = final_event.get("estimated_cost_usd")
        metadata_cost = metadata.get("estimated_cost_usd")
        if (
            final_event.get("success") is not True
            or final_event.get("model") != metadata["model"]
            or final_event.get("prompt_version") != run["prompt_version"]
            or any(
                final_event.get(field) != metadata.get(field)
                for field in ("input_tokens", "output_tokens", "cached_input_tokens")
            )
            or (Decimal(str(final_cost)) if final_cost is not None else None)
            != (Decimal(str(metadata_cost)) if metadata_cost is not None else None)
        ):
            reasons.append({"message_id": message_id, "reason": "inconsistent_success_metadata"})
        if any(event.get("prompt_version") != run["prompt_version"] for event in sequence.values()):
            reasons.append({"message_id": message_id, "reason": "inconsistent_attempt_provenance"})

    def known_sum(field: str) -> float:
        return sum(event.get(field) or 0 for event in known_events)

    usage_known = all(
        event.get("input_tokens") is not None and event.get("output_tokens") is not None
        for event in known_events
    )
    cost_known = all(event.get("estimated_cost_usd") is not None for event in known_events)
    return {
        "observed_model_calls": len(events),
        "input_tokens_known_subtotal": known_sum("input_tokens"),
        "output_tokens_known_subtotal": known_sum("output_tokens"),
        "estimated_cost_usd_known_subtotal": known_sum("estimated_cost_usd"),
        "usage_complete": not reasons and usage_known,
        "cost_complete": not reasons and cost_known,
        "accounting_incomplete_reasons": reasons,
    }


def build_report(run: dict, run_dir: Path) -> dict:
    return {
        "run_id": run["run_id"],
        "prompt_version": run["prompt_version"],
        "configured_model": run["execution"]["model"],
        "actual_models": sorted(
            {
                row["result"]["metadata"]["model"]
                for row in run["outcomes"]
                if row["state"] == "succeeded"
            }
        ),
        "inputs": run["inputs"],
        "scores": score_run(run),
        "usage": usage_summary(run_dir / "model_calls.jsonl", run),
    }


async def evaluate(
    *,
    run_id: str,
    prompt_version: str,
    settings: Settings,
    classifier: Classifier | None = None,
    live: bool = False,
    resume: bool = False,
    runs_dir: Path = RUNS_DIR,
) -> dict:
    """Only an explicit live flag permits constructing the real provider."""
    if classifier is None and not live:
        raise ValueError("Real evaluation requires explicit --live intent.")
    validate_run_id(run_id)
    prompt = load_prompt(prompt_version)
    prompt_path = PROMPT_DIRECTORY / f"{prompt.version}.md"
    messages = load_messages()
    run_dir = runs_dir / run_id
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError("Named evaluation run does not exist.")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    lock_path = run_dir / ".run.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    pipeline = None
    log_handler = None
    run = None
    primary_error = None
    try:
        if resume:
            loaded = load_run(run_dir)
            verify_resume(loaded, run_id, prompt.version, prompt_path, settings)
            run = loaded
            for outcome in run["outcomes"]:
                if outcome["state"] == "running":
                    message_id = outcome["id"]
                    outcome.clear()
                    outcome.update({"id": message_id, "state": "interrupted_unknown"})
            run["finished_at"] = None
        else:
            run = new_run(run_id, prompt.version, prompt_path, settings)
        atomic_json(run_dir / "run.json", run)

        configure_json_logging()
        logger = logging.getLogger(LOGGER_NAME)
        log_handler = logging.FileHandler(run_dir / "model_calls.jsonl", encoding="utf-8")
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(log_handler)
        pipeline = ClassificationPipeline(
            classifier if classifier is not None else OpenAIClassifier(settings),
            prompt=prompt,
            settings=settings,
        )
        by_id = {message["id"]: message["message"] for message in messages}
        pending = deque(outcome for outcome in run["outcomes"] if outcome["state"] == "pending")
        checkpoint_lock = asyncio.Lock()
        stop_scheduling = False

        async def worker() -> None:
            nonlocal stop_scheduling
            while True:
                async with checkpoint_lock:
                    if stop_scheduling or not pending:
                        return
                    outcome = pending.popleft()
                    outcome["state"] = "running"
                    atomic_json(run_dir / "run.json", run)
                message_id = outcome["id"]
                try:
                    result = await pipeline.classify_with_metadata(
                        message_id=message_id, raw_message=by_id[message_id]
                    )
                except asyncio.CancelledError:
                    async with checkpoint_lock:
                        outcome.update({"state": "interrupted_unknown"})
                        atomic_json(run_dir / "run.json", run)
                    raise
                except Exception as error:
                    category = failure_category(error)
                    async with checkpoint_lock:
                        outcome.update({"state": "failed", "failure_category": category})
                        if isinstance(error, ProviderError) and error.kind in {
                            ErrorKind.AUTHENTICATION,
                            ErrorKind.CONFIGURATION,
                        }:
                            stop_scheduling = True
                        atomic_json(run_dir / "run.json", run)
                else:
                    async with checkpoint_lock:
                        outcome.update(
                            {"state": "succeeded", "result": result.model_dump(mode="json")}
                        )
                        atomic_json(run_dir / "run.json", run)

        workers = [asyncio.create_task(worker()) for _ in range(settings.concurrency_limit)]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        return run
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error = None
        try:
            if run is not None:
                run["finished_at"] = utc_now()
                atomic_json(run_dir / "run.json", run)
                atomic_json(run_dir / "report.json", build_report(run, run_dir))
        except BaseException as error:
            cleanup_error = error
        try:
            if log_handler is not None:
                logging.getLogger(LOGGER_NAME).removeHandler(log_handler)
                log_handler.close()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        try:
            if pipeline is not None:
                await pipeline.aclose()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        try:
            lock_path.unlink(missing_ok=True)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error
