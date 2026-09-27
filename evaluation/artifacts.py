"""Frozen fixture validation and durable, content-safe evaluation artifacts."""

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from app.classification.prompts import PROMPT_DIRECTORY, load_prompt
from app.classification.result import ClassificationResult
from app.schemas import Classification, ClassificationContent
from app.summary_language import DETECTOR_PACKAGE, DETECTOR_VERSION, POLICY_VERSION

ROOT = Path(__file__).resolve().parents[1]
MESSAGES_PATH = ROOT / "evaluation" / "messages.json"
REFERENCES_PATH = ROOT / "etiquetas_esperadas.json"
RUNS_DIR = ROOT / "evaluation" / "runs"
EXPECTED_IDS = [f"MSG-{number:02d}" for number in range(1, 13)]
ARTIFACT_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_messages(path: Path = MESSAGES_PATH) -> list[dict]:
    messages = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(messages, list)
        or not all(isinstance(item, dict) for item in messages)
        or [item.get("id") for item in messages] != EXPECTED_IDS
    ):
        raise ValueError("Messages must contain the twelve Annex IDs in order.")
    for item in messages:
        if set(item) != {"id", "source_area", "message"}:
            raise ValueError("Invalid message fixture fields.")
        if not all(isinstance(item[key], str) and item[key].strip() for key in item):
            raise ValueError("Message fixture values must be non-empty strings.")
    return messages


def load_references(path: Path = REFERENCES_PATH) -> list[dict]:
    references = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(references, list)
        or not all(isinstance(item, dict) for item in references)
        or [item.get("id") for item in references] != EXPECTED_IDS
    ):
        raise ValueError("References must contain the twelve Annex IDs in order.")
    for item in references:
        if set(item) != {"id", "expected", "justification"}:
            raise ValueError("Invalid reference fixture fields.")
        ClassificationContent.model_validate(item["expected"])
        if not isinstance(item["justification"], str) or not item["justification"].strip():
            raise ValueError("References require a justification.")
    return references


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("Run ID must contain only letters, digits, dots, underscores, or hyphens.")
    return run_id


def git_provenance() -> dict:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_revision": revision, "git_dirty": dirty}


def safe_settings(settings) -> dict:
    return {
        "model": settings.model,
        "temperature": 0,
        "timeout_seconds": settings.timeout_seconds,
        "max_attempts": settings.max_attempts,
        "concurrency_limit": settings.concurrency_limit,
        "backoff_base_seconds": settings.backoff_base_seconds,
        "backoff_max_seconds": settings.backoff_max_seconds,
        "input_price_per_million": str(settings.input_price_per_million)
        if settings.input_price_per_million is not None
        else None,
        "output_price_per_million": str(settings.output_price_per_million)
        if settings.output_price_per_million is not None
        else None,
        "cached_input_price_per_million": str(settings.cached_input_price_per_million)
        if settings.cached_input_price_per_million is not None
        else None,
        "summary_language_policy": {
            "id": POLICY_VERSION,
            "detector": DETECTOR_PACKAGE,
            "detector_version": DETECTOR_VERSION,
        },
    }


def input_fingerprints(prompt_path: Path) -> dict:
    return {
        "prompt_sha256": sha256_file(prompt_path),
        "messages_sha256": sha256_file(MESSAGES_PATH),
        "references_sha256": sha256_file(REFERENCES_PATH),
    }


def verify_prompt_fingerprint(run: dict) -> None:
    prompt = load_prompt(run["prompt_version"])
    prompt_path = PROMPT_DIRECTORY / f"{prompt.version}.md"
    if run["inputs"]["prompt_sha256"] != sha256_file(prompt_path):
        raise ValueError("Prompt file differs from the recorded evaluation run.")


def new_run(run_id: str, prompt_version: str, prompt_path: Path, settings) -> dict:
    return {
        "schema_version": ARTIFACT_VERSION,
        "run_id": validate_run_id(run_id),
        "created_at": utc_now(),
        "finished_at": None,
        **git_provenance(),
        "prompt_version": prompt_version,
        "inputs": input_fingerprints(prompt_path),
        "execution": safe_settings(settings),
        "outcomes": [{"id": message_id, "state": "pending"} for message_id in EXPECTED_IDS],
    }


def verify_resume(run: dict, run_id: str, prompt_version: str, prompt_path: Path, settings) -> None:
    if (
        run.get("schema_version") != ARTIFACT_VERSION
        or run.get("run_id") != run_id
        or run.get("prompt_version") != prompt_version
        or run.get("inputs") != input_fingerprints(prompt_path)
        or run.get("execution") != safe_settings(settings)
        or [item.get("id") for item in run.get("outcomes", [])] != EXPECTED_IDS
    ):
        raise ValueError("Run provenance differs; start a new named run.")
    for outcome in run["outcomes"]:
        if outcome.get("state") not in {
            "pending",
            "running",
            "succeeded",
            "failed",
            "interrupted_unknown",
        }:
            raise ValueError("Invalid checkpoint state.")
        if outcome["state"] == "succeeded":
            result = ClassificationResult.model_validate_json(json.dumps(outcome["result"]))
            if (
                result.classification.id != outcome["id"]
                or result.metadata.prompt_version != prompt_version
            ):
                raise ValueError("Checkpoint classification provenance differs.")


def atomic_json(path: Path, value: object) -> None:
    """Replace one complete JSON snapshot without exposing a partial checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_run(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def complete_classifications(run: dict) -> list[Classification]:
    verify_prompt_fingerprint(run)
    outcomes = run["outcomes"]
    if [outcome["id"] for outcome in outcomes] != EXPECTED_IDS or any(
        outcome["state"] != "succeeded" for outcome in outcomes
    ):
        raise ValueError("All twelve messages must succeed before export.")
    classifications = []
    for outcome in outcomes:
        result = ClassificationResult.model_validate_json(json.dumps(outcome["result"]))
        if (
            result.classification.id != outcome["id"]
            or result.metadata.prompt_version != run["prompt_version"]
        ):
            raise ValueError("Result identity or prompt provenance differs from its run.")
        classifications.append(result.classification)
    return classifications


def export_results(run: dict, output: Path) -> list[dict]:
    results = [item.model_dump(mode="json") for item in complete_classifications(run)]
    atomic_json(output, results)
    return results
