"""Offline, exact scoring of discrete fields and descriptive confidence review."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from app.classification.result import ClassificationResult
from app.schemas import Classification
from evaluation.artifacts import (
    EXPECTED_IDS,
    MESSAGES_PATH,
    REFERENCES_PATH,
    load_references,
    sha256_file,
    verify_prompt_fingerprint,
)

SCORED_FIELDS = ("categoria", "prioridad", "area_sugerida", "idioma", "requiere_info")


def _unique_by_id(rows: list[dict], *, name: str) -> dict[str, dict]:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {name} IDs are not allowed.")
    unexpected = set(ids) - set(EXPECTED_IDS)
    if unexpected:
        raise ValueError(f"Unexpected {name} IDs are not allowed.")
    return {row["id"]: row for row in rows}


def _confidence_stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
    }


def score_outcomes(
    outcomes: list[dict], *, references: list[dict] | None = None, prompt_version: str | None = None
) -> dict:
    """Keep all twelve references in every denominator, including failures and omissions."""
    references = references if references is not None else load_references()
    reference_by_id = _unique_by_id(references, name="reference")
    if set(reference_by_id) != set(EXPECTED_IDS):
        raise ValueError("Exactly twelve reference IDs are required.")
    by_id = _unique_by_id(outcomes, name="outcome")
    correct = dict.fromkeys(SCORED_FIELDS, 0)
    mismatches = []
    failures = []
    exact_ids = []
    successful_confidence = []
    mistaken_confidence = []

    for message_id in EXPECTED_IDS:
        expected = reference_by_id[message_id]["expected"]
        outcome = by_id.get(message_id)
        if outcome is None or outcome.get("state") != "succeeded":
            state = outcome.get("state", "missing") if outcome else "missing"
            failure = {"id": message_id, "state": state}
            if outcome and state == "failed":
                failure["category"] = outcome.get("failure_category", "execution_error")
            failures.append(failure)
            mismatches.append({"id": message_id, "state": state, "fields": []})
            continue
        classification = Classification.model_validate(outcome["classification"])
        if classification.id != message_id:
            raise ValueError("Classification ID does not match its outcome ID.")
        if prompt_version is not None and classification.version_prompt != prompt_version:
            raise ValueError("Classification prompt version does not match its run.")
        differences = []
        for field in SCORED_FIELDS:
            actual = getattr(classification, field)
            if actual == expected[field]:
                correct[field] += 1
            else:
                differences.append({"field": field, "expected": expected[field], "actual": actual})
        pair = {"id": message_id, "confidence": classification.confianza}
        if differences:
            mismatches.append({"id": message_id, "state": "succeeded", "fields": differences})
            mistaken_confidence.append(pair)
        else:
            exact_ids.append(message_id)
        successful_confidence.append(pair)

    total_correct = sum(correct.values())
    exact_confidence = [
        row["confidence"] for row in successful_confidence if row["id"] in exact_ids
    ]
    mistaken_values = [row["confidence"] for row in mistaken_confidence]
    return {
        "fields": {
            field: {"correct": correct[field], "total": 12, "accuracy": correct[field] / 12}
            for field in SCORED_FIELDS
        },
        "overall_exact_fields": {
            "correct": total_correct,
            "total": 60,
            "accuracy": total_correct / 60,
        },
        "all_five_fields": {
            "correct": len(exact_ids),
            "total": 12,
            "accuracy": len(exact_ids) / 12,
        },
        "exact_message_ids": exact_ids,
        "mismatches": mismatches,
        "failures": failures,
        "failure_counts": dict(Counter(row["state"] for row in failures)),
        "failure_category_counts": dict(
            Counter(row["category"] for row in failures if "category" in row)
        ),
        "confidence": {
            "exact": _confidence_stats(exact_confidence),
            "disagreement": _confidence_stats(mistaken_values),
            "lowest_successful": sorted(
                successful_confidence, key=lambda row: (row["confidence"], row["id"])
            ),
            "highest_confidence_mistakes": sorted(
                mistaken_confidence, key=lambda row: (-row["confidence"], row["id"])
            ),
        },
    }


def score_run(run: dict) -> dict:
    verify_prompt_fingerprint(run)
    if run["inputs"]["messages_sha256"] != sha256_file(MESSAGES_PATH) or run["inputs"][
        "references_sha256"
    ] != sha256_file(REFERENCES_PATH):
        raise ValueError("Frozen fixtures differ from the recorded evaluation run.")
    outcomes = []
    for item in run["outcomes"]:
        if item["state"] == "succeeded":
            result = ClassificationResult.model_validate_json(json.dumps(item["result"]))
            outcomes.append(
                {
                    "id": item["id"],
                    "state": "succeeded",
                    "classification": result.classification.model_dump(mode="json"),
                }
            )
        else:
            outcomes.append(item)
    return score_outcomes(outcomes, prompt_version=run["prompt_version"])


def score_result_file(path: Path) -> dict:
    results = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise ValueError("Results must be a top-level JSON array.")
    versions = {Classification.model_validate(row).version_prompt for row in results}
    if len(versions) > 1:
        raise ValueError("All results must use the same prompt version.")
    return score_outcomes(
        [{"id": row["id"], "state": "succeeded", "classification": row} for row in results],
        prompt_version=next(iter(versions), None),
    )


def compare_runs(baseline: dict, candidate: dict) -> dict:
    if baseline["prompt_version"] == candidate["prompt_version"]:
        raise ValueError("Comparison requires different prompt versions.")
    if baseline["inputs"]["prompt_sha256"] == candidate["inputs"]["prompt_sha256"]:
        raise ValueError("Comparison requires a changed prompt file.")
    if (
        any(
            baseline["inputs"][name] != candidate["inputs"][name]
            for name in ("messages_sha256", "references_sha256")
        )
        or baseline["execution"] != candidate["execution"]
    ):
        raise ValueError("Comparison requires identical fixtures and execution settings.")
    for run in (baseline, candidate):
        if (
            len(run["outcomes"]) != len(EXPECTED_IDS)
            or {item["id"] for item in run["outcomes"]} != set(EXPECTED_IDS)
            or any(item["state"] != "succeeded" for item in run["outcomes"])
        ):
            raise ValueError("Comparison requires two complete successful runs.")
        for item in run["outcomes"]:
            result = ClassificationResult.model_validate_json(json.dumps(item["result"]))
            if result.classification.id != item["id"] or (
                result.classification.version_prompt != run["prompt_version"]
            ):
                raise ValueError("Comparison result provenance differs from its run.")
    models = [
        {item["result"]["metadata"]["model"] for item in run["outcomes"]}
        for run in (baseline, candidate)
    ]
    if len(models[0]) != 1 or models[0] != models[1]:
        raise ValueError("Comparison requires the same actual returned model.")
    baseline_score = score_run(baseline)
    candidate_score = score_run(candidate)
    baseline_errors = {
        (row["id"], field["field"])
        for row in baseline_score["mismatches"]
        for field in row["fields"]
    }
    candidate_errors = {
        (row["id"], field["field"])
        for row in candidate_score["mismatches"]
        for field in row["fields"]
    }
    return {
        "baseline": baseline["run_id"],
        "candidate": candidate["run_id"],
        "field_delta": {
            field: candidate_score["fields"][field]["correct"]
            - baseline_score["fields"][field]["correct"]
            for field in SCORED_FIELDS
        },
        "overall_exact_field_delta": candidate_score["overall_exact_fields"]["correct"]
        - baseline_score["overall_exact_fields"]["correct"],
        "corrected": [
            {"id": message_id, "field": field}
            for message_id, field in sorted(baseline_errors - candidate_errors)
        ],
        "regressions": [
            {"id": message_id, "field": field}
            for message_id, field in sorted(candidate_errors - baseline_errors)
        ],
    }
