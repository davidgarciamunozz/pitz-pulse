"""Pure artifact and scoring checks; these tests never construct a provider."""

import json
from copy import deepcopy
from decimal import Decimal

import pytest

from app.classification.result import ClassificationResult, SuccessfulCallMetadata
from app.config import Settings
from app.schemas import Classification
from evaluation.artifacts import (
    EXPECTED_IDS,
    MESSAGES_PATH,
    REFERENCES_PATH,
    atomic_json,
    export_results,
    input_fingerprints,
    load_messages,
    load_references,
    new_run,
    sha256_file,
    verify_resume,
)
from evaluation.cli import main
from evaluation.scoring import compare_runs, score_outcomes, score_result_file, score_run


def complete_run():
    run = new_run(
        "synthetic-v1",
        Settings().prompt_version,
        MESSAGES_PATH.parent.parent / "prompts/v1.md",
        Settings(),
    )
    references = load_references()
    for outcome, reference in zip(run["outcomes"], references, strict=True):
        classification = Classification.model_validate(
            {"id": outcome["id"], **reference["expected"], "confianza": 0.8, "version_prompt": "v1"}
        )
        result = ClassificationResult(
            classification=classification,
            metadata=SuccessfulCallMetadata(
                model="test-model", attempt=1, latency_ms=1.0, prompt_version="v1"
            ),
        )
        outcome.update({"state": "succeeded", "result": result.model_dump(mode="json")})
    return run


def scoreable(run):
    return [
        {"id": item["id"], "state": "succeeded", "classification": item["result"]["classification"]}
        for item in run["outcomes"]
    ]


def synthetic_candidate(baseline):
    candidate = deepcopy(baseline)
    candidate["run_id"] = "synthetic-v2"
    candidate["prompt_version"] = "v2"
    candidate["inputs"]["prompt_sha256"] = "f" * 64
    for item in candidate["outcomes"]:
        item["result"]["classification"]["version_prompt"] = "v2"
        item["result"]["metadata"]["prompt_version"] = "v2"
    return candidate


def test_frozen_fixtures_load_with_exact_ids_and_valid_references():
    messages = load_messages()
    references = load_references()
    assert [item["id"] for item in messages] == EXPECTED_IDS
    assert [item["id"] for item in references] == EXPECTED_IDS
    assert all(item["justification"] for item in references)


def test_id_join_scores_only_five_discrete_fields_and_reports_mismatches():
    outcomes = scoreable(complete_run())
    outcomes[0]["classification"]["categoria"] = "consulta"
    outcomes[1]["classification"]["prioridad"] = "baja"
    outcomes[0]["classification"]["resumen"] = "Texto diferente que no se puntúa."
    report = score_outcomes(list(reversed(outcomes)), prompt_version="v1")
    assert report["fields"]["categoria"] == {"correct": 11, "total": 12, "accuracy": 11 / 12}
    assert report["fields"]["prioridad"]["correct"] == 11
    assert report["overall_exact_fields"] == {"correct": 58, "total": 60, "accuracy": 58 / 60}
    assert report["all_five_fields"]["correct"] == 10
    assert report["mismatches"][0]["fields"] == [
        {"field": "categoria", "expected": "bug", "actual": "consulta"}
    ]
    assert report["mismatches"][1]["fields"][0]["field"] == "prioridad"


def test_failed_and_missing_results_remain_in_every_denominator():
    outcomes = scoreable(complete_run())
    outcomes[0] = {"id": "MSG-01", "state": "failed", "failure_category": "rate_limit"}
    outcomes.pop(1)
    report = score_outcomes(outcomes)
    assert all(row["correct"] == 10 and row["total"] == 12 for row in report["fields"].values())
    assert report["overall_exact_fields"] == {"correct": 50, "total": 60, "accuracy": 50 / 60}
    assert report["failure_counts"] == {"failed": 1, "missing": 1}
    assert report["failure_category_counts"] == {"rate_limit": 1}
    assert [row["id"] for row in report["failures"]] == ["MSG-01", "MSG-02"]


@pytest.mark.parametrize("invalid", ["duplicate", "unexpected"])
def test_duplicate_and_unexpected_result_ids_are_rejected(invalid):
    outcomes = scoreable(complete_run())
    if invalid == "duplicate":
        outcomes.append(deepcopy(outcomes[0]))
    else:
        outcomes[0]["id"] = "MSG-99"
    with pytest.raises(ValueError):
        score_outcomes(outcomes)


def test_confidence_statistics_and_ordering_include_empty_group_nulls():
    outcomes = scoreable(complete_run())
    outcomes[0]["classification"]["categoria"] = "consulta"
    outcomes[0]["classification"]["confianza"] = 0.95
    outcomes[1]["classification"]["confianza"] = 0.1
    report = score_outcomes(outcomes)
    assert report["confidence"]["exact"]["count"] == 11
    assert report["confidence"]["exact"]["median"] == 0.8
    assert report["confidence"]["disagreement"] == {"count": 1, "mean": 0.95, "median": 0.95}
    assert report["confidence"]["lowest_successful"][0]["id"] == "MSG-02"
    assert report["confidence"]["highest_confidence_mistakes"][0]["id"] == "MSG-01"
    perfect = score_outcomes(scoreable(complete_run()))
    assert perfect["confidence"]["disagreement"] == {"count": 0, "mean": None, "median": None}


def test_export_has_exact_external_shape_and_refuses_partial_run(tmp_path):
    run = complete_run()
    output = tmp_path / "resultados.json"
    exported = export_results(run, output)
    assert json.loads(output.read_text(encoding="utf-8")) == exported
    assert [row["id"] for row in exported] == EXPECTED_IDS
    assert set(exported[0]) == {
        "id",
        "categoria",
        "prioridad",
        "area_sugerida",
        "idioma",
        "resumen",
        "requiere_info",
        "pregunta_seguimiento",
        "confianza",
        "version_prompt",
    }
    assert score_result_file(output)["overall_exact_fields"]["correct"] == 60
    run["outcomes"][0] = {"id": "MSG-01", "state": "failed", "failure_category": "timeout"}
    with pytest.raises(ValueError, match="twelve"):
        export_results(run, tmp_path / "partial.json")
    assert not (tmp_path / "partial.json").exists()


def test_offline_scoring_never_loads_settings_or_a_provider(tmp_path, monkeypatch, capsys):
    output = tmp_path / "resultados.json"
    export_results(complete_run(), output)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Offline scoring must not read settings or create a provider.")

    monkeypatch.setattr("evaluation.cli.Settings.from_env", forbidden)
    assert main(["score", "--results", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["scores"]["overall_exact_fields"]["correct"] == 60


def test_provenance_is_exact_and_secrets_never_enter_artifacts():
    prompt = MESSAGES_PATH.parent.parent / "prompts/v1.md"
    settings = Settings(
        api_key="synthetic-openai-secret",
        service_api_key="synthetic-service-secret",
        input_price_per_million=Decimal("1"),
        output_price_per_million=Decimal("2"),
    )
    run = new_run("provenance", "v1", prompt, settings)
    assert run["inputs"] == input_fingerprints(prompt)
    assert run["inputs"]["messages_sha256"] == sha256_file(MESSAGES_PATH)
    assert run["inputs"]["references_sha256"] == sha256_file(REFERENCES_PATH)
    assert run["execution"]["model"] == settings.model
    assert run["execution"]["input_price_per_million"] == "1"
    serialized = json.dumps(run)
    assert "synthetic-openai-secret" not in serialized
    assert "synthetic-service-secret" not in serialized
    verify_resume(run, "provenance", "v1", prompt, settings)
    with pytest.raises(ValueError):
        verify_resume(run, "provenance", "v1", prompt, Settings(model="changed"))


def test_scoring_rejects_fixture_hash_drift_without_rewriting_references():
    run = complete_run()
    run["inputs"]["references_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Frozen fixtures"):
        score_run(run)


def test_score_and_export_verify_original_prompt_bytes(tmp_path):
    run = complete_run()
    assert score_run(run)["overall_exact_fields"]["correct"] == 60
    assert len(export_results(run, tmp_path / "valid.json")) == 12
    run["inputs"]["prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Prompt file differs"):
        score_run(run)
    with pytest.raises(ValueError, match="Prompt file differs"):
        export_results(run, tmp_path / "rejected.json")
    assert not (tmp_path / "rejected.json").exists()


def test_atomic_checkpoint_preserves_previous_document_on_replace_failure(tmp_path, monkeypatch):
    output = tmp_path / "run.json"
    atomic_json(output, {"state": "before"})

    def replacement_fails(*_args):
        raise OSError("simulated failed replacement")

    monkeypatch.setattr("evaluation.artifacts.os.replace", replacement_fails)
    with pytest.raises(OSError):
        atomic_json(output, {"state": "after"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"state": "before"}
    assert list(tmp_path.iterdir()) == [output]


def test_compare_reports_corrections_and_regressions_without_a_v2_prompt(monkeypatch):
    baseline = complete_run()
    baseline["outcomes"][0]["result"]["classification"]["categoria"] = "consulta"
    candidate = synthetic_candidate(baseline)
    monkeypatch.setattr("evaluation.scoring.verify_prompt_fingerprint", lambda _run: None)
    candidate["outcomes"][0]["result"]["classification"]["categoria"] = "bug"
    candidate["outcomes"][1]["result"]["classification"]["prioridad"] = "baja"
    report = compare_runs(baseline, candidate)
    assert report["overall_exact_field_delta"] == 0
    assert report["corrected"] == [{"id": "MSG-01", "field": "categoria"}]
    assert report["regressions"] == [{"id": "MSG-02", "field": "prioridad"}]
    assert score_run(candidate)["all_five_fields"]["correct"] == 11


@pytest.mark.parametrize("invalid", ["missing", "failed", "duplicate", "unexpected"])
def test_compare_rejects_incomplete_or_invalid_candidate_without_v2(invalid, monkeypatch):
    baseline = complete_run()
    candidate = synthetic_candidate(baseline)
    monkeypatch.setattr("evaluation.scoring.verify_prompt_fingerprint", lambda _run: None)
    if invalid == "missing":
        candidate["outcomes"].pop()
    elif invalid == "failed":
        candidate["outcomes"][0] = {"id": "MSG-01", "state": "failed"}
    elif invalid == "duplicate":
        candidate["outcomes"][1]["id"] = "MSG-01"
    else:
        candidate["outcomes"][1]["id"] = "MSG-99"
    with pytest.raises(ValueError, match="complete|unique"):
        compare_runs(baseline, candidate)
