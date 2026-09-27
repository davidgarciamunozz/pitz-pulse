"""Runtime policy provenance must not rewrite or revalidate frozen experiments."""

from copy import deepcopy

import pytest

from app.config import Settings
from evaluation.artifacts import (
    ROOT,
    load_run,
    new_run,
    sha256_file,
    verify_resume,
)
from evaluation.scoring import compare_runs, score_result_file, score_run


def test_historical_runs_still_score_without_revalidating_summary_language():
    v1 = load_run(ROOT / "evaluation/runs/v1-01")
    v2 = load_run(ROOT / "evaluation/runs/v2-02")
    assert score_run(v1)["overall_exact_fields"]["correct"] == 47
    assert score_run(v2)["overall_exact_fields"]["correct"] == 47
    assert score_result_file(ROOT / "resultados.json")["overall_exact_fields"]["correct"] == 47
    comparison = compare_runs(v1, v2)
    assert len(comparison["corrected"]) == 6
    assert len(comparison["regressions"]) == 6
    assert "summary_language_policy" not in v1["execution"]
    assert "summary_language_policy" not in v2["execution"]
    assert sha256_file(ROOT / "resultados.json") == (
        "20da4b89aacec623cb97ad7bfcc53e71ede1ef1b3fd23aa6235c2f24402d3cd5"
    )


def test_new_run_records_policy_and_resume_rejects_policy_drift():
    prompt = ROOT / "prompts/v1.md"
    settings = Settings()
    run = new_run("new-policy", "v1", prompt, settings)
    assert run["execution"]["summary_language_policy"] == {
        "id": "spanish-summary-v1",
        "detector": "langdetect",
        "detector_version": "1.0.9",
    }
    verify_resume(run, "new-policy", "v1", prompt, settings)
    drifted = deepcopy(run)
    drifted["execution"]["summary_language_policy"]["id"] = "older-policy"
    with pytest.raises(ValueError, match="provenance"):
        verify_resume(drifted, "new-policy", "v1", prompt, settings)


def test_old_and_new_execution_settings_cannot_be_compared_as_prompt_only():
    historical = load_run(ROOT / "evaluation/runs/v1-01")
    changed = deepcopy(historical)
    changed["prompt_version"] = "v2"
    changed["inputs"]["prompt_sha256"] = "f" * 64
    changed["execution"]["summary_language_policy"] = {
        "id": "spanish-summary-v1",
        "detector": "langdetect",
        "detector_version": "1.0.9",
    }
    with pytest.raises(ValueError, match="execution settings"):
        compare_runs(historical, changed)
