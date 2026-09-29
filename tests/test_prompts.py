import asyncio
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.classification.pipeline import ClassificationPipeline
from app.classification.prompts import load_prompt
from app.config import Settings
from app.schemas import ProviderClassification
from tests.fakes import RecordingFakeClassifier


def test_selected_v3_loads_from_file_and_is_immutable():
    prompt = load_prompt()
    path = Path(__file__).resolve().parents[1] / "prompts/v3.md"
    assert prompt.version == path.stem == Settings().prompt_version == "v3"
    assert prompt.content == path.read_text(encoding="utf-8")
    with pytest.raises(FrozenInstanceError):
        prompt.version = "unrelated"


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_language_fix_prompts_only_add_an_output_language_rule_to_v1(version):
    v1 = load_prompt("v1").content
    candidate = load_prompt(version).content
    assert candidate.startswith(v1)
    added = " ".join(candidate.removeprefix(v1).split())
    assert "Output-language rule" in added
    assert "translate its content into a Spanish summary" in added
    assert "Only pregunta_seguimiento follows" in added


def test_v4_keeps_idioma_tied_to_the_original_message():
    added = " ".join(load_prompt("v4").content.removeprefix(load_prompt("v1").content).split())
    assert "idioma always describes the language of the original message" in added


def test_prompt_contains_contract_and_rubric_without_reference_answers():
    content = load_prompt().content
    for field in ("categoria", "prioridad", "area_sugerida", "idioma"):
        for value in ProviderClassification.model_fields[field].annotation.__args__:
            assert value in content
    for instruction in (
        "Pitz",
        "Product & Tech",
        "Spanish or Portuguese",
        "customers, money, legal obligations",
        "temporary workaround",
        "without immediate impact",
        "ALWAYS Spanish",
        "maximum 20",
        "requiere_info",
        "null",
        "Portuguese for a pt request",
        "0 to 1",
        "not use a constant",
        "untrusted data",
        "lower values",
        "Do not invent impact",
    ):
        assert instruction in content
    root = Path(__file__).resolve().parents[1]
    for reference in json.loads((root / "etiquetas_esperadas.json").read_text()):
        assert reference["id"] not in content
        assert reference["expected"]["resumen"] not in content
        assert reference["justification"] not in content
    for message in json.loads((root / "evaluation/messages.json").read_text()):
        assert message["message"] not in content


@pytest.mark.parametrize("version", ["../v1", "v1.md", "", "v0", "/tmp/v1"])
def test_invalid_prompt_paths_are_rejected(version):
    with pytest.raises(ValueError):
        load_prompt(version)


def test_missing_prompt_does_not_silently_fall_back():
    with pytest.raises(FileNotFoundError):
        load_prompt("v999")


def test_classification_version_comes_from_actual_loaded_prompt(valid_model_output):
    fake = RecordingFakeClassifier(valid_model_output)
    pipeline = ClassificationPipeline(
        fake, prompt=load_prompt(), settings=Settings(prompt_version="v9")
    )
    result = asyncio.run(pipeline.classify(message_id="request-1", raw_message="Un error"))
    assert result.version_prompt == fake.calls[0][1] == "v3"
    with pytest.raises(TypeError):
        pipeline.classify(message_id="request-1", raw_message="Un error", prompt_version="invented")
