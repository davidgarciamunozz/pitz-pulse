import json
from pathlib import Path

from app.schemas import ClassificationContent

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_assessment_fixtures_cover_the_same_twelve_unique_messages():
    messages = json.loads((PROJECT_ROOT / "evaluation/messages.json").read_text(encoding="utf-8"))
    references = json.loads((PROJECT_ROOT / "etiquetas_esperadas.json").read_text(encoding="utf-8"))
    expected_ids = [f"MSG-{number:02d}" for number in range(1, 13)]

    assert [message["id"] for message in messages] == expected_ids
    assert [reference["id"] for reference in references] == expected_ids

    for message in messages:
        assert set(message) == {"id", "source_area", "message"}
        assert isinstance(message["message"], str) and message["message"].strip()
        assert isinstance(message["source_area"], str) and message["source_area"].strip()

    for reference in references:
        assert set(reference) == {"id", "expected", "justification"}
        assert isinstance(reference["justification"], str) and reference["justification"].strip()
        # Human labels have no model confidence or prompt provenance to invent.
        content = ClassificationContent.model_validate(reference["expected"])
        assert content.model_dump() == reference["expected"]
