"""Load an immutable prompt definition from its versioned file."""

import re
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIRECTORY = Path(__file__).resolve().parents[2] / "prompts"


@dataclass(frozen=True)
class PromptDefinition:
    version: str
    content: str


def load_prompt(version: str = "v3") -> PromptDefinition:
    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("Prompt version must use the form v1, v2, and so on.")
    path = PROMPT_DIRECTORY / f"{version}.md"
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError("Prompt content must not be empty.")
    return PromptDefinition(version=path.stem, content=content)
