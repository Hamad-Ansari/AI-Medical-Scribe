import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schema import Segment  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "aci-bench"


@pytest.fixture
def segments() -> list[Segment]:
    """A tiny synthetic consultation (invented for tests, no real person)."""
    rows = [
        ("doctor", "what brings you in today ?"),
        ("patient", "my right knee has been hurting for about two weeks ."),
        ("doctor", "any swelling ?"),
        ("patient", "yes , it swells after i walk ."),
        ("doctor", "on exam there is mild effusion of the right knee ."),
        ("doctor", "i think this is a knee sprain . let's get an x-ray and start ibuprofen 400 mg ."),
    ]
    return [Segment(id=i + 1, speaker=s, text=t) for i, (s, t) in enumerate(rows)]


@pytest.fixture
def valid_note_dict() -> dict:
    return {
        "subjective": [
            {"text": "Right knee pain for two weeks", "evidence": [2], "unsupported": False},
            {"text": "Swelling after walking", "evidence": [3, 4], "unsupported": False},
        ],
        "objective": [{"text": "Mild right knee effusion", "evidence": [5], "unsupported": False}],
        "assessment": [{"text": "Right knee sprain", "evidence": [2, 6], "unsupported": False}],
        "plan": [
            {"text": "Right knee x-ray", "evidence": [2, 6], "unsupported": False},
            {"text": "Ibuprofen 400 mg", "evidence": [6], "unsupported": False},
        ],
    }


class FakeLLM:
    """Scripted stand-in for OllamaClient; records every call."""

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def chat(self, messages, *, format=None, options=None):
        self.calls.append({"messages": [dict(m) for m in messages], "format": format, "options": options})
        if not self.outputs:
            raise AssertionError("FakeLLM called more times than scripted")
        return self.outputs.pop(0)


@pytest.fixture
def fake_llm_factory():
    return FakeLLM


@pytest.fixture
def valid_note_json(valid_note_dict) -> str:
    return json.dumps(valid_note_dict)


def has_aci_data() -> bool:
    return (DATA_DIR / "valid.csv").exists()
