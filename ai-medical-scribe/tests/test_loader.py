"""ACI-Bench loader tests (phase 1)."""
import pytest

from app.loader import load_split, note_to_soap, parse_dialogue

from .conftest import has_aci_data

DIALOGUE = """[doctor] hi , how are you ?
[patient] i have had a cough for three days .
[patient_guest] he also had a fever .
continued wrapped line .
[doctor] okay . i will order a chest x-ray ."""


def test_parse_dialogue_basic():
    segs = parse_dialogue(DIALOGUE, max_words=0)
    assert [s.id for s in segs] == [1, 2, 3, 4]
    assert [s.speaker for s in segs] == ["doctor", "patient", "patient_guest", "doctor"]
    assert segs[2].text.endswith("continued wrapped line .")  # untagged line appended
    assert all(s.start is None for s in segs)


def test_long_turn_split_at_sentences():
    long_turn = "[doctor] " + " ".join(f"sentence number {i} is here ." for i in range(20))
    segs = parse_dialogue(long_turn, max_words=20)
    assert len(segs) > 1
    assert all(s.speaker == "doctor" for s in segs)
    assert all(s.text.endswith(".") for s in segs)  # never cut mid-sentence
    assert " ".join(s.text for s in segs) == long_turn.removeprefix("[doctor] ")


def test_parse_dialogue_empty_raises():
    with pytest.raises(ValueError, match="no '\\[speaker\\] text' turns"):
        parse_dialogue("just text with no tags")


NOTE = """CHIEF COMPLAINT

Cough.

HISTORY OF PRESENT ILLNESS

Cough for 3 days with fever.

PHYSICAL EXAM

Lungs clear.

ASSESSMENT AND PLAN

Acute bronchitis.
• Medical Reasoning: Likely viral.
• Additional Testing: Chest x-ray.
• Medical Treatment: Rest and fluids.

Patient Agreements: The patient agrees with the plan.

INSTRUCTIONS

Return in 1 week.
"""


def test_note_to_soap_mapping_and_ap_split():
    raw, soap = note_to_soap(NOTE)
    assert set(raw) >= {"CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "PHYSICAL EXAM", "ASSESSMENT AND PLAN"}
    assert "Cough." in soap["subjective"] and "fever" in soap["subjective"]
    assert soap["objective"] == "Lungs clear."
    assert "Acute bronchitis." in soap["assessment"] and "Likely viral" in soap["assessment"]
    assert "Chest x-ray" not in soap["assessment"]
    for frag in ("Chest x-ray", "Rest and fluids", "Patient Agreements", "Return in 1 week"):
        assert frag in soap["plan"]


@pytest.mark.skipif(not has_aci_data(), reason="ACI-Bench CSVs not downloaded")
def test_real_valid_split_loads():
    cases = load_split("valid")
    assert len(cases) == 20
    for c in cases:
        assert c.segments and c.segments[0].id == 1
        ids = [s.id for s in c.segments]
        assert ids == list(range(1, len(ids) + 1))
        # every reference note yields subjective + assessment + plan text
        assert c.reference_soap["subjective"]
        assert c.reference_soap["assessment"] or c.reference_soap["plan"]


@pytest.mark.skipif(not has_aci_data(), reason="ACI-Bench CSVs not downloaded")
def test_unknown_split():
    with pytest.raises(ValueError, match="unknown split"):
        load_split("nope")


def test_unknown_subheader_inherits_enclosing_section():
    note = "PHYSICAL EXAM\n\nGeneral: well.\n\nMSK\n\nTender right knee.\n\nPLAN\n\nIce."
    _, soap = note_to_soap(note)
    assert "Tender right knee." in soap["objective"]
    assert soap["plan"] == "Ice."
