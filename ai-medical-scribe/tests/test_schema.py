"""Schema validation tests (phase 1)."""
import pytest
from pydantic import ValidationError

from app.schema import SECTIONS, NoteItem, Segment, SOAPNote, segments_to_prompt, soap_json_schema


def test_valid_note_parses(valid_note_dict):
    note = SOAPNote.model_validate(valid_note_dict)
    assert note.n_items() == 6
    assert [s for s, _, _ in note.items()][:2] == ["subjective", "subjective"]


@pytest.mark.parametrize("missing", SECTIONS)
def test_missing_section_rejected(valid_note_dict, missing):
    valid_note_dict.pop(missing)
    with pytest.raises(ValidationError):
        SOAPNote.model_validate(valid_note_dict)


def test_extra_top_level_key_rejected(valid_note_dict):
    valid_note_dict["diagnosis_codes"] = []
    with pytest.raises(ValidationError):
        SOAPNote.model_validate(valid_note_dict)


def test_extra_item_key_rejected(valid_note_dict):
    valid_note_dict["plan"][0]["confidence"] = 0.9
    with pytest.raises(ValidationError):
        SOAPNote.model_validate(valid_note_dict)


@pytest.mark.parametrize("bad", [["two"], "2", [1.5], None])
def test_bad_evidence_types_rejected(bad):
    with pytest.raises(ValidationError):
        NoteItem.model_validate({"text": "x", "evidence": bad, "unsupported": False})


@pytest.mark.parametrize("text", ["", "   ", "x" * 301])
def test_bad_text_rejected(text):
    with pytest.raises(ValidationError):
        NoteItem(text=text, evidence=[1])


def test_unsupported_must_be_bool():
    with pytest.raises(ValidationError):
        NoteItem.model_validate({"text": "x", "evidence": [], "unsupported": "maybe"})


def test_empty_evidence_is_valid_but_left_for_verifier():
    # The schema accepts it; phase-2 verify.py flags it. Rejecting here would hide it.
    it = NoteItem(text="Patient smokes", evidence=[])
    assert it.evidence == [] and it.unsupported is False


def test_evidence_deduplicated_and_text_stripped():
    it = NoteItem(text="  Knee pain ", evidence=[3, 2, 3])
    assert it.text == "Knee pain" and it.evidence == [3, 2]


def test_empty_sections_allowed():
    note = SOAPNote(subjective=[], objective=[], assessment=[], plan=[])
    assert note.n_items() == 0


def test_json_schema_is_flat_and_complete():
    sch = soap_json_schema()
    assert "$defs" not in sch and "$ref" not in str(sch)
    assert set(sch["required"]) == set(SECTIONS)
    item = sch["properties"]["plan"]["items"]
    assert set(item["properties"]) == {"text", "evidence", "unsupported"}


def test_segment_validation():
    with pytest.raises(ValidationError):
        Segment(id=0, speaker="doctor", text="hi")  # ids are 1-based
    with pytest.raises(ValidationError):
        Segment(id=1, speaker="doctor", text="   ")
    with pytest.raises(ValidationError):
        Segment(id=1, speaker="doctor", text="hi", start=-1.0)


def test_segments_to_prompt(segments):
    out = segments_to_prompt(segments)
    assert out.splitlines()[0] == "[1] DOCTOR: what brings you in today ?"
    assert len(out.splitlines()) == len(segments)


def test_segments_to_prompt_empty():
    with pytest.raises(ValueError, match="no segments"):
        segments_to_prompt([])


def test_section_and_evidence_bounds_enforced(valid_note_dict):
    from app.schema import MAX_EVIDENCE_PER_ITEM, MAX_ITEMS_PER_SECTION

    too_many = dict(valid_note_dict, plan=[{"text": f"p{i}", "evidence": [1]} for i in range(MAX_ITEMS_PER_SECTION + 1)])
    with pytest.raises(ValidationError):
        SOAPNote.model_validate(too_many)
    with pytest.raises(ValidationError):
        NoteItem(text="x", evidence=list(range(1, MAX_EVIDENCE_PER_ITEM + 2)))
    sch = soap_json_schema()
    assert sch["properties"]["plan"]["maxItems"] == MAX_ITEMS_PER_SECTION  # goes into the grammar
