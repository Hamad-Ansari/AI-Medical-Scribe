"""Evidence linking tests (phase 2)."""
from app.evidence import SegmentIndex, build_premise, resolve, speaker_label
from app.schema import NoteItem


def test_resolve_splits_valid_invalid(segments):
    by_id = {s.id: s for s in segments}
    r = resolve(NoteItem(text="x", evidence=[2, 99, 4, 0]), by_id)
    assert r.valid_ids == [2, 4] and r.invalid_ids == [99, 0]


def test_premise_has_speakers_and_order(segments):
    by_id = {s.id: s for s in segments}
    prem, ctx = build_premise([5, 2], by_id, short_words=0)
    assert prem.startswith("Patient: my right knee") and "Doctor: on exam" in prem
    assert ctx == []


def test_short_answer_gets_previous_turn_as_context(segments):
    by_id = {s.id: s for s in segments}
    # seg 4 "yes , it swells after i walk ." has 7 words < 8 -> include seg 3 question
    prem, ctx = build_premise([4], by_id, short_words=8)
    assert ctx == [3]
    assert prem.startswith("Doctor: any swelling ?")


def test_context_not_duplicated_when_already_cited(segments):
    by_id = {s.id: s for s in segments}
    prem, ctx = build_premise([3, 4], by_id, short_words=8)
    assert 3 not in ctx and prem.count("any swelling") == 1  # seg 3 cited, not re-added
    assert ctx == [2]  # seg 3 is itself short, so its own previous turn is context


def test_speaker_labels():
    assert speaker_label("patient_guest") == "Patient's companion"
    assert speaker_label("nurse") == "Nurse"


def test_suggestions_rank_relevant_segment(segments):
    idx = SegmentIndex(segments)
    assert idx.suggest("Swelling of the knee after walking")[0] == 4
    assert idx.suggest("ibuprofen 400 mg")[0] == 6
    assert idx.suggest("okay so um") == []  # only stopwords -> nothing
    assert 6 not in idx.suggest("ibuprofen", exclude={6})


import pytest  # noqa: E402

from app.evidence import _numbers, ungrounded_critical_tokens  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("lisinopril 20 mg", {"20"}),
    ("150/90", {"150", "90"}),
    ("one hundred and twenty", {"120"}),
    ("eighty-five", {"85"}),
    ("a1c was 6.5", {"6.5"}),          # 'a1c' is not a number
    ("for two weeks", {"2"}),
    ("okay , right .", set()),
])
def test_number_parsing(text, expected):
    assert _numbers(text) == expected


@pytest.mark.parametrize("item,premise,missing", [
    ("Lasix 40 mg daily", "Doctor: increase your lasix to 80 mg once a day .", ["40"]),
    ("Lasix 80 mg daily", "Doctor: increase your lasix to eighty mg once a day .", []),
    ("BP 150/90", "Doctor: your blood pressure is one fifty over ninety .", []),  # colloquial
    ("Left knee pain", "Patient: my right knee hurts .", ["left"]),
    ("Bilateral knee pain", "Patient: both of my knees hurt .", ["bilateral"]),
    ("Pain for 3 weeks", "Patient: it has hurt for about two weeks .", ["3"]),
    ("Knee pain", "Patient: my right knee hurts .", []),
])
def test_ungrounded_critical_tokens(item, premise, missing):
    assert ungrounded_critical_tokens(item, premise) == missing
