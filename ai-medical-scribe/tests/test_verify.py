"""Verifier tests (phase 2). Status rules are tested with a scripted backend so
they are deterministic; the real NLI model is tested in test_verify_nli.py."""
import json

import pytest

from app.config import Settings
from app.schema import SOAPNote
from app.verify import Judgement, OllamaJudge, Status, get_entailer, verify_note


class FakeEntailer:
    name = "fake"

    def __init__(self, table=None, default=(0.95, 0.0)):
        self.table = table or {}
        self.default = default
        self.calls = []

    def judge(self, pairs):
        self.calls.extend(pairs)
        out = []
        for _prem, hyp, _sec in pairs:
            ent, con = self.table.get(hyp, self.default)
            out.append(Judgement(entailment=ent, contradiction=con))
        return out


def note(**sections):
    base = {"subjective": [], "objective": [], "assessment": [], "plan": []}
    base.update(sections)
    return SOAPNote.model_validate(base)


def item(text, evidence, unsupported=False):
    return {"text": text, "evidence": evidence, "unsupported": unsupported}


def test_all_supported(segments, valid_note_dict):
    rep = verify_note(SOAPNote.model_validate(valid_note_dict), segments, entailer=FakeEntailer())
    assert rep.counts == {"supported": 6, "flagged": 0, "unsupported": 0}
    assert all(not i.reasons for i in rep.items)
    assert rep.items[0].entailment == 0.95


def test_empty_evidence_unsupported_and_not_sent_to_backend(segments):
    fe = FakeEntailer()
    rep = verify_note(note(subjective=[item("Patient smokes", [])]), segments, entailer=fe)
    it = rep.items[0]
    assert it.status == Status.unsupported and "No transcript segment cited." in it.reasons
    assert fe.calls == []


def test_nonexistent_ids_unsupported(segments):
    rep = verify_note(note(plan=[item("MRI", [42, 77])]), segments, entailer=FakeEntailer())
    it = rep.items[0]
    assert it.status == Status.unsupported and it.invalid_ids == [42, 77]
    assert "do not exist" in it.reasons[0]


def test_partially_invalid_ids_flagged(segments):
    rep = verify_note(note(plan=[item("Ibuprofen 400 mg", [6, 99])]), segments, entailer=FakeEntailer())
    it = rep.items[0]
    assert it.status == Status.flagged and it.valid_ids == [6] and it.invalid_ids == [99]


def test_generator_self_flag_is_unsupported(segments):
    fe = FakeEntailer()
    rep = verify_note(note(assessment=[item("Possible meniscal tear", [6], unsupported=True)]), segments, entailer=fe)
    assert rep.items[0].status == Status.unsupported
    assert "Generator marked" in rep.items[0].reasons[0]
    assert fe.calls == []  # no point checking a line the generator already disowned


@pytest.mark.parametrize(
    "ent,con,expected,phrase",
    [
        (0.95, 0.01, Status.supported, None),
        (0.50, 0.10, Status.supported, None),  # threshold is inclusive
        (0.35, 0.10, Status.flagged, "weakly supported"),
        (0.05, 0.02, Status.unsupported, "does not state"),
        (0.02, 0.97, Status.unsupported, "contradicts"),
    ],
)
def test_nli_thresholds(segments, ent, con, expected, phrase):
    fe = FakeEntailer({"Start naproxen": (ent, con)})  # no numbers -> only the NLI rule fires
    rep = verify_note(note(plan=[item("Start naproxen", [6])]), segments, entailer=fe)
    it = rep.items[0]
    assert it.status == expected
    if phrase:
        assert phrase in it.reasons[-1]


def test_over_citation_flagged(segments):
    s = Settings()
    many = list(range(1, 7))  # 6 segments == max_evidence_ok -> fine
    rep = verify_note(note(subjective=[item("Knee pain", many)]), segments, entailer=FakeEntailer(), settings=s)
    assert rep.items[0].status == Status.supported
    from dataclasses import replace

    s2 = replace(s, max_evidence_ok=3)
    rep = verify_note(note(subjective=[item("Knee pain", many)]), segments, entailer=FakeEntailer(), settings=s2)
    assert rep.items[0].status == Status.flagged and "too broad" in rep.items[0].reasons[0]


def test_red_beats_amber(segments):
    fe = FakeEntailer({"MRI ordered": (0.01, 0.0)})
    rep = verify_note(note(plan=[item("MRI ordered", [6, 99])]), segments, entailer=fe)
    assert rep.items[0].status == Status.unsupported
    assert len(rep.items[0].reasons) == 2  # both problems are reported


def test_premise_sent_to_backend_contains_context(segments):
    fe = FakeEntailer()
    verify_note(note(subjective=[item("Knee swelling after walking", [4])]), segments, entailer=fe)
    prem, hyp, sec = fe.calls[0]
    assert prem.startswith("Doctor: any swelling ?") and sec == "subjective"


def test_suggestions_only_for_non_supported(segments):
    fe = FakeEntailer({"Ibuprofen 400 mg": (0.01, 0.0)})
    rep = verify_note(
        note(subjective=[item("Right knee pain", [2])], plan=[item("Ibuprofen 400 mg", [1])]),
        segments, entailer=fe,
    )
    ok, bad = rep.items
    assert ok.suggested_evidence == []
    assert bad.status == Status.unsupported and bad.suggested_evidence[0] == 6
    assert bad.evidence == [1]  # suggestion never rewrites the citation


def test_report_lookup_and_order(segments, valid_note_dict):
    rep = verify_note(SOAPNote.model_validate(valid_note_dict), segments, entailer=FakeEntailer())
    assert [(i.section, i.index) for i in rep.items][:3] == [("subjective", 0), ("subjective", 1), ("objective", 0)]
    assert rep.for_item("plan", 1).text == "Ibuprofen 400 mg"
    with pytest.raises(KeyError):
        rep.for_item("plan", 9)


def test_empty_note(segments):
    rep = verify_note(note(), segments, entailer=FakeEntailer())
    assert rep.items == [] and rep.counts["supported"] == 0


# ---- Ollama judge backend (scripted LLM) ----
def test_ollama_judge_mapping(segments, fake_llm_factory):
    llm = fake_llm_factory([
        json.dumps({"supported": True, "reason": "stated in segment 6"}),
        json.dumps({"supported": False, "reason": "dose is 400 mg, not 800 mg"}),
        "not json at all",
    ])
    judge = OllamaJudge(client=llm)
    rep = verify_note(
        note(plan=[item("Ibuprofen 400 mg", [6]), item("Ibuprofen 800 mg", [6]), item("X-ray", [6])]),
        segments, entailer=judge,
    )
    a, b, c = rep.items
    assert a.status == Status.supported
    assert b.status == Status.unsupported and "400 mg, not 800 mg" in b.reasons[0]
    assert c.status == Status.flagged and "judge error" in c.reasons[0]  # judge failure -> doctor checks it
    call = llm.calls[0]
    assert call["options"]["temperature"] == 0 and call["format"]["required"] == ["supported", "reason"]
    assert "NOTE LINE (plan): Ibuprofen 400 mg" in call["messages"][1]["content"]


def test_get_entailer_flag(monkeypatch):
    monkeypatch.setenv("SCRIBE_VERIFIER", "ollama")
    assert get_entailer().name.startswith("ollama:")
    monkeypatch.setenv("SCRIBE_VERIFIER", "nli")
    assert get_entailer().name == "nli:cross-encoder/nli-deberta-v3-base"
    monkeypatch.setenv("SCRIBE_VERIFIER", "magic")
    with pytest.raises(ValueError, match="unknown SCRIBE_VERIFIER"):
        get_entailer()


def test_ungrounded_number_flags_even_if_nli_says_supported(segments):
    # NLI (scripted) says entailed, but "800" never appears in segment 6 -> amber
    rep = verify_note(note(plan=[item("Ibuprofen 800 mg", [6])]), segments, entailer=FakeEntailer())
    it = rep.items[0]
    assert it.status == Status.flagged and it.ungrounded_tokens == ["800"]
    assert "'800'" in it.reasons[0]


def test_ungrounded_laterality_flagged(segments):
    rep = verify_note(note(objective=[item("Mild left knee effusion", [5])]), segments, entailer=FakeEntailer())
    assert rep.items[0].status == Status.flagged and rep.items[0].ungrounded_tokens == ["left"]
