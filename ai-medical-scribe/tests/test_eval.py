"""Evaluation logic (eval/run_eval.py) with a scripted LLM and a fake NLI model."""
import json
import math
import sys
from pathlib import Path

import pytest

from app.schema import SOAPNote

from .conftest import FakeLLM, has_aci_data
from .test_verify import FakeEntailer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import run_eval  # noqa: E402


def test_reference_facts_strip_bullets_labels_and_headers():
    text = ("Right knee pain.\n"
            "• Medical Reasoning: She has a sprain. It should heal in two weeks.\n"
            "- Medical Treatment: Start ibuprofen 400 mg twice daily.\n"
            "1. Follow up in 2 weeks if no better.")
    assert run_eval.reference_facts(text) == [
        "She has a sprain.", "It should heal in two weeks.",
        "Start ibuprofen 400 mg twice daily.", "Follow up in 2 weeks if no better.",
    ]  # "Right knee pain." (3 words) is a header, not a fact


def test_rouge_l_edge_cases():
    assert run_eval.rouge_l("knee pain for two weeks", "knee pain for two weeks") == pytest.approx(1.0)
    assert run_eval.rouge_l("", "knee pain") == 0.0          # nothing generated: 0
    assert math.isnan(run_eval.rouge_l("knee pain", ""))     # nothing to compare against: not scored
    assert 0 < run_eval.rouge_l("right knee pain", "left knee pain for two weeks") < 1


def test_chunk_lines_respects_word_limit():
    chunks = run_eval.chunk_lines(["one two three"] * 10, max_words=7)
    assert len(chunks) == 5 and all(len(c.split()) <= 7 for c in chunks)


def test_invented_uses_whole_transcript_not_only_citations(segments, valid_note_dict):
    note = SOAPNote.model_validate(valid_note_dict)
    note.plan[0].evidence = []                    # wrong / missing citation...
    invented_text = "Start metformin 500 mg"
    note.plan.append(note.plan[0].model_copy(update={"text": invented_text, "evidence": [6]}))
    ent = FakeEntailer(table={invented_text: (0.05, 0.1)})   # ...but only metformin is unsupported anywhere
    lines = run_eval.find_invented(note, segments, ent)
    assert [ln["text"] for ln in lines if ln["invented"]] == [invented_text]
    # the uncited plan line was still checked against transcript windows
    assert any(h == note.plan[0].text for _, h, _ in ent.calls)


def test_missing_facts_counted_per_section(valid_note_dict):
    note = SOAPNote.model_validate(valid_note_dict)
    ref = {"subjective": "The patient has right knee pain for two weeks.",
           "objective": "There is mild effusion of the right knee.",
           "assessment": "", "plan": "We will check her blood sugar today."}
    ent = FakeEntailer(table={"We will check her blood sugar today.": (0.1, 0.0)})
    facts = run_eval.find_missing(note, ref, ent)
    assert [(f["section"], f["missing"]) for f in facts] == [
        ("subjective", False), ("objective", False), ("plan", True)]


def test_empty_note_misses_everything():
    note = SOAPNote.model_validate({s: [] for s in ("subjective", "objective", "assessment", "plan")})
    facts = run_eval.find_missing(note, {"subjective": "The knee has hurt for two weeks."}, FakeEntailer())
    assert facts and all(f["missing"] for f in facts)


@pytest.mark.skipif(not has_aci_data(), reason="ACI-Bench data not downloaded")
def test_generate_then_score_end_to_end(tmp_path, valid_note_json):
    """Both stages on 2 real ACI cases: one valid note, one model failure; cached notes are reused."""
    from app.loader import load_split

    cases = load_split("valid", limit=2)
    llm = FakeLLM([valid_note_json, "not json", "still not json"])
    run_eval.generate(["fake-model"], cases, tmp_path, client=llm)
    files = sorted((tmp_path / "notes" / "fake-model").glob("*.json"))
    assert len(files) == 2
    recs = [json.loads(f.read_text()) for f in files]
    assert [r["ok"] for r in recs] == [True, False]
    assert all("[NAME]" in " ".join(s["text"] for s in r["segments"]) for r in recs)  # redacted before the LLM

    run_eval.generate(["fake-model"], cases, tmp_path, client=FakeLLM([]))  # cached: no LLM calls

    rows = run_eval.score(["fake-model"], cases, tmp_path, entailer=FakeEntailer(), bertscore_model=None)
    ok, failed = rows
    assert failed["ok"] is False and "invalid" in failed["error"]
    assert ok["n_items"] == SOAPNote.model_validate(recs[0]["note"]).n_items()
    assert ok["ref_facts"] > 0 and 0 <= ok["missing"] <= ok["ref_facts"]
    assert 0 <= ok["rougeL_note"] <= 1
    summ = run_eval.summarize(rows)[0]
    assert summ["valid_notes"] == 1 and summ["cases"] == 2
    md = run_eval.markdown([summ], "valid", None)
    assert "| fake-model | 1/2 |" in md and "BERTScore" not in md
    assert (tmp_path / "eval_cases.csv").exists() and (tmp_path / "eval_facts.csv").exists()


def test_update_readme_replaces_only_the_marked_tables(tmp_path):
    import update_readme

    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- EVAL_TABLES_START (filled by eval/update_readme.py) -->\nOLD\n"
                      "<!-- EVAL_TABLES_END -->\n\nHow to read this\n")
    summary = tmp_path / "eval_summary.md"
    summary.write_text("## Note quality\n\n| Model | Valid notes |\n|---|---|\n| qwen2.5:7b | 20/20 |\n\n"
                       "Speech-to-text ... WER\n\nInvented = note line ...\n")
    out = update_readme.update(readme, summary)
    assert "OLD" not in out and "| qwen2.5:7b | 20/20 |" in out
    assert out.startswith("intro\n") and out.endswith("How to read this\n") and "Speech-to-text" not in out
