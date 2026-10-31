"""Generation tests with a fake LLM (phase 1). No network, deterministic."""
import json

import httpx
import pytest

from app.config import Settings
from app.generate import GenerationError, OllamaClient, build_messages, generate_note, parse_note


def test_generate_happy_path(segments, valid_note_json, fake_llm_factory):
    llm = fake_llm_factory([valid_note_json])
    res = generate_note(segments, client=llm)
    assert res.attempts == 1 and res.note.n_items() == 6
    call = llm.calls[0]
    assert call["options"]["temperature"] == 0.1
    assert set(call["format"]["required"]) == {"subjective", "objective", "assessment", "plan"}
    user = call["messages"][1]["content"]
    assert "[1] DOCTOR: what brings you in today ?" in user
    assert "[6] DOCTOR:" in user


def test_prompt_contains_rules(segments):
    sysmsg = build_messages(segments)[0]["content"]
    for phrase in ("ONLY information", "cite the ID", '"unsupported": true', "short and atomic"):
        assert phrase in sysmsg


def test_retry_once_on_invalid_json(segments, valid_note_json, fake_llm_factory):
    llm = fake_llm_factory(["{not json", valid_note_json])
    res = generate_note(segments, client=llm)
    assert res.attempts == 2
    retry_msgs = llm.calls[1]["messages"]
    assert retry_msgs[-2]["role"] == "assistant" and retry_msgs[-2]["content"] == "{not json"
    assert "could not be accepted" in retry_msgs[-1]["content"]


def test_retry_once_on_schema_violation(segments, valid_note_dict, fake_llm_factory):
    bad = dict(valid_note_dict)
    bad.pop("plan")
    llm = fake_llm_factory([json.dumps(bad), json.dumps(valid_note_dict)])
    res = generate_note(segments, client=llm)
    assert res.attempts == 2
    assert "plan" in llm.calls[1]["messages"][-1]["content"]  # error message names the field


def test_fails_after_two_invalid(segments, fake_llm_factory):
    llm = fake_llm_factory(["nope", '{"subjective": []}'])
    with pytest.raises(GenerationError, match="after 2 attempts"):
        generate_note(segments, client=llm)
    assert len(llm.calls) == 2  # exactly one retry, no more


def test_empty_transcript_rejected(fake_llm_factory):
    with pytest.raises(ValueError, match="empty transcript"):
        generate_note([], client=fake_llm_factory([]))


def test_parse_note_tolerates_code_fence(valid_note_json):
    assert parse_note(f"```json\n{valid_note_json}\n```").n_items() == 6


# ---- OllamaClient over a mocked HTTP transport ----
def _client(handler):
    return OllamaClient(Settings(), transport=httpx.MockTransport(handler))


def test_ollama_client_payload(valid_note_json):
    seen = {}

    def handler(req: httpx.Request):
        seen.update(json.loads(req.content))
        assert req.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"role": "assistant", "content": valid_note_json}})

    out = _client(handler).chat([{"role": "user", "content": "hi"}], format={"type": "object"}, options={"temperature": 0.1})
    assert out == valid_note_json
    assert seen["stream"] is False and seen["format"] == {"type": "object"}
    assert seen["model"] == "qwen2.5:7b"  # default model


def test_ollama_client_missing_model():
    c = _client(lambda req: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(GenerationError, match="ollama pull"):
        c.chat([], format=None, options={})


def test_ollama_client_server_down():
    def handler(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(GenerationError, match="ollama serve"):
        _client(handler).chat([], format=None, options={})


def test_duplicate_items_dropped_and_evidence_merged(segments, valid_note_dict, fake_llm_factory):
    valid_note_dict["plan"].append({"text": "  ibuprofen 400 MG ", "evidence": [5], "unsupported": False})
    res = generate_note(segments, client=fake_llm_factory([json.dumps(valid_note_dict)]))
    plan = res.note.plan
    assert [p.text for p in plan] == ["Right knee x-ray", "Ibuprofen 400 mg"]
    assert plan[1].evidence == [6, 5]  # merged from the duplicate


def test_num_predict_cap_sent(segments, valid_note_json, fake_llm_factory):
    llm = fake_llm_factory([valid_note_json])
    generate_note(segments, client=llm)
    assert llm.calls[0]["options"]["num_predict"] == 3072


def test_ollama_client_timeout():
    def handler(req):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(GenerationError, match="timed out"):
        _client(handler).chat([], format=None, options={})


def test_num_batch_only_sent_when_set(segments, valid_note_json):
    """SCRIBE_NUM_BATCH (low-RAM knob) is passed to Ollama only when configured."""
    import dataclasses

    from app.config import get_settings
    from app.generate import generate_note

    from .conftest import FakeLLM

    llm = FakeLLM([valid_note_json, valid_note_json])
    generate_note(segments, client=llm, settings=get_settings())
    assert "num_batch" not in llm.calls[0]["options"]
    generate_note(segments, client=llm, settings=dataclasses.replace(get_settings(), num_batch=128))
    assert llm.calls[1]["options"]["num_batch"] == 128
