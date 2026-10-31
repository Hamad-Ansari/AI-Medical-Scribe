"""API tests (FastAPI TestClient) with a scripted LLM, fake verifier and fake voices."""
import io
import json
import wave

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import Services, create_app
from app.store import NoteStore

from .conftest import FakeLLM, has_aci_data
from .test_tts import FakeVoice
from .test_verify import FakeEntailer

DIALOGUE = "\n".join([
    "[doctor] hi , brian . what brings you in today ? you can reach me at 555-201-3344 .",
    "[patient] my right knee has been hurting for about two weeks .",
    "[doctor] any swelling ?",
    "[patient] yes , it swells after i walk .",
    "[doctor] on exam there is mild effusion of the right knee .",
    "[doctor] i think this is a knee sprain . let's get an x-ray and start ibuprofen 400 mg .",
])


@pytest.fixture
def make_client(tmp_path, valid_note_json):
    def _make(llm=None, entailer=None):
        svc = Services(
            store=NoteStore(tmp_path / "api.db"),
            llm=llm or FakeLLM([valid_note_json] * 10),
            entailer=entailer or FakeEntailer(),
            voices={"doctor": FakeVoice(), "patient": FakeVoice()},
        )
        return TestClient(create_app(svc)), svc
    return _make


@pytest.fixture
def client(make_client):
    return make_client()[0]


@pytest.fixture
def note_id(client):
    r = client.post("/process", json={"dialogue": DIALOGUE})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------- /process
def test_process_dialogue_creates_redacted_draft(make_client):
    client, svc = make_client()
    r = client.post("/process", json={"dialogue": DIALOGUE})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "draft"
    assert body["source"]["redactions"] == {"[NAME]": 1, "[PHONE]": 1}
    seg_text = " ".join(s["text"] for s in body["segments"])
    assert "brian" not in seg_text and "555-201-3344" not in seg_text and "[NAME]" in seg_text
    prompt = " ".join(m["content"] for m in svc.llm.calls[0]["messages"])
    assert "brian" not in prompt and "555-201-3344" not in prompt
    assert body["counts"] == {"supported": 6, "flagged": 0, "unsupported": 0}
    # stored copy is redacted too
    stored = svc.store.get(body["id"]).doc.model_dump_json()
    assert "brian" not in stored and "555-201-3344" not in stored


@pytest.mark.skipif(not has_aci_data(), reason="ACI-Bench not downloaded")
def test_process_aci_case(client):
    r = client.post("/process", json={"encounter_id": "D2N068", "split": "valid", "max_turns": 10})
    assert r.status_code == 201, r.text
    b = r.json()
    assert b["source"]["encounter_id"] == "D2N068" and len(b["segments"]) == 10
    assert b["source"]["input"] == "text"


@pytest.mark.parametrize("payload, fragment", [
    ({}, "exactly one"),
    ({"dialogue": DIALOGUE, "encounter_id": "D2N068"}, "exactly one"),
    ({"dialogue": "   "}, "exactly one"),
    ({"dialogue": DIALOGUE, "input": "audio"}, "needs an encounter_id"),
    ({"encounter_id": "D2N068", "max_turns": 0}, "greater than or equal"),
])
def test_process_validation_errors(client, payload, fragment):
    r = client.post("/process", json=payload)
    assert r.status_code == 422
    assert fragment in json.dumps(r.json())


def test_process_rejects_non_json(client):
    r = client.post("/process", content=b"not json", headers={"content-type": "text/plain"})
    assert r.status_code == 422 and "must be JSON" in r.json()["detail"]


@pytest.mark.skipif(not has_aci_data(), reason="ACI-Bench not downloaded")
def test_process_unknown_encounter(client):
    r = client.post("/process", json={"encounter_id": "NOPE123"})
    assert r.status_code == 404 and "NOPE123" in r.json()["detail"]


def _wav_bytes(samples: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


@pytest.mark.parametrize("data, fragment", [
    (b"", "empty"),
    (b"this is not audio at all" * 10, "could not decode"),
    (_wav_bytes(np.zeros(16000 * 2)), "silent"),
    (_wav_bytes((np.random.default_rng(0).standard_normal(800) * 3000)), "too short"),
])
def test_bad_audio_upload_gives_clear_422(client, data, fragment):
    r = client.post("/process", files={"audio": ("a.wav", data, "audio/wav")})
    assert r.status_code == 422, r.text
    assert r.json()["detail"].startswith("invalid audio:") and fragment in r.json()["detail"]


def test_upload_needs_audio_field(client):
    r = client.post("/process", files={"file": ("a.wav", b"x", "audio/wav")})
    assert r.status_code == 422 and "'audio'" in r.json()["detail"]


def test_ollama_down_gives_503(make_client):
    class DownLLM:
        def chat(self, *a, **k):
            raise httpx.ConnectError("connection refused")
    client, _ = make_client(llm=DownLLM())
    r = client.post("/process", json={"dialogue": DIALOGUE})
    assert r.status_code == 503 and "ollama serve" in r.json()["detail"]


def test_invalid_llm_output_gives_502(make_client):
    client, _ = make_client(llm=FakeLLM(["not json", "{\"still\": \"bad\"}"]))
    r = client.post("/process", json={"dialogue": DIALOGUE})
    assert r.status_code == 502 and "could not be validated" in r.json()["detail"]


# ---------------------------------------------------------------- GET / PUT
def test_get_note_and_404(client, note_id):
    r = client.get(f"/note/{note_id}")
    assert r.status_code == 200 and r.json()["id"] == note_id
    assert len(r.json()["verification"]["items"]) == 6
    assert client.get("/note/9999").status_code == 404


def _note(client, nid):
    return client.get(f"/note/{nid}").json()["note"]


def test_edit_delete_add_reverifies_only_changed_lines(make_client):
    ent = FakeEntailer()
    client, _ = make_client(entailer=ent)
    nid = client.post("/process", json={"dialogue": DIALOGUE}).json()["id"]
    n_initial = len(ent.calls)
    note = _note(client, nid)
    note["plan"][1]["text"] = "Ibuprofen 400 mg three times daily"        # edit
    del note["subjective"][1]                                            # delete
    note["plan"].append({"text": "Follow up in two weeks", "evidence": [], "unsupported": False})  # add
    r = client.put(f"/note/{nid}", json={"note": note, "actor": "dr_test"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["status"] == "edited" and b["rechecked"] == 2
    assert len(ent.calls) - n_initial == 1  # the added line has no evidence -> no NLI call
    items = {(i["section"], i["index"]): i for i in b["verification"]["items"]}
    assert items[("plan", 2)]["status"] == "unsupported"          # doctor-added line without evidence: red
    assert items[("plan", 1)]["text"] == "Ibuprofen 400 mg three times daily"
    assert len(b["note"]["subjective"]) == 1
    a = client.get(f"/note/{nid}/audit").json()[-1]
    assert a["action"] == "edited" and a["after"]["by"] == "dr_test"
    assert len(a["before"]["note"]["subjective"]) == 2 and len(a["after"]["note"]["subjective"]) == 1


def test_unchanged_save_rechecks_nothing(client, note_id):
    r = client.put(f"/note/{note_id}", json={"note": _note(client, note_id)})
    assert r.status_code == 200 and r.json()["rechecked"] == 0 and r.json()["status"] == "draft"


def test_doctor_edits_are_redacted_before_storage(make_client):
    client, svc = make_client()
    nid = client.post("/process", json={"dialogue": DIALOGUE}).json()["id"]
    note = _note(client, nid)
    note["plan"].append({"text": "Call Mrs. Jennifer Smith at 555-888-1234", "evidence": [], "unsupported": True})
    r = client.put(f"/note/{nid}", json={"note": note})
    assert r.json()["redactions"] >= 2
    stored = svc.store.get(nid).doc.model_dump_json() + json.dumps(client.get(f"/note/{nid}/audit").json())
    assert "Smith" not in stored and "555-888-1234" not in stored


def test_put_invalid_note_is_422(client, note_id):
    note = _note(client, note_id)
    note["plan"][0]["text"] = "   "
    assert client.put(f"/note/{note_id}", json={"note": note}).status_code == 422
    note = _note(client, note_id)
    note["extra_section"] = []
    assert client.put(f"/note/{note_id}", json={"note": note}).status_code == 422


# ---------------------------------------------------------------- approval gate
def test_approval_gate_end_to_end(client, note_id):
    assert client.get(f"/final/{note_id}").status_code == 409
    assert client.get("/final").json() == []
    assert client.get(f"/note/{note_id}/audio").status_code == 409   # can't read a draft aloud
    r = client.post(f"/note/{note_id}/approve", json={"actor": "dr_test"})
    assert r.status_code == 200 and r.json()["status"] == "approved"
    assert client.get(f"/final/{note_id}").status_code == 200
    assert [n["id"] for n in client.get("/final").json()] == [note_id]
    assert client.put(f"/note/{note_id}", json={"note": _note(client, note_id)}).status_code == 409
    assert client.post(f"/note/{note_id}/approve").status_code == 409
    actions = [a["action"] for a in client.get(f"/note/{note_id}/audit").json()]
    assert actions == ["created", "approved"]


def test_approve_with_red_lines_is_allowed_and_recorded(make_client, valid_note_dict):
    bad = dict(valid_note_dict)
    bad["plan"] = bad["plan"] + [{"text": "Start metformin", "evidence": [], "unsupported": True}]
    client, _ = make_client(llm=FakeLLM([json.dumps(bad)]))
    nid = client.post("/process", json={"dialogue": DIALOGUE}).json()["id"]
    r = client.post(f"/note/{nid}/approve", json={"actor": "dr_test"})
    assert r.status_code == 200
    a = client.get(f"/note/{nid}/audit").json()[-1]
    assert a["after"]["line_status_at_approval"]["unsupported"] == 1


def test_read_aloud_only_after_approval(client, note_id):
    client.post(f"/note/{note_id}/approve")
    r = client.get(f"/note/{note_id}/audio")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
    assert client.get(f"/note/{note_id}/audit").json()[-1]["action"] == "read_aloud"


def test_notes_listing_and_filter(client):
    a = client.post("/process", json={"dialogue": DIALOGUE}).json()["id"]
    b = client.post("/process", json={"dialogue": DIALOGUE}).json()["id"]
    client.post(f"/note/{b}/approve")
    assert [n["id"] for n in client.get("/notes").json()] == [b, a]
    assert [n["id"] for n in client.get("/notes?status=draft").json()] == [a]
    assert client.get("/notes?status=final").status_code == 422


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


# ---------------------------------------------------------------- integration (real models)
@pytest.mark.slow
def test_integration_audio_upload_to_segments(make_client, tmp_path):
    """FastAPI + real audio -> Whisper transcription -> redacted segments -> draft."""
    from app.tts import get_voices, synthesize_dialogue, voice_path
    from app.config import get_settings

    s = get_settings()
    if not voice_path(s.doctor_voice).exists():
        pytest.skip("Piper voices not downloaded (make voices)")
    dlg = [("doctor", "hi brian , what brings you in today ?"),
           ("patient", "my right knee has been hurting for about two weeks ."),
           ("doctor", "let's get an x-ray and start ibuprofen 400 mg .")]
    audio = synthesize_dialogue(dlg, tmp_path / "upload.wav", voices=get_voices(), deterministic=True)
    client, svc = make_client()
    with open(audio.path, "rb") as fh:
        r = client.post("/process", files={"audio": ("visit.wav", fh, "audio/wav")})
    assert r.status_code == 201, r.text
    b = r.json()
    text = " ".join(x["text"] for x in b["segments"]).lower()
    assert "knee" in text and "ibuprofen" in text and "400" in text
    assert "brian" not in text  # redacted after transcription
    assert all(x["start"] is not None for x in b["segments"])
    assert b["source"]["input"] == "upload" and b["source"]["speaker_source"] == "none"


@pytest.mark.slow
@pytest.mark.skipif(not has_aci_data(), reason="ACI-Bench not downloaded")
def test_integration_aci_case_as_audio(make_client):
    """ACI case -> Piper -> Whisper with speaker alignment -> draft via the API."""
    from app.tts import get_voices

    client, svc = make_client()
    svc.voices = get_voices()
    r = client.post("/process", json={"encounter_id": "D2N068", "input": "audio", "max_turns": 6})
    assert r.status_code == 201, r.text
    b = r.json()
    assert b["source"]["speaker_source"] == "reference"
    assert {x["speaker"] for x in b["segments"]} <= {"doctor", "patient", "patient_guest"}
    assert b["source"]["audio_duration_s"] > 5


# ---------------------------------------------------------------- low-memory mode
def test_low_memory_releases_only_lazily_built_models(tmp_path, valid_note_json, monkeypatch):
    """SCRIBE_LOW_MEMORY=1 frees cached models before the LLM runs - but never injected ones."""
    import dataclasses

    from app import main
    from app.config import get_settings

    built = []
    monkeypatch.setattr(main, "get_entailer", lambda s: built.append(1) or FakeEntailer())
    svc = Services(store=NoteStore(tmp_path / "lm.db"), llm=FakeLLM([valid_note_json] * 4),
                   settings=dataclasses.replace(get_settings(), low_memory=True),
                   voices={"doctor": FakeVoice(), "patient": FakeVoice()})
    client = TestClient(create_app(svc))
    assert client.post("/process", json={"dialogue": DIALOGUE}).status_code == 201
    assert client.post("/process", json={"dialogue": DIALOGUE}).status_code == 201
    assert len(built) == 2                       # NLI dropped before the 2nd generation, rebuilt after it
    assert isinstance(svc.voices["doctor"], FakeVoice)   # injected voices untouched
