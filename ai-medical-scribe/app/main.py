"""FastAPI service: transcript -> redact -> SOAP draft -> doctor review -> approval.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints (specified)
    POST /process                 create a draft note (JSON or multipart audio upload)
    GET  /note/{id}               note + redacted transcript + per-line verification
    PUT  /note/{id}               save doctor edits (redacted, changed lines re-verified)
    POST /note/{id}/approve       doctor approval - the only way a note becomes final
Extra
    GET  /notes                   list notes (optionally ?status=)
    GET  /note/{id}/audit         audit history
    GET  /final, /final/{id}      approved notes only (409 for anything else)
    GET  /note/{id}/audio         approved note read aloud by Piper (WAV)
    GET  /cases                   ACI-Bench encounter ids for the UI picker
    GET  /health

Nothing is final until ``POST /note/{id}/approve``; approved notes are read-only
(enforced again by SQLite triggers in app.store).
"""
from __future__ import annotations

import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.config import Settings, get_settings
from app.generate import GenerationError, LLMClient
from app.redact import install_log_redaction, redact_text
from app.schema import SECTIONS, NoteItem, SOAPNote
from app.store import (AuditEntry, NoteDocument, NoteLocked, NoteNotFound, NoteStatus, NoteStore,
                       NotApproved)
from app.verify import Entailer, VerificationReport, get_entailer, reverify_note

log = logging.getLogger("app.main")
install_log_redaction(logging.getLogger("app"))


# ------------------------------------------------------------------ services
@dataclass
class Services:
    """Injected dependencies. ``None`` means "build the real one lazily"."""

    store: NoteStore
    settings: Settings = field(default_factory=get_settings)
    llm: LLMClient | None = None
    entailer: Entailer | None = None
    voices: dict | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _owned: set = field(default_factory=set, repr=False)  # models built here (safe to release)

    def get_entailer(self) -> Entailer:
        with self._lock:
            if self.entailer is None:
                self.entailer = get_entailer(self.settings)
                self._owned.add("entailer")
            return self.entailer

    def release_models(self) -> None:
        """SCRIBE_LOW_MEMORY=1: drop every cached model (Whisper, Piper, NLI) so the
        LLM is the only big model in RAM while it runs. Each reloads lazily."""
        import gc

        from app import stt, tts

        with self._lock:
            stt._whisper.cache_clear()
            tts._load_voice.cache_clear()
            for name in self._owned:  # never drop injected (test) models
                setattr(self, name, None)
            self._owned.clear()
        gc.collect()
        try:  # hand freed heap pages back to the OS (glibc keeps them otherwise: ~250 MB here)
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):  # not glibc (macOS, musl)
            pass

    def get_voices(self) -> dict:
        with self._lock:
            if self.voices is None:
                from app.tts import get_voices

                self.voices = get_voices(self.settings)
                self._owned.add("voices")
            return self.voices


# ------------------------------------------------------------------ API models
class ProcessRequest(BaseModel):
    """Either an ACI-Bench case (``encounter_id``) or a pasted ``dialogue``."""

    encounter_id: str | None = Field(default=None, examples=["D2N068"])
    split: str = "valid"
    dialogue: str | None = Field(default=None, description="'[doctor] ...\\n[patient] ...' lines (synthetic only)")
    input: Literal["text", "audio"] = Field(
        default="text", description="'audio' synthesises the case with Piper and transcribes it with Whisper")
    max_turns: int | None = Field(default=None, ge=1, description="use only the first N turns (quick demos)")

    @model_validator(mode="after")
    def _one_source(self):
        if bool(self.encounter_id) == bool(self.dialogue and self.dialogue.strip()):
            raise ValueError("give exactly one of 'encounter_id' or 'dialogue'")
        if self.dialogue and self.input == "audio":
            raise ValueError("input='audio' needs an encounter_id (pasted dialogue is text-only)")
        return self


class NoteSummary(BaseModel):
    id: int
    status: NoteStatus
    created_at: str
    encounter_id: str | None = None
    input: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    n_items: int = 0


class NoteView(BaseModel):
    id: int
    status: NoteStatus
    created_at: str
    note: SOAPNote
    segments: list
    verification: VerificationReport | None
    source: dict[str, Any]
    counts: dict[str, int]


class UpdateRequest(BaseModel):
    note: SOAPNote
    actor: str = Field(default="doctor", min_length=1, max_length=80)


class UpdateResponse(NoteView):
    rechecked: int = Field(description="lines re-verified because they were new or changed")
    redactions: int = Field(description="PII replacements made in the doctor's edits")


class ApproveRequest(BaseModel):
    actor: str = Field(default="doctor", min_length=1, max_length=80)


def _view(rec) -> NoteView:
    v = rec.doc.verification
    return NoteView(id=rec.id, status=rec.status, created_at=rec.created_at, note=rec.doc.note,
                    segments=[s.model_dump() for s in rec.doc.segments], verification=v,
                    source=rec.doc.source, counts=v.counts if v else {})


def _summary(rec) -> NoteSummary:
    v = rec.doc.verification
    return NoteSummary(id=rec.id, status=rec.status, created_at=rec.created_at,
                       encounter_id=rec.doc.source.get("encounter_id"), input=rec.doc.source.get("input"),
                       counts=v.counts if v else {}, n_items=rec.doc.note.n_items())


def _redact_note(note: SOAPNote) -> tuple[SOAPNote, int]:
    """Doctor edits may contain PII too: redact before it is stored or logged."""
    total = 0
    data: dict[str, list[NoteItem]] = {}
    for sec in SECTIONS:
        items = []
        for it in getattr(note, sec):
            r = redact_text(it.text)
            total += r.count
            items.append(it.model_copy(update={"text": r.text}) if r.count else it)
        data[sec] = items
    return SOAPNote(**data), total


# ------------------------------------------------------------------ processing
def _process_case(svc: Services, req: ProcessRequest):
    from app.loader import load_case, parse_dialogue
    from app.pipeline import generate_and_verify, prepare_from_audio, prepare_from_segments

    t0 = time.time()
    source: dict[str, Any] = {"input": req.input}
    if req.dialogue:
        segs = parse_dialogue(req.dialogue, max_words=svc.settings.max_segment_words)
        source["encounter_id"] = None
    else:
        case = load_case(req.encounter_id, split=req.split)
        segs = case.segments
        source.update(encounter_id=case.encounter_id, split=req.split)
    if req.max_turns:
        segs = segs[: req.max_turns]
        source["max_turns"] = req.max_turns
    if not segs:
        raise ValueError("transcript has no segments")
    if req.input == "audio":
        from app.tts import synthesize_dialogue

        with tempfile.TemporaryDirectory() as td:
            audio = synthesize_dialogue(segs, Path(td) / "consultation.wav", voices=svc.get_voices())
            prepared = prepare_from_audio(audio.path, reference_turns=audio.turns)
    else:
        prepared = prepare_from_segments(segs)
    return _finish(svc, prepared, source, t0)


def _process_upload(svc: Services, path: Path, filename: str):
    from app.pipeline import prepare_from_audio

    t0 = time.time()
    prepared = prepare_from_audio(path)  # no reference turns: speakers via pyannote (flag) or "unknown"
    return _finish(svc, prepared, {"input": "upload", "filename": Path(filename).name, "encounter_id": None}, t0)


def _finish(svc: Services, prepared, source: dict, t0: float):
    from app.generate import generate_note
    from app.verify import verify_note

    # generate first, *then* load the verifier: the LLM (Ollama) and the NLI
    # model are never both being loaded at once
    if svc.settings.low_memory:
        svc.release_models()
    gen = generate_note(prepared.segments, client=svc.llm, settings=svc.settings)
    t_gen = time.time() - t0
    rep = verify_note(gen.note, prepared.segments, entailer=svc.get_entailer(), settings=svc.settings)
    source.update(
        model=gen.model, attempts=gen.attempts,
        redactions=prepared.redaction.counts_by_type, speaker_source=prepared.speaker_source,
        audio_duration_s=prepared.audio_duration_s, elapsed_s=round(time.time() - t0, 1),
        generation_s=round(t_gen, 1), verifier=rep.method,
    )
    doc = NoteDocument(note=gen.note, segments=prepared.segments, verification=rep, source=source)
    rec = svc.store.create(doc)
    log.info("note %s created: %d items, %s", rec.id, doc.note.n_items(), rep.counts)
    return rec


# ------------------------------------------------------------------ app
def create_app(services: Services | None = None) -> FastAPI:
    svc = services or Services(store=NoteStore(get_settings().db_path))
    app = FastAPI(title="AI Medical Scribe (research demo, synthetic data only)", version="0.4.0")
    app.state.services = svc

    # ---- error mapping: every failure gets a clear status + message
    def _err(status: int, msg: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": msg})

    @app.exception_handler(NoteNotFound)
    async def _nf(_, e):
        return _err(404, str(e))

    @app.exception_handler(NoteLocked)
    async def _locked(_, e):
        return _err(409, str(e))

    @app.exception_handler(NotApproved)
    async def _na(_, e):
        return _err(409, str(e))

    @app.exception_handler(GenerationError)
    async def _gen(_, e):
        return _err(502, f"LLM output could not be validated: {e}")

    @app.exception_handler(httpx.TransportError)
    async def _ollama(_, e):
        return _err(503, f"Ollama is not reachable at {svc.settings.ollama_host} - start it with `ollama serve` ({type(e).__name__})")

    # ---- routes
    @app.get("/health")
    def health():
        return {"status": "ok", "model": svc.settings.model, "verifier": svc.settings.verifier}

    @app.get("/cases")
    def cases(split: str = "valid") -> list[str]:
        from app.loader import load_split

        try:
            return [c.encounter_id for c in load_split(split)]
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))

    @app.post("/process", response_model=NoteView, status_code=201, openapi_extra={
        "requestBody": {"content": {
            "application/json": {"schema": ProcessRequest.model_json_schema()},
            "multipart/form-data": {"schema": {"type": "object", "properties": {
                "audio": {"type": "string", "format": "binary"}}, "required": ["audio"]}},
        }, "required": True}})
    async def process(request: Request):
        from app.stt import AudioValidationError

        ctype = request.headers.get("content-type", "")
        try:
            if ctype.startswith("multipart/form-data"):
                form = await request.form()
                upload = form.get("audio")
                if upload is None or not hasattr(upload, "read"):
                    raise HTTPException(422, "multipart upload needs a file field named 'audio'")
                data = await upload.read()
                if len(data) > svc.settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"audio file larger than {svc.settings.max_upload_mb} MB")
                suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
                with tempfile.TemporaryDirectory() as td:
                    path = Path(td) / f"upload{suffix}"
                    path.write_bytes(data)
                    rec = await run_in_threadpool(_process_upload, svc, path, upload.filename or "audio")
            else:
                try:
                    body = ProcessRequest.model_validate(await request.json())
                except ValidationError as e:
                    raise HTTPException(422, [{"loc": x["loc"], "msg": x["msg"]} for x in e.errors()])
                except ValueError:
                    raise HTTPException(422, "request body must be JSON (or multipart with an 'audio' file)")
                rec = await run_in_threadpool(_process_case, svc, body)
        except AudioValidationError as e:
            raise HTTPException(422, f"invalid audio: {e}")
        except KeyError as e:
            raise HTTPException(404, f"unknown encounter: {e}")
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))
        return _view(rec)

    @app.get("/notes", response_model=list[NoteSummary])
    def notes(status: NoteStatus | None = None):
        return [_summary(r) for r in svc.store.list(status)]

    @app.get("/note/{note_id}", response_model=NoteView)
    def get_note(note_id: int):
        return _view(svc.store.get(note_id))

    @app.put("/note/{note_id}", response_model=UpdateResponse)
    async def update_note(note_id: int, body: UpdateRequest):
        rec = svc.store.get(note_id)
        if rec.status is NoteStatus.approved:
            raise NoteLocked(f"note {note_id} is approved and can no longer be edited")
        note, n_red = _redact_note(body.note)
        report, n_checked = await run_in_threadpool(
            reverify_note, note, rec.doc.segments, rec.doc.note, rec.doc.verification, svc.get_entailer()
            if _needs_entailer(note, rec) else _NoEntailer(), svc.settings)
        new = svc.store.update(note_id, note, report, actor=body.actor)
        return UpdateResponse(**_view(new).model_dump(), rechecked=n_checked, redactions=n_red)

    @app.post("/note/{note_id}/approve", response_model=NoteView)
    def approve(note_id: int, body: ApproveRequest | None = None):
        return _view(svc.store.approve(note_id, actor=(body or ApproveRequest()).actor))

    @app.get("/note/{note_id}/audit", response_model=list[AuditEntry])
    def audit(note_id: int):
        return svc.store.audit(note_id)

    @app.get("/final", response_model=list[NoteSummary])
    def final_list():
        return [_summary(r) for r in svc.store.list(NoteStatus.approved)]

    @app.get("/final/{note_id}", response_model=NoteView)
    def final(note_id: int):
        return _view(svc.store.final(note_id))

    @app.get("/note/{note_id}/audio", responses={200: {"content": {"audio/wav": {}}}})
    async def note_audio(note_id: int):
        rec = svc.store.final(note_id)  # 409 unless approved
        from app.tts import read_note_aloud

        def _speak() -> bytes:
            with tempfile.TemporaryDirectory() as td:
                path = read_note_aloud(rec.doc.note, Path(td) / "note.wav", voices=svc.get_voices())
                return path.read_bytes()

        wav = await run_in_threadpool(_speak)
        svc.store.log(note_id, "read_aloud", {"bytes": len(wav)})
        return Response(content=wav, media_type="audio/wav")

    return app


class _NoEntailer:
    """Placeholder when every line is unchanged - avoids loading the NLI model."""

    name = "unchanged"

    def judge(self, pairs):  # pragma: no cover - never called
        raise AssertionError("no lines need re-verification")


def _needs_entailer(note: SOAPNote, rec) -> bool:
    old = {(s, it.text, tuple(it.evidence), it.unsupported) for s, _, it in rec.doc.note.items()}
    return any((s, it.text, tuple(it.evidence), it.unsupported) not in old for s, _, it in note.items())


def __getattr__(name: str):
    # `uvicorn app.main:app` builds the real app on first access; importing the
    # module (e.g. in tests) has no side effects such as creating data/scribe.db.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
