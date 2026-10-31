"""SOAP note generation with a local Ollama model and structured outputs.

Pipeline contract
-----------------
input : numbered, *already redacted* transcript segments
output: a validated :class:`SOAPNote` where every item cites segment IDs

Privacy: this module never logs transcript or note text - only counts,
timings and error types.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.schema import MAX_EVIDENCE_PER_ITEM, SECTIONS, NoteItem, Segment, SOAPNote, segments_to_prompt, soap_json_schema

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a careful clinical documentation assistant. You write a SOAP note from a
doctor-patient conversation transcript. The note will be checked line by line
by a doctor, so faithfulness matters more than completeness or style.

The transcript is a list of numbered segments: "[ID] SPEAKER: text".

RULES (follow all of them):
1. Use ONLY information stated in the numbered transcript segments. Do not use
   outside medical knowledge to add diagnoses, doses, findings or plans.
2. Every item must cite the ID(s) of the segment(s) that support it in
   "evidence". Cite the smallest set of segments that fully supports the item.
3. If you want to write something the transcript does not clearly support,
   set "unsupported": true instead of guessing. Never fabricate numbers,
   medication doses, test results or exam findings.
4. Keep items short and atomic: one fact per item, no more than ~20 words.
5. Do not include names, phone numbers, emails or ID numbers. Refer to people
   as "the patient", "the doctor", "the patient's wife", etc.
6. Section definitions:
   - subjective: what the patient reports - chief complaint, symptoms and their
     history, relevant past medical/surgical/family/social history, current
     medications, allergies, review of systems.
   - objective: what the clinician observes or measures - vital signs, physical
     exam findings, lab/imaging results stated in the conversation.
   - assessment: diagnoses or clinical impressions the doctor states.
   - plan: tests ordered, treatments, medication changes, referrals,
     patient education, follow-up.
   A section may be an empty list if the transcript has nothing for it.
7. Output only JSON with keys subjective, objective, assessment, plan. Each is a
   list of objects: {"text": "<short fact>", "evidence": [<segment ids>], "unsupported": false}.
"""

USER_TEMPLATE = """\
TRANSCRIPT ({n} segments, IDs {first}-{last}):
{transcript}

Write the SOAP note as JSON now. Cite segment IDs for every item."""

RETRY_TEMPLATE = """\
Your previous answer could not be accepted: {error}
Return ONLY a JSON object with exactly the keys subjective, objective,
assessment, plan. Each value is a list of {{"text": str, "evidence": [int],
"unsupported": bool}}. Follow all the rules above."""


class GenerationError(RuntimeError):
    """Raised when the model fails to produce a valid note after retrying."""


class LLMClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, format: dict[str, Any] | None, options: dict[str, Any]) -> str:
        """Return the assistant message content."""


class OllamaClient:
    """Minimal client for Ollama's ``POST /api/chat`` (non-streaming)."""

    def __init__(self, settings: Settings | None = None, transport: httpx.BaseTransport | None = None):
        self.settings = settings or get_settings()
        self._client = httpx.Client(
            base_url=self.settings.ollama_host,
            timeout=self.settings.request_timeout_s,
            transport=transport,
        )

    def chat(self, messages, *, format=None, options=None, model: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.settings.keep_alive,
            "options": options or {},
        }
        if format is not None:
            payload["format"] = format
        try:
            resp = self._client.post("/api/chat", json=payload)
        except httpx.ConnectError as e:
            raise GenerationError(
                f"cannot reach Ollama at {self.settings.ollama_host} - is `ollama serve` running?"
            ) from e
        except httpx.TimeoutException as e:
            raise GenerationError(
                f"Ollama timed out after {self.settings.request_timeout_s:.0f}s "
                "(raise SCRIBE_TIMEOUT_S or use a smaller model / GPU)"
            ) from e
        if resp.status_code == 404:
            raise GenerationError(f"model {payload['model']!r} not found - run `ollama pull {payload['model']}`")
        if resp.status_code >= 400:
            raise GenerationError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()["message"]["content"]


@dataclass
class GenerationResult:
    note: SOAPNote
    attempts: int
    model: str
    duration_s: float
    raw_outputs: list[str] = field(default_factory=list, repr=False)


def build_messages(segments: list[Segment]) -> list[dict[str, str]]:
    transcript = segments_to_prompt(segments)
    user = USER_TEMPLATE.format(
        n=len(segments), first=segments[0].id, last=segments[-1].id, transcript=transcript
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_note(raw: str) -> SOAPNote:
    """Parse model output into a SOAPNote (raises ValueError / ValidationError)."""
    raw = raw.strip()
    if raw.startswith("```"):  # tolerate fenced output from non-grammar backends
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    data = json.loads(raw)
    return SOAPNote.model_validate(data)


def dedupe_items(note: SOAPNote) -> tuple[SOAPNote, int]:
    """Drop exact-duplicate items (same section, case-insensitive text).

    Small models sometimes repeat lines; duplicates add nothing for the doctor.
    The first occurrence keeps its evidence merged with the duplicates'.
    """
    data: dict[str, list] = {}
    dropped = 0
    for sec in SECTIONS:
        seen: dict[str, NoteItem] = {}
        for it in getattr(note, sec):
            key = " ".join(it.text.lower().split())
            if key in seen:
                first = seen[key]
                first.evidence = list(dict.fromkeys(first.evidence + it.evidence))[:MAX_EVIDENCE_PER_ITEM]
                first.unsupported = first.unsupported or it.unsupported
                dropped += 1
            else:
                seen[key] = it.model_copy(deep=True)
        data[sec] = list(seen.values())
    return SOAPNote(**data), dropped


def _short_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        errs = e.errors()[:5]
        return "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in errs)
    return f"{type(e).__name__}: {str(e)[:200]}"


def generate_note(
    segments: list[Segment],
    client: LLMClient | None = None,
    settings: Settings | None = None,
    max_attempts: int = 2,
) -> GenerationResult:
    """Generate and validate a SOAP note. Retries once on invalid output."""
    if not segments:
        raise ValueError("cannot generate a note from an empty transcript")
    settings = settings or get_settings()
    client = client or OllamaClient(settings)
    options = {
        "temperature": settings.temperature,
        "num_ctx": settings.num_ctx,
        "num_predict": settings.num_predict,
        "repeat_penalty": settings.repeat_penalty,
        "repeat_last_n": settings.repeat_last_n,
        "seed": settings.seed,
    }
    if settings.num_batch > 0:
        options["num_batch"] = settings.num_batch
    schema = soap_json_schema()
    messages = build_messages(segments)

    t0 = time.perf_counter()
    raws: list[str] = []
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        raw = client.chat(messages, format=schema, options=options)
        raws.append(raw)
        try:
            note = parse_note(raw)
        except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
            last_err = e
            log.warning("generation attempt %d invalid (%s)", attempt, type(e).__name__)
            messages = messages + [
                {"role": "assistant", "content": raw[:4000]},
                {"role": "user", "content": RETRY_TEMPLATE.format(error=_short_error(e))},
            ]
            continue
        note, n_dupes = dedupe_items(note)
        dur = time.perf_counter() - t0
        log.info("generated note: %d items (%d duplicates dropped), attempt %d, %.1fs", note.n_items(), n_dupes, attempt, dur)
        return GenerationResult(note=note, attempts=attempt, model=settings.model, duration_s=dur, raw_outputs=raws)

    raise GenerationError(f"model output invalid after {max_attempts} attempts: {_short_error(last_err)}")
