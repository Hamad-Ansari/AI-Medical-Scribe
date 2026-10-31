"""End-to-end pipeline with redaction enforced *first*.

    audio  --stt-->  segments --redact--> segments' --generate--> note --verify--> report
    text   --load-->  segments --redact--> ...

Everything downstream of :func:`redact_segments` (LLM prompt, logs, database,
UI) only ever sees redacted text. The original transcript is not returned.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.generate import GenerationResult, LLMClient, generate_note
from app.redact import RedactionResult, install_log_redaction, redact_segments
from app.schema import Segment
from app.verify import Entailer, VerificationReport, verify_note

log = logging.getLogger(__name__)
install_log_redaction(logging.getLogger("app"))


@dataclass
class PreparedTranscript:
    segments: list[Segment]  # REDACTED
    redaction: RedactionResult
    source: str  # "text" | "audio"
    audio_duration_s: float | None = None
    speaker_source: str | None = None


@dataclass
class PipelineResult:
    transcript: PreparedTranscript
    generation: GenerationResult
    verification: VerificationReport


def prepare_from_segments(segments: list[Segment]) -> PreparedTranscript:
    if not segments:
        raise ValueError("transcript has no segments")
    red, res = redact_segments(segments)
    log.info("redacted transcript: %d segments, %d replacements", len(red), res.count)
    return PreparedTranscript(segments=red, redaction=res, source="text")


def prepare_from_audio(audio_path: str | Path, reference_turns: list[Segment] | None = None,
                       diarize: bool | None = None) -> PreparedTranscript:
    from app.stt import transcribe

    tr = transcribe(audio_path, reference_turns=reference_turns, diarize=diarize)
    if not tr.segments:
        raise ValueError("no speech recognised in audio")
    red, res = redact_segments(tr.segments)
    log.info("redacted transcript: %d segments, %d replacements", len(red), res.count)
    return PreparedTranscript(segments=red, redaction=res, source="audio",
                              audio_duration_s=tr.duration_s, speaker_source=tr.speaker_source)


def generate_and_verify(prepared: PreparedTranscript, client: LLMClient | None = None,
                        entailer: Entailer | None = None) -> PipelineResult:
    gen = generate_note(prepared.segments, client=client)
    rep = verify_note(gen.note, prepared.segments, entailer=entailer)
    return PipelineResult(transcript=prepared, generation=gen, verification=rep)
