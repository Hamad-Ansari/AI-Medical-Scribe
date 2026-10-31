"""Speech-to-text with faster-whisper (local, CPU-friendly int8).

Output: numbered segments ``{id, start, end, speaker, text}``.

Speaker labels, in order of preference:
  1. reference turns with timings (ACI-Bench audio synthesised by :mod:`app.tts`):
     every Whisper *word* is assigned to the turn it overlaps most, so a Whisper
     segment that spans a speaker change is split correctly;
  2. pyannote diarization (``diarize=True`` / ``SCRIBE_DIARIZE=1``; needs
     ``pip install pyannote.audio`` and a free ``HF_TOKEN`` with the model terms
     accepted) -> generic labels ``speaker_1``, ``speaker_2`` ...;
  3. otherwise ``unknown``.

Privacy: transcripts are not logged. Redact segments (``app.redact``) before
storing them anywhere.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import Settings, get_settings
from app.loader import _chunk_turn
from app.schema import Segment

log = logging.getLogger(__name__)
SAMPLE_RATE = 16_000
MIN_DURATION_S = 0.3
SILENCE_RMS = 1e-4


class AudioValidationError(ValueError):
    """Raised for missing, empty, unreadable, too-short or silent audio."""


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str
    turn_id: int | None = None


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    duration_s: float
    model: str
    speaker_source: str  # "reference" | "pyannote" | "none"
    elapsed_s: float = 0.0
    words: list[Word] = field(default_factory=list, repr=False)


# ------------------------------------------------------------------ audio
def load_audio(path: str | Path) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32, with clear validation errors."""
    p = Path(path)
    if not p.exists():
        raise AudioValidationError(f"audio file not found: {p}")
    if p.stat().st_size == 0:
        raise AudioValidationError("audio file is empty (0 bytes)")
    try:
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(p), sampling_rate=SAMPLE_RATE)
    except Exception as e:  # PyAV raises many types for corrupt input
        raise AudioValidationError(f"could not decode audio ({type(e).__name__}); is it a valid audio file?") from e
    return validate_samples(audio)


def validate_samples(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        raise AudioValidationError("audio contains no samples")
    dur = audio.size / SAMPLE_RATE
    if dur < MIN_DURATION_S:
        raise AudioValidationError(f"audio too short ({dur:.2f}s < {MIN_DURATION_S}s)")
    if float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < SILENCE_RMS:
        raise AudioValidationError("audio is silent")
    return audio.astype(np.float32)


# ------------------------------------------------------------------ speakers
def assign_speakers(words: list[Word], turns: list[SpeakerTurn]) -> list[SpeakerTurn | None]:
    """For each word return the turn with the largest time overlap.

    Words that overlap no turn (e.g. in a pause) go to the turn whose
    boundary is nearest to the word's midpoint.
    """
    if not turns:
        return [None] * len(words)
    out: list[SpeakerTurn | None] = []
    for w in words:
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(w.end, t.end) - max(w.start, t.start)
            if ov > best_ov:
                best, best_ov = t, ov
        if best is None:
            mid = (w.start + w.end) / 2
            best = min(turns, key=lambda t: 0 if t.start <= mid <= t.end else min(abs(mid - t.start), abs(mid - t.end)))
        out.append(best)
    return out


def words_to_segments(words: list[Word], owners: list[SpeakerTurn | None], max_words: int = 50) -> list[Segment]:
    """Group consecutive words with the same owner turn into numbered segments."""
    groups: list[tuple[SpeakerTurn | None, list[Word]]] = []
    for w, o in zip(words, owners):
        key = (o.speaker, o.turn_id) if o else None
        if groups and ((groups[-1][0].speaker, groups[-1][0].turn_id) if groups[-1][0] else None) == key:
            groups[-1][1].append(w)
        else:
            groups.append((o, [w]))
    segments: list[Segment] = []
    for owner, ws in groups:
        text = "".join(w.text for w in ws).strip()
        if not text:
            continue
        speaker = owner.speaker if owner else "unknown"
        # split very long groups at sentence ends, keeping word timings
        chunks = _chunk_turn(text, max_words)
        if len(chunks) == 1:
            segments.append(Segment(id=len(segments) + 1, start=round(ws[0].start, 2), end=round(ws[-1].end, 2), speaker=speaker, text=text))
            continue
        i = 0
        for ch in chunks:
            n = len(ch.split())
            part = ws[i : i + n] or ws[-1:]
            segments.append(Segment(id=len(segments) + 1, start=round(part[0].start, 2), end=round(part[-1].end, 2), speaker=speaker, text=ch))
            i += n
    return segments


def diarize_pyannote(path: str | Path, settings: Settings | None = None) -> list[SpeakerTurn]:
    """Optional pyannote diarization -> generic speaker turns."""
    settings = settings or get_settings()
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise RuntimeError("diarization requested but pyannote.audio is not installed: pip install pyannote.audio") from e
    if not settings.hf_token:
        raise RuntimeError("diarization needs HF_TOKEN (free) with pyannote/speaker-diarization-3.1 terms accepted")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=settings.hf_token)
    diar = pipe(str(path))
    labels: dict[str, str] = {}
    turns = []
    for seg, _, lab in diar.itertracks(yield_label=True):
        labels.setdefault(lab, f"speaker_{len(labels) + 1}")  # order of first appearance
        turns.append(SpeakerTurn(seg.start, seg.end, labels[lab], len(turns)))
    return turns


# ------------------------------------------------------------------ whisper
@lru_cache(maxsize=2)
def _whisper(model_name: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device="cpu", compute_type=compute_type)


def transcribe(
    audio_path: str | Path,
    reference_turns: list[Segment] | None = None,
    diarize: bool | None = None,
    settings: Settings | None = None,
) -> TranscriptionResult:
    """Transcribe audio into numbered, speaker-labelled segments."""
    settings = settings or get_settings()
    audio = load_audio(audio_path)
    diarize = settings.diarize if diarize is None else diarize
    t0 = time.perf_counter()
    model = _whisper(settings.whisper_model, settings.whisper_compute_type)
    seg_iter, info = model.transcribe(
        audio,
        language="en",
        beam_size=settings.whisper_beam_size,
        word_timestamps=True,
        condition_on_previous_text=False,  # avoids repetition loops on long audio
        temperature=0.0,  # no sampling fallback -> deterministic, reproducible WER
        vad_filter=False,  # keep timings aligned with reference turns
    )
    words = [Word(w.start, w.end, w.word) for s in seg_iter for w in (s.words or [])]

    if reference_turns:
        if any(t.start is None or t.end is None for t in reference_turns):
            raise ValueError("reference_turns must have start/end times (use tts.synthesize_dialogue)")
        turns = [SpeakerTurn(t.start, t.end, t.speaker, t.id) for t in reference_turns]
        source = "reference"
    elif diarize:
        turns = diarize_pyannote(audio_path, settings)
        source = "pyannote"
    else:
        turns, source = [], "none"

    owners = assign_speakers(words, turns)
    segments = words_to_segments(words, owners, settings.max_segment_words)
    elapsed = time.perf_counter() - t0
    log.info("transcribed %.1fs audio -> %d segments in %.1fs (%s speakers)", info.duration, len(segments), elapsed, source)
    return TranscriptionResult(segments=segments, duration_s=float(info.duration), model=settings.whisper_model,
                               speaker_source=source, elapsed_s=elapsed, words=words)
