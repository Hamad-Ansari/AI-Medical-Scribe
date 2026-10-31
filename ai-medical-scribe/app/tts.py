"""Text-to-speech with Piper (local, free). Two voices: doctor and patient.

* :func:`synthesize_dialogue` - turn a text-only ACI-Bench dialogue into one WAV
  file and return the exact start/end time of every turn. Those timings let the
  STT step attach ACI-Bench speaker labels to Whisper's words.
* :func:`read_note_aloud` - read a SOAP note section by section.

Note on privacy: the synthesised consultation audio is *input* data - it is
the stand-in for a recording and contains the same synthetic names as the
ACI-Bench CSV it came from. Pipeline outputs are always produced from the
redacted transcript.
"""
from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from app.config import Settings, get_settings
from app.schema import SECTIONS, Segment, SOAPNote

PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
ROLE_OF_SPEAKER = {"doctor": "doctor", "patient": "patient", "patient_guest": "patient"}


class TTSError(RuntimeError):
    pass


class Voice(Protocol):
    sample_rate: int

    def speak(self, text: str) -> np.ndarray:
        """Return int16 mono samples."""


class PiperVoiceAdapter:
    def __init__(self, model_path: Path):
        from piper import PiperVoice

        if not model_path.exists():
            raise TTSError(f"Piper voice not found: {model_path} - run `python scripts/download_voices.py`")
        self._voice = PiperVoice.load(str(model_path))
        # onnxruntime's CPU arena keeps the peak allocation of the longest
        # sentence forever (~1 GB for two voices on a long consultation).
        # Re-create the session with the arena off: same speed, flat memory.
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        self._voice.session = onnxruntime.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sample_rate = int(self._voice.config.sample_rate)

    def speak(self, text: str, deterministic: bool = False) -> np.ndarray:
        """``deterministic`` turns off VITS sampling noise: identical audio on every
        run (slightly flatter prosody). Used by the STT evaluation for reproducible WER."""
        cfg = None
        if deterministic:
            from piper import SynthesisConfig

            cfg = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)
        chunks = [c.audio_int16_array for c in self._voice.synthesize(text, syn_config=cfg)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


@lru_cache(maxsize=4)
def _load_voice(path: str) -> PiperVoiceAdapter:
    return PiperVoiceAdapter(Path(path))


def voice_path(name: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.piper_dir / f"{name}.onnx"


def voice_url(name: str) -> tuple[str, str]:
    """HF URLs for a voice name like 'en_US-ryan-medium'."""
    lang_region, speaker, quality = name.split("-")
    base = f"{PIPER_VOICES_URL}{lang_region.split('_')[0]}/{lang_region}/{speaker}/{quality}/{name}"
    return base + ".onnx", base + ".onnx.json"


def get_voices(settings: Settings | None = None) -> dict[str, Voice]:
    settings = settings or get_settings()
    return {
        "doctor": _load_voice(str(voice_path(settings.doctor_voice, settings))),
        "patient": _load_voice(str(voice_path(settings.patient_voice, settings))),
    }


# ------------------------------------------------------------------ text prep
_DETOK = [
    (re.compile(r"\s+([,.?!;:%])"), r"\1"),
    (re.compile(r"\b(\w+) n't\b"), r"\1n't"),
    (re.compile(r"\b(gon|wan|got) na\b"), r"\1na"),
    (re.compile(r"\s+'(s|re|ve|ll|d|m)\b"), r"'\1"),
    (re.compile(r"\s{2,}"), " "),
]
_PLACEHOLDER_SPEECH = {"[NAME]": "the patient", "[PHONE]": "a phone number", "[EMAIL]": "an email address", "[ID]": "an I D"}


def speakable(text: str) -> str:
    """ACI-Bench tokenised text -> natural text for TTS ("do n't ." -> "don't.")."""
    for rx, rep in _DETOK:
        text = rx.sub(rep, text)
    for ph, spoken in _PLACEHOLDER_SPEECH.items():
        text = text.replace(ph, spoken)
    # espeak only starts a new sentence before a capital letter; all-lowercase
    # ACI text becomes one huge run-on utterance that the VITS model rushes,
    # garbles or truncates (and that balloons onnxruntime memory).
    # Piper can't say "mm-hmm": Whisper hears "And my mom." / "I don't know." -
    # invented content. "uh-huh" survives the round trip as a harmless filler.
    text = _BACKCHANNEL.sub("uh-huh", text)
    text = re.sub(r"\bm+-m+\b", "uh-uh", text, flags=re.I)  # "mm-mm" = no; keep it a negative
    text = _STUTTER.sub(r"\2", text)  # "y-you" -> "you"
    text = _TRAIL_HYPHEN.sub(r"\1", text)  # "under- out" -> "under out"
    text = _ELLIPSIS.sub(", ", text).replace(" ,", ",")  # "um... um..." -> pauses, not stops
    text = _PRONOUN_I.sub(lambda m: m.group(0).upper(), text)
    text = _SENT_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    return text.strip()


_BACKCHANNEL = re.compile(r"\b(?:m+-?h+m+|m+\s+h+m+|mhm|mmm+)\b", re.I)  # never "5 mm" or "mm-mm" (= no)
_STUTTER = re.compile(r"\b(?!(?:uh|mm|um)-)([a-z]{1,3})-(?:\1-)*(\1[a-z]*)\b", re.I)  # true repeats only
_TRAIL_HYPHEN = re.compile(r"\b([a-z]+)-(?=\s)", re.I)
_ELLIPSIS = re.compile(r"\s*(?:\.\s*){2,}")
_PRONOUN_I = re.compile(r"\bi(?=\b|')")
_SENT_START = re.compile(r"(^|[.!?]\s+)([a-z])")


# ------------------------------------------------------------------ audio io
def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.astype(np.int16).tobytes())
    return path


@dataclass
class DialogueAudio:
    path: Path
    sample_rate: int
    duration_s: float
    turns: list[Segment]  # same ids/speakers/text as input, with start/end filled


def _as_segments(dialogue) -> list[Segment]:
    if hasattr(dialogue, "segments"):  # ACICase
        return list(dialogue.segments)
    out = []
    for i, d in enumerate(dialogue):
        if isinstance(d, Segment):
            out.append(d)
        else:
            spk, text = d
            out.append(Segment(id=i + 1, speaker=spk, text=text))
    return out


def synthesize_dialogue(
    dialogue,
    out_path: str | Path,
    voices: dict[str, Voice] | None = None,
    pause_s: float = 0.35,
    deterministic: bool = False,
) -> DialogueAudio:
    """Synthesise a dialogue to one WAV and return per-turn timings.

    ``dialogue``: an ACICase, a list of Segments, or a list of (speaker, text).
    """
    segs = _as_segments(dialogue)
    if not segs:
        raise TTSError("dialogue is empty")
    voices = voices or get_voices()
    rates = {v.sample_rate for v in voices.values()}
    if len(rates) != 1:
        raise TTSError(f"voices must share a sample rate, got {rates}")
    sr = rates.pop()
    gap = np.zeros(int(pause_s * sr), dtype=np.int16)
    parts: list[np.ndarray] = []
    turns: list[Segment] = []
    t = 0.0
    for seg in segs:
        role = ROLE_OF_SPEAKER.get(seg.speaker, "patient")
        text = speakable(seg.text)
        audio = voices[role].speak(text, deterministic=True) if deterministic else voices[role].speak(text)
        dur = len(audio) / sr
        turns.append(seg.model_copy(update={"start": round(t, 3), "end": round(t + dur, 3)}))
        parts += [audio, gap]
        t += dur + pause_s
    samples = np.concatenate(parts)
    path = write_wav(Path(out_path), samples, sr)
    return DialogueAudio(path=path, sample_rate=sr, duration_s=len(samples) / sr, turns=turns)


def note_to_speech_text(note: SOAPNote) -> str:
    titles = {"subjective": "Subjective", "objective": "Objective", "assessment": "Assessment", "plan": "Plan"}
    parts = []
    for sec in SECTIONS:
        items = getattr(note, sec)
        if items:
            lines = ". ".join(speakable(it.text).rstrip(".") for it in items)
            parts.append(f"{titles[sec]}. {lines}.")
        else:
            parts.append(f"{titles[sec]}. Nothing documented.")
    return " ".join(parts)


def read_note_aloud(note: SOAPNote, out_path: str | Path, voices: dict[str, Voice] | None = None, role: str = "doctor") -> Path:
    """Read a SOAP note aloud (doctor voice) into a WAV file.

    The approval gate lives in the API/UI: only approved notes are passed here.
    """
    voices = voices or get_voices()
    v = voices[role]
    pause = np.zeros(int(0.4 * v.sample_rate), dtype=np.int16)
    chunks: list[np.ndarray] = []
    for sentence in re.split(r"(?<=\.)\s+", note_to_speech_text(note)):
        chunks += [v.speak(sentence), pause]
    return write_wav(Path(out_path), np.concatenate(chunks), v.sample_rate)


def iter_voice_files(settings: Settings | None = None) -> Iterable[tuple[str, Path]]:
    settings = settings or get_settings()
    for name in {settings.doctor_voice, settings.patient_voice}:
        onnx, js = voice_url(name)
        yield onnx, settings.piper_dir / f"{name}.onnx"
        yield js, settings.piper_dir / f"{name}.onnx.json"
