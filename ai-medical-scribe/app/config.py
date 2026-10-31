"""Central configuration. Every setting can be overridden with an env var.

Nothing in here is secret: the whole stack is local and free.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # --- LLM (Ollama) ---
    ollama_host: str = field(default_factory=lambda: _env("SCRIBE_OLLAMA_HOST", "http://localhost:11434"))
    model: str = field(default_factory=lambda: _env("SCRIBE_MODEL", "qwen2.5:7b"))
    temperature: float = field(default_factory=lambda: float(_env("SCRIBE_TEMPERATURE", "0.1")))
    num_ctx: int = field(default_factory=lambda: int(_env("SCRIBE_NUM_CTX", "8192")))
    seed: int = field(default_factory=lambda: int(_env("SCRIBE_SEED", "0")))
    # hard cap on generated tokens - stops runaway/looping JSON on small models
    num_predict: int = field(default_factory=lambda: int(_env("SCRIBE_NUM_PREDICT", "3072")))
    # Ollama defaults are 1.1 / 64. Tiny models (0.5b) loop less with ~1.15 / 256.
    repeat_penalty: float = field(default_factory=lambda: float(_env("SCRIBE_REPEAT_PENALTY", "1.1")))
    repeat_last_n: int = field(default_factory=lambda: int(_env("SCRIBE_REPEAT_LAST_N", "64")))
    # how long Ollama keeps the model in RAM after a call ("0" frees it at once -
    # useful on small machines so the NLI verifier has room)
    keep_alive: str = field(default_factory=lambda: _env("SCRIBE_KEEP_ALIVE", "5m"))
    # prompt batch size; smaller = less RAM in the Ollama runner, slower prompt
    # processing. Unset (0) = Ollama's default. ~2 GB machines: 128
    num_batch: int = field(default_factory=lambda: int(_env("SCRIBE_NUM_BATCH", "0")))
    request_timeout_s: float = field(default_factory=lambda: float(_env("SCRIBE_TIMEOUT_S", "600")))

    # --- Data ---
    data_dir: Path = field(default_factory=lambda: Path(_env("SCRIBE_DATA_DIR", str(PROJECT_ROOT / "data" / "aci-bench"))))
    # Turns longer than this many words are split into sentence-group segments
    # so evidence citations stay precise. 0 disables splitting.
    max_segment_words: int = field(default_factory=lambda: int(_env("SCRIBE_MAX_SEGMENT_WORDS", "50")))

    # --- Verifier (phase 2) ---
    verifier: str = field(default_factory=lambda: _env("SCRIBE_VERIFIER", "nli"))  # "nli" | "ollama"
    nli_model: str = field(default_factory=lambda: _env("SCRIBE_NLI_MODEL", "cross-encoder/nli-deberta-v3-base"))
    # low-RAM alternative: cross-encoder/nli-deberta-v3-xsmall (~300 MB less, see eval/verifier_eval.py)
    judge_model: str = field(default_factory=lambda: _env("SCRIBE_JUDGE_MODEL", _env("SCRIBE_MODEL", "qwen2.5:7b")))
    # entailment >= this -> supported
    entail_threshold: float = field(default_factory=lambda: float(_env("SCRIBE_ENTAIL_THRESHOLD", "0.5")))
    # entailment below this (and not contradicted) -> unsupported; between -> flagged
    weak_threshold: float = field(default_factory=lambda: float(_env("SCRIBE_WEAK_THRESHOLD", "0.2")))
    contradiction_threshold: float = field(default_factory=lambda: float(_env("SCRIBE_CONTRADICTION_THRESHOLD", "0.5")))
    # an item citing more segments than this is flagged as "evidence too broad"
    max_evidence_ok: int = field(default_factory=lambda: int(_env("SCRIBE_MAX_EVIDENCE_OK", "6")))
    # cited segments shorter than this ("yes .", "mm-hmm .") get the previous segment as context
    short_segment_words: int = field(default_factory=lambda: int(_env("SCRIBE_SHORT_SEGMENT_WORDS", "8")))

    # --- Storage / API (phase 4) ---
    db_path: Path = field(default_factory=lambda: Path(_env("SCRIBE_DB", str(PROJECT_ROOT / "data" / "scribe.db"))))
    api_url: str = field(default_factory=lambda: _env("SCRIBE_API_URL", "http://localhost:8000"))  # used by the UI
    max_upload_mb: int = field(default_factory=lambda: int(_env("SCRIBE_MAX_UPLOAD_MB", "50")))
    # ~2 GB RAM machines: free Whisper/Piper after transcription so the LLM and
    # the NLI verifier fit (costs a few seconds of reload on the next audio request)
    low_memory: bool = field(default_factory=lambda: _env("SCRIBE_LOW_MEMORY", "0") == "1")

    # --- Speech (phase 3) ---
    piper_dir: Path = field(default_factory=lambda: Path(_env("SCRIBE_PIPER_DIR", str(PROJECT_ROOT / "models" / "piper"))))
    doctor_voice: str = field(default_factory=lambda: _env("SCRIBE_DOCTOR_VOICE", "en_US-ryan-medium"))
    patient_voice: str = field(default_factory=lambda: _env("SCRIBE_PATIENT_VOICE", "en_US-amy-medium"))
    audio_dir: Path = field(default_factory=lambda: Path(_env("SCRIBE_AUDIO_DIR", str(PROJECT_ROOT / "data" / "audio"))))
    whisper_model: str = field(default_factory=lambda: _env("SCRIBE_WHISPER_MODEL", "base.en"))
    whisper_compute_type: str = field(default_factory=lambda: _env("SCRIBE_WHISPER_COMPUTE", "int8"))
    whisper_beam_size: int = field(default_factory=lambda: int(_env("SCRIBE_WHISPER_BEAM", "1")))
    diarize: bool = field(default_factory=lambda: _env("SCRIBE_DIARIZE", "0") == "1")
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))


def get_settings() -> Settings:
    """Fresh settings each call so tests can monkeypatch env vars."""
    return Settings()
