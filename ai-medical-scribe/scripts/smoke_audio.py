"""Phase 3 smoke test: ACI dialogue -> Piper audio -> faster-whisper -> redaction.

    python scripts/smoke_audio.py --split valid --index 0 --turns 16 [--generate]

Prints the redacted, speaker-labelled transcript (what the LLM / DB / logs see),
the redaction counts, WER and timings. --generate also runs Ollama + verifier.
The synthesised WAV contains the synthetic ACI names, exactly like the CSV it
came from; it is input data and is written to data/audio/ (git-ignored).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--turns", type=int, default=16, help="first N turns (0 = whole dialogue)")
    ap.add_argument("--generate", action="store_true", help="also run Ollama generation + verifier")
    ap.add_argument("--natural", action="store_true", help="Piper sampling noise on (varies run to run)")
    a = ap.parse_args()

    import jiwer

    from app.config import get_settings
    from app.loader import load_split
    from app.pipeline import generate_and_verify, prepare_from_audio
    from app.textnorm import normalize_for_wer
    from app.tts import synthesize_dialogue

    case = load_split(a.split, limit=a.index + 1)[a.index]
    turns = case.segments[: a.turns] if a.turns else case.segments
    out = get_settings().audio_dir / f"{case.encounter_id}.wav"

    t = time.time()
    audio = synthesize_dialogue(turns, out, deterministic=not a.natural)
    print(f"[tts] {case.encounter_id}: {len(turns)} turns -> {audio.duration_s:.1f}s audio in {time.time() - t:.1f}s  ({out})")

    t = time.time()
    prepared = prepare_from_audio(audio.path, reference_turns=audio.turns)
    print(f"[stt+redact] {len(prepared.segments)} segments in {time.time() - t:.1f}s, speakers from {prepared.speaker_source}")
    print(f"[redact] {prepared.redaction.count} replacements {prepared.redaction.counts_by_type}\n")
    for s in prepared.segments:
        print(f"  [{s.id:>2}] {s.start:6.1f}-{s.end:6.1f}  {s.speaker:<13} {s.text}")

    ref = normalize_for_wer(" ".join(x.text for x in turns))
    hyp = normalize_for_wer(" ".join(x.text for x in prepared.segments))
    print(f"\n[wer] {jiwer.wer(ref, hyp):.1%}  (vs. ACI text; placeholders count as errors here)")

    if a.generate:
        t = time.time()
        res = generate_and_verify(prepared)
        print(f"\n[generate+verify] {time.time() - t:.0f}s")
        for it in res.verification.items:
            print(f"  {it.status:<11} {it.section:<10} {it.text}  {it.evidence}")


if __name__ == "__main__":
    main()
