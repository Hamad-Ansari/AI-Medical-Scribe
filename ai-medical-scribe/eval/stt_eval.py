"""STT evaluation: ACI-Bench dialogue -> Piper audio -> faster-whisper -> WER.

    python eval/stt_eval.py --n 3 --max-turns 40 [--model base.en]

Metrics per case:
  wer            word error rate of the whole transcript (jiwer, normalised text)
  turn_attr_wer  per-turn WER after each Whisper word is attributed to a reference
                 turn by time overlap - a word put in the wrong turn (and hence
                 possibly the wrong speaker) costs twice, so this measures
                 speaker/turn assignment as well as recognition
  num_recall     share of numbers in the reference (doses, vitals, durations)
                 that appear in the transcript - WER can hide a dropped "1000 mg"
  rtf            real-time factor (processing time / audio duration)
The summary also reports pooled WER (total errors / total reference words).

Audio is synthesised with deterministic Piper (no sampling noise) so results are
reproducible. Synthetic audio is far cleaner than a real clinic room: treat the
WER here as a lower bound, not a deployment estimate.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import re
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")


def run(n: int = 3, max_turns: int = 40, split: str = "valid", out_csv: Path | None = None) -> list[dict]:
    import jiwer

    from app.loader import load_split
    from app.stt import SpeakerTurn, assign_speakers, transcribe
    from app.textnorm import normalize_for_wer
    from app.tts import synthesize_dialogue

    rows = []
    for case in load_split(split, limit=n):
        # use whole turns (not 50-word chunks) as the unit of speaker attribution
        turns = case.segments[:max_turns] if max_turns else case.segments
        with tempfile.TemporaryDirectory() as td:
            audio = synthesize_dialogue(turns, Path(td) / f"{case.encounter_id}.wav", deterministic=True)
            res = transcribe(audio.path, reference_turns=audio.turns)
        ref = normalize_for_wer(" ".join(t.text for t in audio.turns))
        hyp = normalize_for_wer(" ".join(s.text for s in res.segments))
        owners = assign_speakers(res.words, [SpeakerTurn(t.start, t.end, t.speaker, t.id) for t in audio.turns])
        by_turn: dict[int, list[str]] = {t.id: [] for t in audio.turns}
        for w, o in zip(res.words, owners):
            if o is not None:
                by_turn[o.turn_id].append(w.text)
        turn_ref = " | ".join(normalize_for_wer(t.text) for t in audio.turns)
        turn_hyp = " | ".join(normalize_for_wer("".join(by_turn[t.id])) for t in audio.turns)
        o = jiwer.process_words(ref, hyp)
        ref_nums, hyp_nums = Counter(_NUM.findall(ref)), Counter(_NUM.findall(hyp))
        rows.append({
            "encounter_id": case.encounter_id,
            "turns": len(audio.turns),
            "audio_s": round(audio.duration_s, 1),
            "ref_words": len(ref.split()),
            "errors": o.substitutions + o.deletions + o.insertions,
            "wer": round(o.wer, 4),
            "turn_attr_wer": round(_turnwise_wer(turn_ref, turn_hyp), 4),
            "num_recall": round(sum((ref_nums & hyp_nums).values()) / max(sum(ref_nums.values()), 1), 4),
            "rtf": round(res.elapsed_s / audio.duration_s, 3),
            "model": res.model,
        })
        print(rows[-1], flush=True)
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def _turnwise_wer(turn_ref: str, turn_hyp: str) -> float:
    """Sum of per-turn edit distances / reference words: a word in the wrong turn costs twice."""
    import jiwer

    errs = words = 0
    for r, h in zip(turn_ref.split(" | "), turn_hyp.split(" | ")):
        nr = len(r.split())
        words += nr
        if nr == 0:
            errs += len(h.split())
            continue
        o = jiwer.process_words(r, h if h.strip() else "")
        errs += o.substitutions + o.deletions + o.insertions
    return errs / max(words, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=40, help="0 = whole dialogue")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--model", help="whisper model (overrides SCRIBE_WHISPER_MODEL)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "stt_wer.csv"))
    a = ap.parse_args()
    if a.model:
        os.environ["SCRIBE_WHISPER_MODEL"] = a.model
    rows = run(a.n, a.max_turns, a.split, Path(a.out))
    mean = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    print(f"\n{rows[0]['model']}  cases={len(rows)}  audio={sum(r['audio_s'] for r in rows):.0f}s")
    pooled = sum(r["errors"] for r in rows) / sum(r["ref_words"] for r in rows)
    print(f"  pooled WER {pooled:.1%}   mean WER {mean('wer'):.1%}   turn-attributed WER {mean('turn_attr_wer'):.1%}"
          f"   number recall {mean('num_recall'):.1%}   RTF {mean('rtf'):.2f}")
