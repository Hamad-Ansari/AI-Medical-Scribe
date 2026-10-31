"""Phase-2 smoke test: ACI-Bench case -> SOAP note (Ollama) -> verification report.

    python scripts/smoke_verify.py --split valid --index 0
    python scripts/smoke_verify.py --verifier ollama          # LLM judge instead of NLI

Prints to the terminal only - nothing is written to disk (redaction arrives in phase 3).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLORS = {"supported": "\033[32m", "flagged": "\033[33m", "unsupported": "\033[31m"}
RESET = "\033[0m"
ICON = {"supported": "✓", "flagged": "!", "unsupported": "✗"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--model", help="generation model (overrides SCRIBE_MODEL)")
    ap.add_argument("--verifier", choices=["nli", "ollama"], help="overrides SCRIBE_VERIFIER")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()
    if a.model:
        os.environ["SCRIBE_MODEL"] = a.model
        os.environ.setdefault("SCRIBE_JUDGE_MODEL", a.model)
    if a.verifier:
        os.environ["SCRIBE_VERIFIER"] = a.verifier
    logging.basicConfig(level=logging.WARNING)

    from app.generate import generate_note
    from app.loader import load_split
    from app.verify import verify_note

    case = load_split(a.split, limit=a.index + 1)[a.index]
    by_id = {s.id: s for s in case.segments}
    print(f"case {case.encounter_id}: {len(case.segments)} segments")
    gen = generate_note(case.segments)
    print(f"generated {gen.note.n_items()} items with {gen.model} in {gen.duration_s:.0f}s (attempts={gen.attempts})")
    t = time.perf_counter()
    rep = verify_note(gen.note, case.segments)
    print(f"verified with {rep.method} in {time.perf_counter() - t:.1f}s -> {rep.counts}\n")

    for iv in rep.items:
        c = "" if a.no_color else COLORS[iv.status.value]
        r = "" if a.no_color else RESET
        ent = f" ent={iv.entailment:.2f}" if iv.entailment is not None else ""
        print(f"{c}{ICON[iv.status.value]} [{iv.section[:1].upper()}] {iv.text}  <- {iv.evidence}{ent}{r}")
        for reason in iv.reasons:
            print(f"      - {reason}")
        if iv.suggested_evidence:
            s0 = by_id[iv.suggested_evidence[0]]
            print(f"      ? suggested evidence {iv.suggested_evidence}: [{s0.id}] {s0.speaker}: {s0.text[:90]}")


if __name__ == "__main__":
    main()
