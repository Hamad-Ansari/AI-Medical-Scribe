"""Phase-1 smoke test: one ACI-Bench case -> SOAP note via a real Ollama model.

    python scripts/smoke_generate.py --split valid --index 0 --model qwen2.5:7b

Prints the note to the terminal only (nothing is written to disk).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--model", default=None, help="overrides SCRIBE_MODEL")
    ap.add_argument("--max-segments", type=int, default=0, help="truncate transcript (for tiny CPU boxes)")
    args = ap.parse_args()
    if args.model:
        os.environ["SCRIBE_MODEL"] = args.model
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from app.generate import generate_note
    from app.loader import load_split

    case = load_split(args.split, limit=args.index + 1)[args.index]
    segs = case.segments[: args.max_segments] if args.max_segments else case.segments
    print(f"case {case.encounter_id}: {len(segs)} segments")
    res = generate_note(segs)
    print(f"model={res.model} attempts={res.attempts} time={res.duration_s:.1f}s items={res.note.n_items()}")
    valid_ids = {s.id for s in segs}
    for sec, _, it in res.note.items():
        bad = [e for e in it.evidence if e not in valid_ids]
        flag = " [UNSUPPORTED]" if it.unsupported else ""
        flag += f" [BAD IDS {bad}]" if bad else ""
        flag += " [NO EVIDENCE]" if not it.evidence else ""
        print(f"  {sec[0].upper()} | {it.text}  <- {it.evidence}{flag}")


if __name__ == "__main__":
    main()
