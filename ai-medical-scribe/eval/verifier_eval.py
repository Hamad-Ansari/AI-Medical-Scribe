"""Evaluate the faithfulness verifier on a hand-labelled synthetic gold set.

    python eval/verifier_eval.py                      # NLI backend (default)
    python eval/verifier_eval.py --backend ollama     # LLM judge (needs ollama serve)

Metrics (positive class = hallucinated line):
  catch rate        hallucinations NOT shown green (red or amber)  - higher is better
  red catch rate    hallucinations shown red                        - higher is better
  false-alarm rate  faithful lines NOT shown green                  - lower is better
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
GOLD = ROOT / "tests" / "data" / "verifier_gold.jsonl"


def load_gold(path: Path = GOLD) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(backend: str = "nli", path: Path = GOLD, verbose: bool = False) -> dict:
    os.environ["SCRIBE_VERIFIER"] = backend
    from app.schema import SOAPNote, Segment
    from app.verify import Status, get_entailer, verify_note

    entailer = get_entailer()
    rows = []
    for case in load_gold(path):
        segs = [Segment(id=i + 1, **s) for i, s in enumerate(case["segments"])]
        note = {"subjective": [], "objective": [], "assessment": [], "plan": []}
        note[case["section"]] = [{"text": case["text"], "evidence": [s.id for s in segs], "unsupported": False}]
        iv = verify_note(SOAPNote.model_validate(note), segs, entailer=entailer, suggest=False).items[0]
        rows.append({**case, "status": iv.status.value, "entailment": iv.entailment, "reason": "; ".join(iv.reasons)})

    hall = [r for r in rows if r["label"] == "unsupported"]
    true = [r for r in rows if r["label"] == "supported"]
    m = {
        "backend": entailer.name,
        "n": len(rows),
        "catch_rate": sum(r["status"] != Status.supported for r in hall) / len(hall),
        "red_catch_rate": sum(r["status"] == Status.unsupported for r in hall) / len(hall),
        "false_alarm_rate": sum(r["status"] != Status.supported for r in true) / len(true),
        "true_shown_red": sum(r["status"] == Status.unsupported for r in true) / len(true),
    }
    if verbose:
        for r in rows:
            ok = (r["status"] == "supported") == (r["label"] == "supported")
            ent = f"{r['entailment']:.2f}" if r["entailment"] is not None else "  - "
            print(f"{'✓' if ok else '✗'} gold={r['label'][:5]:5s} got={r['status']:11s} ent={ent} | {r['section'][:4]} | {r['text']}")
    return {"metrics": m, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="nli", choices=["nli", "ollama"])
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    res = run(a.backend, verbose=not a.quiet)
    m = res["metrics"]
    print(f"\n{m['backend']}  n={m['n']}")
    print(f"  hallucination catch rate (red+amber): {m['catch_rate']:.0%}")
    print(f"  hallucination red rate              : {m['red_catch_rate']:.0%}")
    print(f"  false-alarm rate (true not green)   : {m['false_alarm_rate']:.0%}")
    print(f"  true facts shown red                : {m['true_shown_red']:.0%}")
