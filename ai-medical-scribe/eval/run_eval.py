"""Evaluate generated SOAP notes against ACI-Bench reference notes.

    python eval/run_eval.py --models qwen2.5:7b --split valid --n 20
    python eval/run_eval.py --models qwen2.5:0.5b,qwen2.5:1.5b --stage generate   # LLM only
    python eval/run_eval.py --models qwen2.5:0.5b,qwen2.5:1.5b --stage score      # metrics only

Two stages, so the LLM and the scoring models are never in RAM together:
  1. generate - transcript -> redaction -> LLM -> note, cached per case in
     eval/results/notes/<model>/<encounter>.json (re-runs skip finished cases)
  2. score    - per SOAP section: ROUGE-L and BERTScore vs the (redacted)
     reference note; invented and missing facts via the local NLI model;
     verifier colours. Writes eval_cases.csv, eval_summary.csv, eval_summary.md

Fact definitions (automatic, NLI = the verifier's DeBERTa cross-encoder):
  invented fact  a generated note line that NO transcript passage entails
                 (checked against the cited segments and the 4 most similar
                 transcript windows; max entailment < 0.5)
  missing fact   a reference-note sentence that the generated note does not
                 entail (max entailment over the note text < 0.5)
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.evidence import SegmentIndex, build_premise, speaker_label  # noqa: E402
from app.schema import SECTIONS, Segment, SOAPNote  # noqa: E402

RESULTS = ROOT / "eval" / "results"
THRESHOLD = 0.5
SCORED = list(SECTIONS) + ["note"]  # four sections + whole note


# ---------------------------------------------------------------- text helpers
_LABEL_RE = re.compile(r"^\s*(?:[•\-\*]\s*)?[A-Z][A-Za-z /&]{2,40}:\s*")
_BULLET_RE = re.compile(r"^\s*(?:[•\-\*]|\d+[.)])\s*")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def reference_facts(text: str, min_words: int = 4) -> list[str]:
    """Split a reference section into sentence-level facts.

    Drops bullet markers and labels such as "Medical Reasoning:"; keeps
    sentences with >= ``min_words`` words (drops "Right knee pain." headers).
    """
    facts = []
    for line in text.splitlines():
        line = _LABEL_RE.sub("", _BULLET_RE.sub("", line)).strip()
        for sent in _SENT_RE.split(line):
            sent = sent.strip()
            if len(sent.split()) >= min_words:
                facts.append(sent)
    return facts


def section_text(note: SOAPNote, section: str) -> str:
    if section == "note":
        return "\n".join(it.text for sec in SECTIONS for it in getattr(note, sec))
    return "\n".join(it.text for it in getattr(note, section))


def reference_text(ref: dict[str, str], section: str) -> str:
    return "\n\n".join(ref[s] for s in SECTIONS if ref.get(s)) if section == "note" else ref.get(section, "")


def window_premise(center: int, by_id: dict[int, Segment], radius: int = 1) -> str:
    ids = [i for i in range(center - radius, center + radius + 1) if i in by_id]
    return "\n".join(f"{speaker_label(by_id[i].speaker)}: {by_id[i].text}" for i in ids)


def chunk_lines(lines: list[str], max_words: int = 300) -> list[str]:
    """Join lines into premises of <= max_words (NLI input limit is 512 tokens)."""
    chunks, cur, n = [], [], 0
    for ln in lines:
        w = len(ln.split())
        if cur and n + w > max_words:
            chunks.append(". ".join(cur))
            cur, n = [], 0
        cur.append(ln.rstrip("."))
        n += w
    if cur:
        chunks.append(". ".join(cur))
    return chunks


# ---------------------------------------------------------------- metrics
def rouge_l(cand: str, ref: str) -> float:
    from rouge_score import rouge_scorer

    if not ref.strip():
        return math.nan
    if not cand.strip():
        return 0.0
    sc = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    return sc.score(ref, cand)["rougeLsum"].fmeasure  # rougeLsum: newline-separated lines


def find_invented(note: SOAPNote, segments: list[Segment], entailer, k: int = 4,
                  threshold: float = THRESHOLD) -> list[dict]:
    """Per note line: best entailment from any transcript passage."""
    by_id = {s.id: s for s in segments}
    index = SegmentIndex(segments)
    pairs, owners, lines = [], [], []
    for sec, _, item in note.items():
        prem = set()
        valid = [e for e in item.evidence if e in by_id]
        if valid:
            prem.add(build_premise(valid, by_id)[0])
        for sid in index.suggest(item.text, k=k, min_score=0.5):
            prem.add(window_premise(sid, by_id))
        lines.append({"section": sec, "text": item.text, "flagged_by_model": item.unsupported})
        for p in prem:
            pairs.append((p, item.text, sec))
            owners.append(len(lines) - 1)
    best = [0.0] * len(lines)
    for j, owner in zip(entailer.judge(pairs), owners):
        best[owner] = max(best[owner], j.entailment)
    for ln, b in zip(lines, best):
        ln["best_entailment"] = round(b, 3)
        ln["invented"] = b < threshold
    return lines


def find_missing(note: SOAPNote, ref: dict[str, str], entailer, threshold: float = THRESHOLD) -> list[dict]:
    """Per reference fact: is it entailed by the generated note (any section)?"""
    premises = chunk_lines([it.text for _, _, it in note.items()])
    facts = [{"section": sec, "text": f} for sec in SECTIONS for f in reference_facts(ref.get(sec, ""))]
    if not premises:
        return [{**f, "best_entailment": 0.0, "missing": True} for f in facts]
    pairs = [(p, f["text"], f["section"]) for f in facts for p in premises]
    js = entailer.judge(pairs)
    for i, f in enumerate(facts):
        b = max(j.entailment for j in js[i * len(premises):(i + 1) * len(premises)])
        f["best_entailment"] = round(b, 3)
        f["missing"] = b < threshold
    return facts


def bertscore_f1(cands: list[str], refs: list[str], model_type: str) -> list[float]:
    """Rescaled BERTScore F1; empty candidates score 0, empty references NaN."""
    from bert_score import score

    out = [math.nan if not r.strip() else 0.0 for r in refs]
    idx = [i for i, (c, r) in enumerate(zip(cands, refs)) if c.strip() and r.strip()]
    if idx:
        _, _, f = score([cands[i] for i in idx], [refs[i] for i in idx], model_type=model_type,
                        lang="en", rescale_with_baseline=True, batch_size=8, verbose=False)
        for i, v in zip(idx, f.tolist()):
            out[i] = float(v)
    return out


# ---------------------------------------------------------------- stage 1
def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def unload(settings, model: str) -> None:
    import httpx

    try:  # free Ollama's RAM before the next model / the scoring stage
        httpx.post(f"{settings.ollama_host}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except httpx.HTTPError:
        pass


_INFRA_ERR = re.compile(r"runner has unexpectedly stopped|process has terminated|unexpected EOF|"
                        r"signal: killed|Ollama is not reachable|connection", re.I)


def prepare(cases, out: Path) -> None:
    """Redact transcripts + reference notes -> out/prepared/<encounter>.json.

    The reference gets the same redaction as the transcript ([NAME] vs [NAME]).
    Run in its own process by ``generate`` so spaCy/Presidio (~450 MB) is gone
    before the LLM loads, and the scoring stage never needs spaCy at all.
    """
    from app.config import get_settings
    from app.loader import parse_dialogue
    from app.pipeline import prepare_from_segments
    from app.redact import redact_text

    d = out / "prepared"
    d.mkdir(parents=True, exist_ok=True)
    for c in cases:
        path = d / f"{c.encounter_id}.json"
        if path.exists():
            continue
        prep = prepare_from_segments(parse_dialogue(c.dialogue, get_settings().max_segment_words))
        path.write_text(json.dumps({
            "segments": [s.model_dump() for s in prep.segments], "redactions": prep.redaction.count,
            "reference_redacted": {k: redact_text(v).text for k, v in c.reference_soap.items()}}))


def generate(models: list[str], cases, out: Path, client=None, force: bool = False, split: str = "valid") -> None:
    from app.config import get_settings
    from app.generate import GenerationError, OllamaClient, generate_note

    if any(not (out / "prepared" / f"{c.encounter_id}.json").exists() for c in cases):
        if client is not None:  # tests: in-process is fine
            prepare(cases, out)
        else:
            import subprocess

            ids = ",".join(c.encounter_id for c in cases)
            subprocess.run([sys.executable, "-u", __file__, "--stage", "prepare", "--split", split,
                            "--ids", ids, "--out", str(out)], check=True)
    base = get_settings()
    for model in models:
        settings = dataclasses.replace(base, model=model)
        llm = client or OllamaClient(settings)
        d = out / "notes" / slug(model)
        d.mkdir(parents=True, exist_ok=True)
        todo = [c for c in cases if force or not (d / f"{c.encounter_id}.json").exists()]
        for i, case in enumerate(todo, 1):
            p = json.loads((out / "prepared" / f"{case.encounter_id}.json").read_text())
            segs = [Segment(**s) for s in p["segments"]]
            rec = {"model": model, "encounter_id": case.encounter_id, "n_segments": len(segs), **p}
            for infra_try in range(3):
                t0 = time.perf_counter()
                try:
                    g = generate_note(segs, client=llm, settings=settings)
                    rec.update(ok=True, note=g.note.model_dump(), attempts=g.attempts, error=None)
                except (GenerationError, ValueError) as e:
                    rec.update(ok=False, note=None, attempts=2, error=str(e)[:300])
                rec["duration_s"] = round(time.perf_counter() - t0, 1)
                # a crashed Ollama runner (e.g. out of memory) is not the model's fault: retry
                if rec["ok"] or not _INFRA_ERR.search(rec["error"] or ""):
                    break
                print(f"[gen] {model} {case.encounter_id}: Ollama runner crashed, retrying "
                      f"({infra_try + 1}/2)", flush=True)
                time.sleep(5)
            rec["infra_error"] = (not rec["ok"]) and bool(_INFRA_ERR.search(rec["error"] or ""))
            (d / f"{case.encounter_id}.json").write_text(json.dumps(rec, indent=1))
            status = f"{SOAPNote.model_validate(rec['note']).n_items()} lines" if rec["ok"] else "FAILED"
            print(f"[gen] {model:<14} {i:>2}/{len(todo)} {case.encounter_id}  {rec['duration_s']:6.1f}s  {status}",
                  flush=True)
        if client is None:
            unload(settings, model)


# ---------------------------------------------------------------- stage 2
def score(models: list[str], cases, out: Path, entailer=None, bertscore_model: str | None = "distilbert-base-uncased",
          ) -> list[dict]:
    from app.redact import redact_text
    from app.verify import get_entailer, verify_note

    if entailer is None:
        entailer = get_entailer()
        if hasattr(entailer, "batch_size"):
            entailer.batch_size = 4  # long premises (whole notes): keep activations small
    rows, bs_jobs, details = [], [], []
    ref_cache: dict[str, dict[str, str]] = {}
    for model in models:
        for case in cases:
            path = out / "notes" / slug(model) / f"{case.encounter_id}.json"
            if not path.exists():
                print(f"[score] missing {path} - run --stage generate first", flush=True)
                continue
            rec = json.loads(path.read_text())
            row = {"model": model, "encounter_id": case.encounter_id, "ok": rec["ok"],
                   "infra_error": rec.get("infra_error", False),
                   "duration_s": rec["duration_s"], "attempts": rec["attempts"], "error": rec.get("error") or ""}
            if not rec["ok"]:
                rows.append(row)
                continue
            ref = rec.get("reference_redacted")
            if ref is None:  # notes cached by an older version
                if case.encounter_id not in ref_cache:
                    ref_cache[case.encounter_id] = {s: redact_text(t).text for s, t in case.reference_soap.items()}
                ref = ref_cache[case.encounter_id]
            note = SOAPNote.model_validate(rec["note"])
            segs = [Segment(**s) for s in rec["segments"]]

            rep = verify_note(note, segs, entailer=entailer, suggest=False)
            counts = rep.counts
            inv = find_invented(note, segs, entailer)
            mis = find_missing(note, ref, entailer)
            row.update(
                n_items=note.n_items(),
                **{f"items_{s}": len(getattr(note, s)) for s in SECTIONS},
                green=counts.get("supported", 0), amber=counts.get("flagged", 0), red=counts.get("unsupported", 0),
                flagged_by_model=sum(x["flagged_by_model"] for x in inv),
                invented=sum(x["invented"] for x in inv),
                ref_facts=len(mis), missing=sum(x["missing"] for x in mis),
                **{f"missing_{s}": sum(x["missing"] for x in mis if x["section"] == s) for s in SECTIONS},
                **{f"ref_facts_{s}": sum(1 for x in mis if x["section"] == s) for s in SECTIONS},
            )
            for sec in SCORED:
                c, r = section_text(note, sec), reference_text(ref, sec)
                row[f"rougeL_{sec}"] = rouge_l(c, r)
                bs_jobs.append((len(rows), sec, c, r))
            details += [{"model": model, "encounter_id": case.encounter_id, "kind": "line", **x} for x in inv]
            details += [{"model": model, "encounter_id": case.encounter_id, "kind": "ref_fact", **x} for x in mis]
            rows.append(row)
            print(f"[score] {model:<14} {case.encounter_id}  items={row['n_items']:>2} invented={row['invented']:>2} "
                  f"missing={row['missing']:>2}/{row['ref_facts']:<2} rougeL={row['rougeL_note']:.3f}", flush=True)

    if bertscore_model and bs_jobs:
        del entailer
        gc.collect()  # the NLI model goes before BERTScore loads (low-RAM machines)
        f1 = bertscore_f1([j[2] for j in bs_jobs], [j[3] for j in bs_jobs], bertscore_model)
        for (ri, sec, _, _), v in zip(bs_jobs, f1):
            rows[ri][f"bertscore_{sec}"] = v
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "eval_cases.csv", rows)
    _write_csv(out / "eval_facts.csv", details)
    return rows


# ---------------------------------------------------------------- summary
def _mean(xs) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return statistics.fmean(xs) if xs else math.nan


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == model]
        ok = [r for r in rs if r["ok"]]
        infra = [r for r in rs if not r["ok"] and str(r.get("infra_error")) == "True"]
        items = sum(r["n_items"] for r in ok)
        facts = sum(r["ref_facts"] for r in ok)
        s = {"model": model, "cases": len(rs) - len(infra), "valid_notes": len(ok), "infra_failures": len(infra),
             "gen_s_mean": _mean([r["duration_s"] for r in ok]),
             "items_per_note": items / len(ok) if ok else math.nan,
             "invented": sum(r["invented"] for r in ok), "items": items,
             "invented_rate": sum(r["invented"] for r in ok) / items if items else math.nan,
             "missing": sum(r["missing"] for r in ok), "ref_facts": facts,
             "missing_rate": sum(r["missing"] for r in ok) / facts if facts else math.nan,
             "red_rate": sum(r["red"] for r in ok) / items if items else math.nan,
             "amber_rate": sum(r["amber"] for r in ok) / items if items else math.nan,
             "green_rate": sum(r["green"] for r in ok) / items if items else math.nan}
        for sec in SECTIONS:
            f = sum(r[f"ref_facts_{sec}"] for r in ok)
            s[f"missing_rate_{sec}"] = sum(r[f"missing_{sec}"] for r in ok) / f if f else math.nan
        for m in ("rougeL", "bertscore"):
            for sec in SCORED:
                s[f"{m}_{sec}"] = _mean([r.get(f"{m}_{sec}", math.nan) for r in ok])
        out.append(s)
    return out


def _fmt(x, pct: bool = False) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "–"
    return f"{100 * x:.0f}%" if pct else (f"{x:.3f}" if isinstance(x, float) else str(x))


def markdown(summary: list[dict], split: str, bertscore_model: str | None) -> str:
    sec_names = {"subjective": "S", "objective": "O", "assessment": "A", "plan": "P", "note": "Whole note"}
    L = [f"## Note quality on ACI-Bench `{split}` ({summary[0]['cases'] if summary else 0} cases)", ""]
    L += ["| Model | Valid notes | Lines/note | Invented facts | Missing facts | Verifier red / amber / green | Gen. time |",
          "|---|---|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {s['model']} | {s['valid_notes']}/{s['cases']} | {s['items_per_note']:.1f} | "
                 f"{s['invented']}/{s['items']} ({_fmt(s['invented_rate'], True)}) | "
                 f"{s['missing']}/{s['ref_facts']} ({_fmt(s['missing_rate'], True)}) | "
                 f"{_fmt(s['red_rate'], True)} / {_fmt(s['amber_rate'], True)} / {_fmt(s['green_rate'], True)} | "
                 f"{s['gen_s_mean']:.0f} s |")
    for s in summary:
        if s.get("infra_failures"):
            L.append(f"\n{s['model']}: {s['infra_failures']} case(s) excluded - the Ollama runner crashed "
                     "(out of memory), so these say nothing about the model's output.")
    for m, title in (("rougeL", "ROUGE-L (F1)"), ("bertscore", f"BERTScore F1 (rescaled, `{bertscore_model}`)")):
        if m == "bertscore" and not bertscore_model:
            continue
        L += ["", f"**{title}** per section", "",
              "| Model | " + " | ".join(sec_names[s] for s in SCORED) + " |", "|---" * (len(SCORED) + 1) + "|"]
        for s in summary:
            L.append(f"| {s['model']} | " + " | ".join(_fmt(s[f'{m}_{sec}']) for sec in SCORED) + " |")
    L += ["", "**Missing facts by section** (share of reference sentences not covered)", "",
          "| Model | " + " | ".join(sec_names[s] for s in SECTIONS) + " |", "|---" * (len(SECTIONS) + 1) + "|"]
    for s in summary:
        L.append(f"| {s['model']} | " + " | ".join(_fmt(s[f'missing_rate_{sec}'], True) for sec in SECTIONS) + " |")
    stt = RESULTS / "stt_wer.csv"
    if stt.exists():
        rows = [r for r in csv.DictReader(stt.open()) if r.get("ref_words")]
        words = sum(int(r["ref_words"]) for r in rows)
        if words:
            wer = sum(int(r["errors"]) for r in rows) / words
            L += ["", f"Speech-to-text (faster-whisper {rows[0].get('model', '')}, jiwer): pooled WER "
                      f"**{100 * wer:.1f}%** on {len(rows)} synthesised consultations (`eval/results/stt_wer.csv`)."]
    L += ["", "Invented = note line not entailed by any transcript passage; missing = reference sentence not "
              "entailed by the generated note (local NLI, threshold 0.5). Automatic estimates, not clinician review."]
    return "\n".join(L) + "\n"


def load_cases_csv(path: Path) -> list[dict]:
    """eval_cases.csv -> rows with numbers/bools restored (for --stage report)."""
    out = []
    for r in csv.DictReader(path.open()):
        row = {}
        for k, v in r.items():
            if v in ("True", "False"):
                row[k] = v == "True"
            elif k in ("model", "encounter_id", "error"):
                row[k] = v
            else:
                try:
                    row[k] = float(v) if v != "" else math.nan
                    if row[k].is_integer() and not k.startswith(("rougeL", "bertscore")):
                        row[k] = int(row[k])
                except ValueError:
                    row[k] = v
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default=None, help="comma-separated Ollama models (default: SCRIBE_MODEL)")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n", type=int, default=20, help="number of cases (0 = all)")
    ap.add_argument("--stage", choices=["prepare", "generate", "score", "all", "report"], default="all",
                    help="report = rebuild eval_summary.* from eval_cases.csv without re-scoring")
    ap.add_argument("--ids", default=None, help=argparse.SUPPRESS)  # internal: --stage prepare
    ap.add_argument("--bertscore-model", default="distilbert-base-uncased",
                    help="'roberta-large' is the BERTScore default (needs ~2 GB RAM); 'none' to skip")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--force", action="store_true", help="regenerate notes that are already cached")
    a = ap.parse_args(argv)

    from app.config import get_settings
    from app.loader import load_split

    models = [m.strip() for m in (a.models or get_settings().model).split(",") if m.strip()]
    cases = load_split(a.split, limit=a.n or None)
    if a.ids:
        keep = set(a.ids.split(","))
        cases = [c for c in load_split(a.split) if c.encounter_id in keep]
    out = Path(a.out)
    bs = None if a.bertscore_model.lower() == "none" else a.bertscore_model
    if a.stage == "prepare":
        prepare(cases, out)
        return
    if a.stage == "generate":
        generate(models, cases, out, force=a.force, split=a.split)
    elif a.stage == "all":  # separate process: its RAM (spaCy, buffers) is fully returned before scoring
        import subprocess

        args = [sys.executable, "-u", __file__, "--stage", "generate", "--models", ",".join(models),
                "--split", a.split, "--n", str(a.n), "--out", str(out)] + (["--force"] if a.force else [])
        subprocess.run(args, check=True)
    if a.stage in ("score", "all", "report"):
        rows = load_cases_csv(out / "eval_cases.csv") if a.stage == "report" else \
            score(models, cases, out, bertscore_model=bs)
        summ = summarize(rows)
        _write_csv(out / "eval_summary.csv", summ)
        md = markdown(summ, a.split, bs)
        (out / "eval_summary.md").write_text(md)
        print("\n" + md)


if __name__ == "__main__":
    main()
