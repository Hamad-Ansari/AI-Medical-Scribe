"""Faithfulness verifier: is every note line backed by the transcript?

For each item:
  (a) structural checks - cited IDs exist, evidence not empty, not over-cited,
      not self-marked unsupported by the generator;
  (b) entailment - does the cited transcript text support the line?
      backend "nli"    : local DeBERTa-v3 NLI cross-encoder (default, CPU, deterministic)
      backend "ollama" : local LLM judge returning {"supported": bool, "reason": str}
  (c) status = supported | flagged | unsupported, with human-readable reasons.

Status rules (first match wins)
-------------------------------
unsupported (red)   : no evidence cited | none of the cited IDs exist |
                      generator marked unsupported=true | cited text contradicts it |
                      cited text does not support it (entailment < weak_threshold / judge says no)
flagged (amber)     : weakly supported (weak_threshold <= entailment < entail_threshold) |
                      a number or left/right/bilateral in the line is absent from the cited text |
                      some cited IDs don't exist | cites more than max_evidence_ok segments
supported (green)   : everything else

The verifier never edits the note. It only annotates it; the doctor decides.
Privacy: logs contain counts only, never note or transcript text.
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from functools import lru_cache
from typing import Protocol

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.evidence import SegmentIndex, build_premise, resolve, ungrounded_critical_tokens
from app.schema import Section, Segment, SOAPNote

log = logging.getLogger(__name__)


class Status(str, Enum):
    supported = "supported"
    flagged = "flagged"
    unsupported = "unsupported"


class Judgement(BaseModel):
    """Backend output for one (premise, hypothesis) pair."""

    entailment: float | None = None  # NLI probabilities (None for LLM judge)
    contradiction: float | None = None
    supported: bool | None = None  # LLM judge verdict (None for NLI)
    reason: str = ""


class ItemVerification(BaseModel):
    section: Section
    index: int
    text: str
    evidence: list[int]
    status: Status
    reasons: list[str] = Field(default_factory=list)
    valid_ids: list[int] = Field(default_factory=list)
    invalid_ids: list[int] = Field(default_factory=list)
    context_ids: list[int] = Field(default_factory=list, description="previous turns added as context")
    entailment: float | None = None
    contradiction: float | None = None
    ungrounded_tokens: list[str] = Field(default_factory=list, description="numbers/laterality not in cited text")
    suggested_evidence: list[int] = Field(default_factory=list, description="hint only; never changes status")


class VerificationReport(BaseModel):
    method: str
    items: list[ItemVerification]

    @property
    def counts(self) -> dict[str, int]:
        c = {s.value: 0 for s in Status}
        for it in self.items:
            c[it.status.value] += 1
        return c

    def for_item(self, section: str, index: int) -> ItemVerification:
        for it in self.items:
            if it.section == section and it.index == index:
                return it
        raise KeyError((section, index))


# ------------------------------------------------------------------ backends
class Entailer(Protocol):
    name: str

    def judge(self, pairs: list[tuple[str, str, str]]) -> list[Judgement]:
        """pairs = [(premise, hypothesis, section), ...]"""


# Section-aware phrasing: NLI models read "Knee sprain" as neutral against
# "i think this is a knee sprain", but entail "The doctor's assessment: knee sprain".
_FRAMES = {
    "subjective": "The patient reports: {}",
    "objective": "The examination or results show: {}",
    "assessment": "The doctor's assessment: {}",
    "plan": "The doctor's plan: {}",
}


class NLIEntailer:
    """DeBERTa-v3 NLI cross-encoder. Scores the raw line and a section-framed
    version and keeps the variant with the higher entailment probability."""

    def __init__(self, model_name: str | None = None, batch_size: int = 16):
        self.model_name = model_name or get_settings().nli_model
        self.name = f"nli:{self.model_name}"
        self.batch_size = batch_size

    @staticmethod
    @lru_cache(maxsize=2)
    def _load(model_name: str):
        import torch  # local import: torch is only needed for this backend
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(max(1, torch.get_num_threads()))
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        return tok, model, labels["entailment"], labels["contradiction"]

    def _probs(self, premises: list[str], hyps: list[str]) -> list[tuple[float, float]]:
        import torch

        tok, model, ent_i, con_i = self._load(self.model_name)
        out: list[tuple[float, float]] = []
        for b in range(0, len(premises), self.batch_size):
            enc = tok(
                premises[b : b + self.batch_size],
                hyps[b : b + self.batch_size],
                padding=True,
                truncation="only_first",
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                p = model(**enc).logits.softmax(-1)
            out.extend((float(r[ent_i]), float(r[con_i])) for r in p)
        return out

    def judge(self, pairs):
        if not pairs:
            return []
        premises, hyps = [], []
        for prem, hyp, sec in pairs:
            premises += [prem, prem]
            hyps += [hyp, _FRAMES[sec].format(hyp[0].lower() + hyp[1:] if hyp else hyp)]
        probs = self._probs(premises, hyps)
        res = []
        for i in range(len(pairs)):
            raw, framed = probs[2 * i], probs[2 * i + 1]
            ent, con = max(raw, framed, key=lambda x: x[0])
            res.append(Judgement(entailment=ent, contradiction=con))
        return res


JUDGE_SYSTEM = """\
You check a clinical note line against transcript excerpts. Decide whether the
excerpts explicitly support the line. Answer supported=false if the line adds
any detail (dose, side, duration, finding, diagnosis) the excerpts do not state,
or if the excerpts contradict it. Do not use outside medical knowledge.
Return JSON: {"supported": true|false, "reason": "<one short sentence>"}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string", "maxLength": 200}},
    "required": ["supported", "reason"],
}


class OllamaJudge:
    """LLM-as-judge using the local Ollama server (temperature 0)."""

    def __init__(self, client=None, settings: Settings | None = None):
        from app.generate import OllamaClient

        self.settings = settings or get_settings()
        self.client = client or OllamaClient(self.settings)
        self.name = f"ollama:{self.settings.judge_model}"

    def judge(self, pairs):
        out = []
        for prem, hyp, sec in pairs:
            msgs = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"TRANSCRIPT EXCERPTS:\n{prem}\n\nNOTE LINE ({sec}): {hyp}"},
            ]
            try:
                kwargs = {"format": JUDGE_SCHEMA, "options": {"temperature": 0, "seed": 0, "num_predict": 200}}
                if hasattr(self.client, "settings"):  # real OllamaClient: use the judge model
                    kwargs["model"] = self.settings.judge_model
                raw = self.client.chat(msgs, **kwargs)
                data = json.loads(raw)
                out.append(Judgement(supported=bool(data["supported"]), reason=str(data.get("reason", ""))[:200]))
            except Exception as e:  # judge failure must not crash verification
                log.warning("judge failed (%s)", type(e).__name__)
                out.append(Judgement(supported=None, reason=f"judge error: {type(e).__name__}"))
        return out


def get_entailer(settings: Settings | None = None) -> Entailer:
    settings = settings or get_settings()
    if settings.verifier == "nli":
        return NLIEntailer(settings.nli_model)
    if settings.verifier == "ollama":
        return OllamaJudge(settings=settings)
    raise ValueError(f"unknown SCRIBE_VERIFIER={settings.verifier!r} (use 'nli' or 'ollama')")


# ------------------------------------------------------------------ main API
def _decide(item_v: ItemVerification, j: Judgement | None, s: Settings) -> None:
    """Apply the status rules (see module docstring). Mutates item_v."""
    reasons = item_v.reasons
    red = amber = False

    if not item_v.evidence:
        red = True
        reasons.append("No transcript segment cited.")
    elif not item_v.valid_ids:
        red = True
        reasons.append(f"Cited segment(s) {item_v.invalid_ids} do not exist in the transcript.")
    elif item_v.invalid_ids:
        amber = True
        reasons.append(f"Cited segment(s) {item_v.invalid_ids} do not exist; checked against the rest.")

    if j is not None:
        if j.supported is not None:  # LLM judge
            if not j.supported:
                red = True
                reasons.append(f"Judge: not supported by cited text - {j.reason}".rstrip(" -"))
        elif j.entailment is not None:  # NLI
            item_v.entailment = round(j.entailment, 3)
            item_v.contradiction = round(j.contradiction or 0.0, 3)
            if (j.contradiction or 0) >= s.contradiction_threshold and j.entailment < s.entail_threshold:
                red = True
                reasons.append(f"Cited text contradicts this line (contradiction={j.contradiction:.2f}).")
            elif j.entailment < s.weak_threshold:
                red = True
                reasons.append(f"Cited text does not state this (entailment={j.entailment:.2f}).")
            elif j.entailment < s.entail_threshold:
                amber = True
                reasons.append(f"Only weakly supported by cited text (entailment={j.entailment:.2f}).")
        else:  # backend error
            amber = True
            reasons.append(j.reason or "Verifier could not check this line.")

    if item_v.ungrounded_tokens and not red:
        amber = True
        toks = ", ".join(f"'{t}'" for t in item_v.ungrounded_tokens)
        reasons.append(f"Mentions {toks}, not found in the cited text.")

    if len(item_v.valid_ids) > s.max_evidence_ok:
        amber = True
        reasons.append(f"Cites {len(item_v.valid_ids)} segments; evidence too broad to verify precisely.")

    item_v.status = Status.unsupported if red else Status.flagged if amber else Status.supported


def verify_note(
    note: SOAPNote,
    segments: list[Segment],
    entailer: Entailer | None = None,
    settings: Settings | None = None,
    suggest: bool = True,
) -> VerificationReport:
    """Verify every line of ``note`` against ``segments``."""
    settings = settings or get_settings()
    entailer = entailer or get_entailer(settings)
    by_id = {s.id: s for s in segments}
    index = SegmentIndex(segments) if suggest else None

    results: list[ItemVerification] = []
    pairs: list[tuple[str, str, str]] = []
    pair_owner: list[int] = []

    for sec, i, item in note.items():
        r = resolve(item, by_id)
        iv = ItemVerification(
            section=sec, index=i, text=item.text, evidence=item.evidence,
            status=Status.supported, valid_ids=r.valid_ids, invalid_ids=r.invalid_ids,
        )
        if item.unsupported:
            iv.reasons.append("Generator marked this line as not supported by the transcript.")
        if r.valid_ids and not item.unsupported:
            premise, ctx = build_premise(r.valid_ids, by_id, settings.short_segment_words)
            iv.context_ids = ctx
            iv.ungrounded_tokens = ungrounded_critical_tokens(item.text, premise)
            pairs.append((premise, item.text, sec))
            pair_owner.append(len(results))
        results.append(iv)

    judgements: dict[int, Judgement] = {}
    if pairs:
        for owner, j in zip(pair_owner, entailer.judge(pairs)):
            judgements[owner] = j

    for k, iv in enumerate(results):
        if any(r.startswith("Generator marked") for r in iv.reasons):
            iv.status = Status.unsupported
            if not iv.evidence:
                iv.reasons.append("No transcript segment cited.")
        else:
            _decide(iv, judgements.get(k), settings)
        if index is not None and iv.status != Status.supported:
            iv.suggested_evidence = index.suggest(iv.text, k=3, exclude=set(iv.valid_ids))

    report = VerificationReport(method=entailer.name, items=results)
    log.info("verified %d items with %s: %s", len(results), entailer.name, report.counts)
    return report


def reverify_note(
    note: SOAPNote,
    segments: list[Segment],
    previous_note: SOAPNote | None,
    previous_report: VerificationReport | None,
    entailer: Entailer | None = None,
    settings: Settings | None = None,
) -> tuple[VerificationReport, int]:
    """Verify a doctor-edited note, re-checking only new or changed lines.

    A line is unchanged if its section, text, evidence and ``unsupported`` flag
    all match a line of ``previous_note``; it keeps its earlier result (with its
    new index). Returns ``(report, n_rechecked)``.
    """
    old: dict[tuple, ItemVerification] = {}
    if previous_note is not None and previous_report is not None:
        for sec, i, it in previous_note.items():
            try:
                old.setdefault((sec, it.text, tuple(it.evidence), it.unsupported), previous_report.for_item(sec, i))
            except KeyError:
                pass
    keep: dict[tuple[str, int], ItemVerification] = {}
    changed: dict[str, list] = {s: [] for s in ("subjective", "objective", "assessment", "plan")}
    changed_pos: dict[str, list[int]] = {s: [] for s in changed}
    for sec, i, it in note.items():
        hit = old.get((sec, it.text, tuple(it.evidence), it.unsupported))
        if hit is not None:
            keep[(sec, i)] = hit.model_copy(update={"index": i})
        else:
            changed[sec].append(it)
            changed_pos[sec].append(i)
    n_changed = sum(len(v) for v in changed.values())
    fresh: dict[tuple[str, int], ItemVerification] = {}
    method = previous_report.method if previous_report else "none"
    if n_changed:
        sub = SOAPNote(**changed)
        rep = verify_note(sub, segments, entailer=entailer, settings=settings)
        method = rep.method
        for iv in rep.items:
            real = changed_pos[iv.section][iv.index]
            fresh[(iv.section, real)] = iv.model_copy(update={"index": real})
    items = [keep.get((sec, i)) or fresh[(sec, i)] for sec, i, _ in note.items()]
    return VerificationReport(method=method, items=items), n_changed
