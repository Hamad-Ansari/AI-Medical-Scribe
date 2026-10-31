"""ACI-Bench loader (synthetic data only).

ACI-Bench (Yim et al., 2023, CC BY 4.0) ships CSVs with columns
``dataset, encounter_id, dialogue, note``. Dialogues are text with
``[doctor]`` / ``[patient]`` / ``[patient_guest]`` speaker tags; there is no
audio, so phase 3 synthesises it with Piper when an audio path is needed.

This module turns a dialogue into numbered :class:`Segment` objects and maps
the free-form reference note onto the four SOAP sections for evaluation.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.schema import SECTIONS, Segment

SPLITS: dict[str, str] = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test1": "clinicalnlp_taskB_test1.csv",
    "test2": "clinicalnlp_taskC_test2.csv",
    "test3": "clef_taskC_test3.csv",
}
ACI_BASE_URL = "https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/"

_TURN_RE = re.compile(r"^\[(?P<spk>[a-z_]+)\]\s*(?P<text>.*)$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")


class ACICase(BaseModel):
    dataset: str
    encounter_id: str
    split: str
    dialogue: str
    reference_note: str
    segments: list[Segment]
    reference_sections: dict[str, str] = Field(description="raw note headers -> text")
    reference_soap: dict[str, str] = Field(description="subjective/objective/assessment/plan -> text")


# --------------------------------------------------------------------------
# Dialogue -> segments
# --------------------------------------------------------------------------
def _chunk_turn(text: str, max_words: int) -> list[str]:
    """Split one long turn into sentence groups of at most ``max_words`` words.

    A single sentence longer than ``max_words`` is kept whole (we never cut
    mid-sentence, which would break the evidence span).
    """
    if max_words <= 0 or len(text.split()) <= max_words:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in _SENT_SPLIT_RE.split(text):
        n = len(sent.split())
        if cur and cur_len + n > max_words:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(sent)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def parse_dialogue(dialogue: str, max_words: int | None = None) -> list[Segment]:
    """Parse ``[speaker] text`` lines into 1-based numbered segments.

    Lines without a tag are appended to the previous turn (ACI-Bench has a few
    wrapped lines).
    """
    if max_words is None:
        max_words = get_settings().max_segment_words
    turns: list[tuple[str, str]] = []
    for raw in dialogue.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _TURN_RE.match(line)
        if m:
            text = m.group("text").strip()
            if text:
                turns.append((m.group("spk"), text))
        elif turns:
            spk, prev = turns[-1]
            turns[-1] = (spk, f"{prev} {line}")
    if not turns:
        raise ValueError("dialogue contains no '[speaker] text' turns")

    segments: list[Segment] = []
    for spk, text in turns:
        for chunk in _chunk_turn(text, max_words):
            segments.append(Segment(id=len(segments) + 1, speaker=spk, text=chunk))
    return segments


# --------------------------------------------------------------------------
# Reference note -> SOAP
# --------------------------------------------------------------------------
_HEADER_RE = re.compile(r"^(?P<h>[A-Z][A-Z /&]{1,48}?):?\s*$")

HEADER_TO_SOAP: dict[str, str] = {
    # Subjective
    "CHIEF COMPLAINT": "subjective", "CC": "subjective",
    "HISTORY OF PRESENT ILLNESS": "subjective", "HPI": "subjective",
    "REVIEW OF SYSTEMS": "subjective", "REVIEW OF SYMPTOMS": "subjective", "ROS": "subjective",
    "SOCIAL HISTORY": "subjective", "FAMILY HISTORY": "subjective",
    "MEDICAL HISTORY": "subjective", "PAST HISTORY": "subjective", "PAST MEDICAL HISTORY": "subjective",
    "SURGICAL HISTORY": "subjective", "PAST SURGICAL HISTORY": "subjective", "BIRTH HISTORY": "subjective",
    "MEDICATIONS": "subjective", "CURRENT MEDICATIONS": "subjective", "ALLERGIES": "subjective",
    "SUBJECTIVE": "subjective",
    # Objective
    "PHYSICAL EXAM": "objective", "PHYSICAL EXAMINATION": "objective", "EXAM": "objective",
    "VITALS": "objective", "VITALS REVIEWED": "objective", "VITAL SIGNS": "objective",
    "RESULTS": "objective", "EKG": "objective", "OBJECTIVE": "objective",
    # Assessment (bullets inside are split further, see _split_assessment)
    "ASSESSMENT": "assessment", "IMPRESSION": "assessment",
    "ASSESSMENT AND PLAN": "assessment_and_plan",
    # Plan
    "PLAN": "plan", "INSTRUCTIONS": "plan", "ORDERS": "plan", "PROCEDURE": "plan",
}

# Bullet labels inside ASSESSMENT / A&P blocks that are really plan content.
_PLAN_LABELS = re.compile(
    r"^\s*(?:[•\-\*]\s*)?(additional testing|medical treatment|patient education and counseling|"
    r"specialist referrals?|referrals?|patient agreements|follow[- ]?up|orders?|diagnostic testing)\s*:",
    re.I,
)


def split_note_sections(note: str) -> dict[str, str]:
    """Split a reference note on ALL-CAPS header lines -> {header: body}."""
    sections: dict[str, list[str]] = {}
    current = "_PREAMBLE"
    for line in note.splitlines():
        m = _HEADER_RE.match(line.strip())
        if m:  # a line that is only an ALL-CAPS phrase (optionally with ':')
            current = m.group("h").strip()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    out = {h: "\n".join(b).strip() for h, b in sections.items()}
    return {h: b for h, b in out.items() if b}


def _split_assessment(body: str) -> tuple[str, str]:
    """Split an ASSESSMENT(-AND-PLAN) body into (assessment, plan) text.

    ACI-Bench A&P blocks list problems, each followed by bullets such as
    ``• Medical Reasoning:`` (assessment) and ``• Medical Treatment:`` (plan).
    """
    a_lines: list[str] = []
    p_lines: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        (p_lines if _PLAN_LABELS.match(line) else a_lines).append(line.strip())
    return "\n".join(a_lines), "\n".join(p_lines)


def note_to_soap(note: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(raw_sections, soap_sections)`` for a reference note.

    Unknown headers default to *subjective* only if they look like history;
    otherwise they are kept under ``raw_sections`` but not scored.
    """
    raw = split_note_sections(note)
    soap: dict[str, list[str]] = {s: [] for s in SECTIONS}
    prev_target: str | None = None
    for header, body in raw.items():
        target = HEADER_TO_SOAP.get(header)
        if target is None:
            if "HISTORY" in header:
                target = "subjective"
            elif prev_target is not None:
                # unknown ALL-CAPS line (e.g. "MSK" inside PHYSICAL EXAM) is a
                # sub-header: keep it in the enclosing section
                target = prev_target
            else:
                continue
        prev_target = target
        if target in ("assessment", "assessment_and_plan"):
            a, p = _split_assessment(body)
            if a:
                soap["assessment"].append(a)
            if p:
                soap["plan"].append(p)
        else:
            soap[target].append(body)
    return raw, {s: "\n\n".join(v).strip() for s, v in soap.items()}


# --------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------
def split_path(split: str, data_dir: Path | None = None) -> Path:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(SPLITS)}")
    data_dir = data_dir or get_settings().data_dir
    return Path(data_dir) / SPLITS[split]


def load_split(split: str = "valid", data_dir: Path | None = None, limit: int | None = None) -> list[ACICase]:
    path = split_path(split, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python scripts/download_aci_bench.py` first")
    cases: list[ACICase] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw, soap = note_to_soap(row["note"])
            cases.append(
                ACICase(
                    dataset=row["dataset"],
                    encounter_id=row["encounter_id"],
                    split=split,
                    dialogue=row["dialogue"],
                    reference_note=row["note"],
                    segments=parse_dialogue(row["dialogue"]),
                    reference_sections=raw,
                    reference_soap=soap,
                )
            )
            if limit and len(cases) >= limit:
                break
    return cases


def load_case(encounter_id: str, split: str = "valid", data_dir: Path | None = None) -> ACICase:
    for case in load_split(split, data_dir):
        if case.encounter_id == encounter_id:
            return case
    raise KeyError(f"encounter {encounter_id!r} not in split {split!r}")
