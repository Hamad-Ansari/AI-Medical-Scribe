"""Evidence linking: connect each note line to the transcript segments behind it.

* :func:`resolve` splits an item's cited IDs into valid / invalid.
* :func:`build_premise` turns the cited segments into the text the verifier
  checks the line against. Very short cited turns ("yes .", "mm-hmm .") are
  meaningless alone, so the previous segment is included as context - that is
  how "Doctor: any swelling? / Patient: yes" supports "Knee swelling".
* :func:`suggest_evidence` ranks segments by lexical overlap with a line. It is
  a *hint for the doctor* when citations are missing or wrong; it never changes
  a line's verification status.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from app.schema import NoteItem, Segment

SPEAKER_LABELS = {"doctor": "Doctor", "patient": "Patient", "patient_guest": "Patient's companion"}

_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = frozenset(
    """a an the and or but if of to in on at for with by from as is are was were be been being am do does did
    have has had i you he she it we they me him her us them my your his its our their this that these those
    so okay ok um uh mm hmm yeah yes no not n't just like really very well right all any some there here what
    how when where who which can could would should will shall may might must about over into out up down
    then than too also patient doctor pt reports reported states stated""".split()
)


def n_words(text: str) -> int:
    """Word count ignoring ACI-Bench's space-separated punctuation (" , ", " . ")."""
    return sum(1 for t in text.split() if any(ch.isalnum() for ch in t))


def speaker_label(spk: str) -> str:
    return SPEAKER_LABELS.get(spk, spk.replace("_", " ").capitalize())


@dataclass
class ResolvedEvidence:
    valid_ids: list[int] = field(default_factory=list)
    invalid_ids: list[int] = field(default_factory=list)
    context_ids: list[int] = field(default_factory=list)  # added automatically, not cited


def resolve(item: NoteItem, by_id: dict[int, Segment]) -> ResolvedEvidence:
    r = ResolvedEvidence()
    for i in item.evidence:
        (r.valid_ids if i in by_id else r.invalid_ids).append(i)
    return r


def build_premise(ids: list[int], by_id: dict[int, Segment], short_words: int = 8) -> tuple[str, list[int]]:
    """Return (premise_text, context_ids_added) for the cited valid IDs."""
    include: list[int] = []
    context: list[int] = []
    cited = set(ids)
    for i in sorted(set(ids)):
        seg = by_id[i]
        prev = i - 1
        if n_words(seg.text) < short_words and prev in by_id and prev not in cited and prev not in include:
            include.append(prev)
            context.append(prev)
        include.append(i)
    lines = [f"{speaker_label(by_id[i].speaker)}: {by_id[i].text}" for i in sorted(set(include))]
    return " ".join(lines), context


# ---------------------------------------------------------------- suggestions
def _tokens(text: str) -> list[str]:
    out = []
    for w in _WORD_RE.findall(text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        # crude stemming keeps this dependency-free: "swelling"->"swell", "knees"->"knee"
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


class SegmentIndex:
    """Tiny TF-IDF index over transcript segments for evidence suggestions."""

    def __init__(self, segments: list[Segment]):
        self.segments = segments
        self.docs = {s.id: Counter(_tokens(s.text)) for s in segments}
        df: Counter[str] = Counter()
        for c in self.docs.values():
            df.update(c.keys())
        n = max(len(segments), 1)
        self.idf = {t: math.log((n + 1) / (d + 0.5)) for t, d in df.items()}

    def suggest(self, text: str, k: int = 3, exclude: set[int] | None = None, min_score: float = 1.0) -> list[int]:
        q = set(_tokens(text))
        if not q:
            return []
        exclude = exclude or set()
        scored = []
        for sid, counts in self.docs.items():
            if sid in exclude:
                continue
            score = sum(self.idf.get(t, 0.0) for t in q if t in counts)
            if score >= min_score:
                scored.append((score, -sid, sid))
        scored.sort(reverse=True)
        return [sid for _, _, sid in scored[:k]]


def suggest_evidence(text: str, segments: list[Segment], k: int = 3) -> list[int]:
    return SegmentIndex(segments).suggest(text, k=k)


# ---------------------------------------------------------- critical tokens
# Numbers (doses, vitals, durations) and laterality are the details whose
# silent corruption is most dangerous and that NLI models handle least reliably.
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALES = {"hundred": 100, "thousand": 1000}
_LATERALITY = {"left", "right", "bilateral"}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _run_value(words: list[str]) -> int:
    total, cur = 0, 0
    for w in words:
        if w in _UNITS:
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            cur = max(cur, 1) * 100
        elif w == "thousand":
            total += max(cur, 1) * 1000
            cur = 0
    return total + cur


def _numbers(text: str, lenient: bool = False) -> set[str]:
    """Extract numbers as canonical strings.

    Only whole tokens count ("a1c" is not a number). Spoken numbers are parsed
    ("one hundred fifty" -> 150). With ``lenient=True`` (used for the cited
    transcript) every plausible reading is added, e.g. colloquial
    "one fifty" -> 150 as well as 1 and 50, so paraphrases are not flagged.
    """
    out: set[str] = set()
    run: list[str] = []

    def flush():
        if not run:
            return
        words = [w for w in run if w != "and"]
        out.add(str(_run_value(words)))
        if lenient:
            out.update(str(_run_value([w])) for w in words if w not in _SCALES)
            # colloquial "one fifty" / "one twenty five" = 150 / 125
            if len(words) >= 2 and words[0] in _UNITS and 0 < _UNITS[words[0]] < 10 and "hundred" not in words:
                out.add(str(_UNITS[words[0]] * 100 + _run_value(words[1:])))
        run.clear()

    for tok in _TOKEN_RE.findall(text.lower().replace("-", " ")):
        if tok in _UNITS or tok in _TENS or tok in _SCALES or (tok == "and" and run and "hundred" in run):
            run.append(tok)
            continue
        flush()
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            out.add(str(float(tok)).removesuffix(".0"))
    flush()
    return out


def ungrounded_critical_tokens(item_text: str, premise: str) -> list[str]:
    """Numbers / laterality words in the note line that the cited text never mentions."""
    missing = sorted(_numbers(item_text) - _numbers(premise, lenient=True), key=lambda x: float(x))
    item_words = set(re.findall(r"[a-z]+", item_text.lower()))
    prem_words = set(re.findall(r"[a-z]+", premise.lower()))
    missing += sorted((item_words & _LATERALITY) - prem_words)
    return missing
