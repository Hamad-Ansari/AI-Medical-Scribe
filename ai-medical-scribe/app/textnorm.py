"""Text normalisation for WER: makes ACI-Bench references and Whisper output comparable.

Removes formatting differences that are not recognition errors: case,
punctuation, ACI tokenisation ("do n't", "gon na"), fillers ("um", "mm-hmm"),
spoken vs written numbers ("fifty eight" == "58"), hyphens ("58-year-old").
"""
from __future__ import annotations

import re

from app.evidence import _SCALES, _TENS, _UNITS, _run_value

FILLERS = frozenset("um uh uhm hmm mm mhm mmm mmmm hm ah oh er erm".split())
_CONTRACT = [(r"\b(\w+) n't\b", r"\1n't"), (r"\b(gon|wan|got) na\b", r"\1na"), (r"\s+'(s|re|ve|ll|d|m)\b", r"'\1")]
_SPELLING = {"alright": "all right", "okay": "ok", "ok": "ok", "whats": "what's", "gonna": "going to",
             "wanna": "want to", "gotta": "got to", "kinda": "kind of", "x-ray": "xray", "x ray": "xray",
             "e-mail": "email", "mm-hmm": "", "uh-huh": "", "mm-mm": ""}


def _numbers_to_digits(words: list[str]) -> list[str]:
    out, run = [], []

    def flush():
        if run:
            out.append(str(_run_value([w for w in run if w != "and"])))
            run.clear()

    for w in words:
        if w in _UNITS or w in _TENS or w in _SCALES or (w == "and" and run and "hundred" in run):
            run.append(w)
        else:
            flush()
            out.append(w)
    flush()
    return out


def normalize_for_wer(text: str) -> str:
    t = text.lower()
    for a, b in _CONTRACT:
        t = re.sub(a, b, t)
    for a, b in _SPELLING.items():
        t = re.sub(rf"(?<![\w-]){re.escape(a)}(?![\w-])", b, t)
    t = re.sub(r"(\d)-(?=\w)", r"\1 ", t)          # 58-year-old -> 58 year old
    t = re.sub(r"\b(\d+(?:\.\d+)?)(?=[a-z])", r"\1 ", t)  # whisper "1000mg" -> 1000 mg (a1c kept)
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)     # year-old -> year old
    t = re.sub(r"[^\w\s'.%/]", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)       # keep decimals like 6.5
    words = [w.strip("'") for w in t.split()]
    words = [w for w in words if w and w not in FILLERS]
    return " ".join(_numbers_to_digits(words))
