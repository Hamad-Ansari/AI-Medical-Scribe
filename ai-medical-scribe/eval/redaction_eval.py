"""Redaction evaluation on synthetic data.

1. ACI-Bench recall: patient names taken from the properly-cased reference-note
   HPI ("Brian White is a 58-year-old ...") must be redacted wherever they occur
   in the lowercase dialogue.  NOTE: ACI-Bench names appear to come from the same
   Faker name lists used by our gazetteer, so this recall is optimistic.
2. Seeded synthetic PII (Faker, several locales incl. names NOT in the
   gazetteer) inserted into ACI-style sentences -> recall per type.
3. Over-redaction: every redaction across all 207 ACI dialogues, bucketed into
   gold patient name / other gazetteer name / everyday-word name / non-name, on
   the lowercase text and on sentence-capitalised (Whisper-like) text. The last
   two buckets are listed for manual review (clinical words must survive).

    python eval/redaction_eval.py [--seed 1234]
"""
from __future__ import annotations

import argparse
import collections
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.loader import SPLITS, load_split  # noqa: E402
from app.redact import find_pii, name_lexicons, redact_text  # noqa: E402

HPI = re.compile(r"(?:(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+)?([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+)?)\s+is\s+an?\s+\d+[- ]year[- ]old")

TEMPLATES = {
    "NAME": ["hi , {x} . how are you feeling today ?", "so , {x} is a 45 year old female here for follow-up .",
             "my name is {x} .", "dr. {x} will call you with the results .", "this is my wife , {x} ."],
    "PHONE": ["you can reach me at {x} .", "my cell is {x} , call anytime .", "the pharmacy number is {x} ."],
    "EMAIL": ["send the results to {x} please .", "my email is {x} ."],
    "ID": ["my medical record number is {x} .", "the insurance member id is {x} .", "my mrn is {x} ."],
}
# names deliberately outside the Faker gazetteer (generalisation check)
OOV_NAMES = ["ayesha siddiqui", "oluwaseun adeyemi", "wei zhang", "bilal chaudhry", "nguyen thi lan",
             "fatima zahra", "kwame mensah", "priya raghavan", "olusegun bello", "mei ling"]


HONORIFICS = {"at", "dot", "com", "dr", "mr", "mrs", "ms", "lic", "univ", "prof", "sc", "ing", "dipl", "mag", "phd",
              "md", "jr", "sr", "ii", "iii", "iv", "dds", "dvm", "mba", "ba", "bsc", "msc"}


def spoken_digits(num: str) -> str:
    w = "zero one two three four five six seven eight nine".split()
    return " ".join(w[int(c)] for c in num if c.isdigit())


def synth_pii(seed: int, n: int = 40):
    from faker import Faker

    fk = Faker(["en_US", "en_GB", "es_MX", "de_DE"])
    fk.seed_instance(seed)
    rnd = random.Random(seed)
    cases = []
    for _ in range(n):
        name = rnd.choice([fk.name().lower(), fk.first_name().lower(), fk.name(), rnd.choice(OOV_NAMES)])
        cases.append(("NAME", name, rnd.choice(TEMPLATES["NAME"]).format(x=name)))
        ph = rnd.choice([fk.numerify("(###) ###-####"), fk.numerify("###-###-####"), fk.numerify("###.###.####"),
                         fk.numerify("+1 ### ### ####"), spoken_digits(fk.numerify("##########"))])
        cases.append(("PHONE", ph, rnd.choice(TEMPLATES["PHONE"]).format(x=ph)))
        user = fk.user_name()
        em = rnd.choice([fk.email(), f"{user} at gmail dot com"])
        cases.append(("EMAIL", em, rnd.choice(TEMPLATES["EMAIL"]).format(x=em)))
        idv = rnd.choice([fk.numerify("########"), fk.bothify("??-######").lower(), fk.numerify("###-##-####"), fk.bothify("???#######")])
        cases.append(("ID", idv, rnd.choice(TEMPLATES["ID"]).format(x=idv)))
    return cases


def eval_aci() -> dict:
    cases = [c for sp in SPLITS for c in load_split(sp)]
    hits = total = 0
    misses = collections.Counter()
    for c in cases:
        m = HPI.search(c.reference_note)
        if not m:
            continue
        toks = [t.lower() for t in m.group(1).split()]
        for seg in c.segments:
            low = seg.text.lower()
            occ = [t for t in toks if re.search(rf"\b{re.escape(t)}\b", low)]
            if not occ:
                continue
            red = redact_text(seg.text).text.lower()
            for t in occ:
                n_before = len(re.findall(rf"\b{re.escape(t)}\b", low))
                n_after = len(re.findall(rf"\b{re.escape(t)}\b", red))
                total += n_before
                hits += n_before - n_after
                if n_after:
                    misses[t] += n_after
    return {"recall": hits / max(total, 1), "occurrences": total, "missed": misses.most_common(10)}


def eval_synthetic(seed: int) -> dict:
    per = collections.defaultdict(lambda: [0, 0])
    oov = [0, 0]
    leaks = []
    for typ, value, sent in synth_pii(seed):
        red = redact_text(sent).text
        # every alphanumeric token of the PII value must be gone
        # honorifics/degrees that Faker adds ("lic.", "univ.prof.", "b.sc.") are not identifying
        toks = [t for t in re.findall(r"[a-z0-9]+", value.lower()) if t not in HONORIFICS and len(t) > 1]
        if typ == "PHONE" and not value[0].isdigit() and not value.startswith(("(", "+")):
            toks = ["".join(value.split())]  # spoken digits: judge the run as a whole
            ok = "[PHONE]" in red
        else:
            ok = all(not re.search(rf"\b{re.escape(t)}\b", red.lower()) for t in toks)
        per[typ][0] += ok
        per[typ][1] += 1
        if typ == "NAME" and value.lower() in OOV_NAMES:
            oov[0] += ok
            oov[1] += 1
        if not ok:
            leaks.append((typ, sent, red))
    return {"per_type": {k: v[0] / v[1] for k, v in per.items()}, "n": {k: v[1] for k, v in per.items()},
            "oov_name_recall": oov[0] / max(oov[1], 1), "oov_n": oov[1], "leaks": leaks}


def eval_over_redaction(cased: bool = False) -> dict:
    """Classify every redaction in all 207 ACI dialogues.

    gold         the case's own patient name (from the reference-note HPI)
    other_name   another gazetteer name (doctors, nurses, relatives) - presumed correct
    common_word  contains an everyday English word that is also a name ("lot", "ward")
                 -> listed for review; the bulk of these used to be false positives
    non_name     not in the gazetteer at all -> listed for review
    ``cased=True`` first sentence-capitalises the text (tts.speakable) to mimic Whisper.
    """
    from app.redact import AMBIGUOUS_NAMES
    from app.tts import speakable

    first, last = name_lexicons()
    cases = [c for sp in SPLITS for c in load_split(sp)]
    buckets = collections.Counter()
    review = collections.Counter()
    for c in cases:
        gold = {w.lower() for m in HPI.finditer(c.reference_note) for w in m.group(1).split()}
        for seg in c.segments:
            text = speakable(seg.text) if cased else seg.text
            for sp in find_pii(text):
                frag = text[sp.start : sp.end].lower()
                if sp.entity != "PERSON":
                    buckets[sp.entity] += 1
                    continue
                words = re.findall(r"[^\W\d_]+", frag)
                if words and all(w in gold for w in words):
                    buckets["gold"] += 1
                elif any(w in AMBIGUOUS_NAMES for w in words):
                    buckets["common_word"] += 1
                    review[("common_word", frag)] += 1
                elif all(w in first or w in last for w in words):
                    buckets["other_name"] += 1
                else:
                    buckets["non_name"] += 1
                    review[("non_name", frag)] += 1
    return {"buckets": dict(buckets), "review": review.most_common(60)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    r1 = eval_aci()
    print(f"ACI-Bench patient-name recall: {r1['recall']:.1%} of {r1['occurrences']} occurrences; missed {r1['missed']}")
    r2 = eval_synthetic(a.seed)
    print("Seeded synthetic PII recall:", {k: f"{v:.0%} (n={r2['n'][k]})" for k, v in r2["per_type"].items()})
    print(f"  out-of-gazetteer names: {r2['oov_name_recall']:.0%} (n={r2['oov_n']})")
    for typ, sent, red in r2["leaks"]:
        print(f"  LEAK {typ}: {sent!r} -> {red!r}")
    for cased in (False, True):
        r3 = eval_over_redaction(cased)
        label = "sentence-cased (Whisper-like)" if cased else "lowercase (as in ACI)"
        print(f"\nOver-redaction, {label}: {r3['buckets']}")
        for (kind, frag), n in r3["review"]:
            print(f"  {n:3d} {kind:11s} {frag!r}")
