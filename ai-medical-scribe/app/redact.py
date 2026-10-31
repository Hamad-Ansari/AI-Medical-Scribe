"""PII redaction: Presidio + regex, tuned for lowercase ASR-style transcripts.

What is redacted (placeholder):
    person names            -> [NAME]
    phone numbers           -> [PHONE]   (digits or spoken: "five five five ...")
    email addresses         -> [EMAIL]   (literal or spoken: "jo at gmail dot com")
    IDs (MRN, SSN, member/policy/account numbers, long alphanumeric codes) -> [ID]

Why not just Presidio's defaults?  ACI-Bench (like real ASR output) is
lower-case: "hi , brian . how are you ?". spaCy NER misses many lowercase names
and tags clinical words ("palmar", "biopsy") as PERSON. So names come from three
sources, merged by Presidio:
  1. spaCy PERSON (via Presidio), kept only if a token is capitalised in the
     source or is a known name - this drops clinical-word false positives;
  2. a name gazetteer (Faker first/last names, several Latin-script locales).
     Names that double as common words ("ray" in "x-ray", "summer", "mark",
     "grace") only count with context;
  3. context patterns: titles ("dr. ..."), greetings ("hi , ..."),
     "my name is ...", "... is a 58 year old".

Clinical tokens are protected: doses ("20 mg"), vitals ("150/90"), lab names
("a1c", "b12"), spine levels ("l4-l5") are never treated as IDs.

Redaction happens before anything is stored or logged (see
:class:`RedactingFilter` for logs). Originals are never returned or persisted.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache

from app.schema import Segment

PLACEHOLDERS = {"PERSON": "[NAME]", "PHONE_NUMBER": "[PHONE]", "EMAIL_ADDRESS": "[EMAIL]", "ID": "[ID]"}
_PLACEHOLDER_RE = re.compile(r"\[(?:NAME|PHONE|EMAIL|ID)\]")

# --------------------------------------------------------------------- lexicons
# First names that are also everyday / clinical words. They are redacted only
# with context (title, greeting, "my name is", or followed by a known surname).
AMBIGUOUS_NAMES = frozenset("""
ray summer april may june august autumn winter guy mark frank hunter diamond drew dawn jean chase grace jack
will bill art hope joy faith rose ivy holly iris penny sandy rusty sky star angel baby precious destiny
major royal king prince lane page rich max cash price miles bob carol crystal amber ruby pearl jade
gene sue pat don dean earl wade lee ross long young white brown black green gray grey hill wood bell
stone ward rob reed paris ella eve trinity christian justice august harmony heaven melody misty sage
jordan taylor morgan kelly casey tracy terry carter mason parker cooper hunter tanner bailey sydney
nick victor norman sharon dale glen colby stuart wayne austin dallas houston brooklyn london cameron
""".split()) | frozenset(
    # + every gazetteer name that is also an everyday word ("lot", "else", "job",
    # "day", "kick", "sharp"); without this ~half of ACI PERSON hits were words.
    ln.strip() for ln in (Path(__file__).parent / "data" / "common_word_names.txt").read_text().splitlines()
    if ln.strip() and not ln.startswith("#")
)

# never names, whatever spaCy or a greeting pattern says
NOT_NAMES = frozenset("""
there everyone everybody again guys folks doctor doc nurse dragon dax sir maam ma'am honey sweetie buddy
dear man mom dad sweetheart hun friend all you so okay ok well yes no um uh hi hello hey morning afternoon
evening patient he she they it this that who what how good great thanks thank sorry please today now
bye goodbye listen look wait see alright perfect awesome cool sure right wow oh ah ow ouch yep nope yeah
come sit welcome everything nice fantastic excellent wonderful lovely gosh geez anyway actually
mr mrs ms dr prof mister miss aunt uncle
gerd alfredo camila
""".split())

# ACI-Bench tokenises "mr. watson" as "mr . watson", hence \s*\.?
_TITLE_RE = r"(?:dr|doctor|mr|mrs|ms|miss|mister|nurse|professor|prof)\s*\.?"
# abbreviated titles are reliable enough to mark an *unknown* following word as a
# surname ("dr. okonkwo"); "doctor"/"nurse" are not ("the nurse told you").
_ABBR_TITLE_RE = r"(?<![a-z])(?:dr|mr|mrs|ms|prof)\s*\.?"
_RELATION_RE = (r"\b(?:my|his|her|your|their|our)\s+(?:wife|husband|son|daughter|mother|father|mom|dad|sister|brother|"
                r"partner|friend|caregiver|grandson|granddaughter|grandmother|grandfather|aunt|uncle|niece|nephew|fiance|fiancee)")
_GREETING_RE = r"(?:hi|hello|hey|good morning|good afternoon|good evening|morning|thanks|thank you|bye|goodbye|welcome)"

_NAME_LOCALES = ("en_US", "en_GB", "en_IN", "es_MX", "es_ES", "fr_FR", "de_DE", "it_IT", "pt_BR", "nl_NL")


def _faker_names(attr_groups: tuple[str, ...]) -> set[str]:
    import importlib

    out: set[str] = set()
    for loc in _NAME_LOCALES:
        try:
            prov = importlib.import_module(f"faker.providers.person.{loc}").Provider
        except ImportError:
            continue
        for attr in attr_groups:
            v = getattr(prov, attr, None)
            if v:
                out.update(str(k).lower() for k in (v.keys() if hasattr(v, "keys") else v))
    return {n for n in out if re.fullmatch(r"[^\W\d_][^\W\d_'\-]+", n)}


@lru_cache(maxsize=1)
def name_lexicons() -> tuple[frozenset[str], frozenset[str]]:
    """(first_names, last_names), lowercase, excluding NOT_NAMES."""
    first = _faker_names(("first_names", "first_names_male", "first_names_female"))
    last = _faker_names(("last_names",))
    return frozenset(first - NOT_NAMES), frozenset(last - NOT_NAMES)


# ---------------------------------------------------------------- regex rules
_DIGIT_WORDS = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
_ID_KEYWORDS = (
    r"(?:mrn|medical record(?: number)?|record number|patient id|patient number|chart number|member id|"
    r"member number|insurance id|insurance number|policy(?: number)?|account(?: number)?|ssn|social security"
    r"(?: number)?|id number|id|license(?: number)?|passport(?: number)?|case number)"
)

REGEX_RULES: list[tuple[str, re.Pattern[str]]] = [
    # literal email
    ("EMAIL_ADDRESS", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.I)),
    # spoken email: "john dot smith at gmail dot com"
    ("EMAIL_ADDRESS", re.compile(r"\b[a-z0-9][a-z0-9._-]*(?:\s+dot\s+[a-z0-9_-]+)*\s+at\s+[a-z0-9-]+(?:\s+dot\s+(?:com|org|net|edu|gov|co|uk|io|us|ca|pk))+\b", re.I)),
    # IDs introduced by a keyword: "mrn 4829301", "member id is ab-77812", "ssn 123-45-6789"
    ("ID", re.compile(rf"\b{_ID_KEYWORDS}\b(?:\s*(?:number|no\.?|#|is|:|was|of))*\s*([a-z0-9][a-z0-9\-]{{2,}}[0-9][a-z0-9\-]*)", re.I)),
    # date of birth (strong identifier) -> [ID]
    ("ID", re.compile(r"\b(?:date of birth|dob|d\.o\.b\.|born on)\s*(?:is|was|:)?\s*((?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})|(?:[a-z]+\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4}))", re.I)),
    # US SSN pattern
    ("ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # phone (digits): (555) 123-4567, 555-123-4567, +1 555 123 4567, 555.123.4567
    ("PHONE_NUMBER", re.compile(r"(?<![\w/])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?:\s*(?:x|ext\.?)\s*\d{1,5})?(?![\w/])", re.I)),
    # phone (spoken): 7+ digit words in a row
    ("PHONE_NUMBER", re.compile(rf"\b{_DIGIT_WORDS}(?:[\s,-]+{_DIGIT_WORDS}){{6,}}\b", re.I)),
    # long alphanumeric codes with >=4 digits and a letter ("AB1234567", "X9Y8Z7W6")
    ("ID", re.compile(r"\b(?=[a-z0-9-]*[a-z])(?=(?:[a-z-]*\d){4})[a-z0-9]{2,}(?:-?[a-z0-9]+)+\b|\b(?=[a-z]*\d)(?=(?:[a-z]*\d){4})[a-z]{1,3}\d{4,}[a-z]?\b", re.I)),
    # long bare digit runs (8+) that are not phones
    ("ID", re.compile(r"(?<![\d./-])\d{8,}(?![\d./-])")),
]

# tokens that look like codes but are clinical vocabulary
_CLINICAL_CODE_RE = re.compile(
    r"^(?:hba1c|a1c|b12|b6|d3|covid-?19|t[1-9]|l[1-5]|c[1-7]|s[1-5]|g[1-6]p[0-9]+|ct|mri|ekg|ecg|"
    r"[a-z]\d-[a-z]\d|\d+(?:mg|mcg|ml|cc|mmhg|kg|lb|lbs|cm|mm|iu|units?)|(?:19|20)\d0s|covid)$",
    re.I,
)


@dataclass
class Span:
    start: int
    end: int
    entity: str
    score: float
    source: str


@dataclass
class RedactionResult:
    text: str
    count: int
    counts_by_type: dict[str, int] = field(default_factory=dict)


# ----------------------------------------------------------- name detection
def _at_sentence_start(text: str, i: int) -> bool:
    j = i - 1
    while j >= 0 and text[j] in " \t":
        j -= 1
    return j < 0 or text[j] in ".!?]\n"


_WORD_RE = re.compile(r"[^\W\d_][^\W\d_'\-]*(?:['\-][^\W\d_]+)*")


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    from spacy.lang.en.stop_words import STOP_WORDS  # plain word list, no model load

    return frozenset(STOP_WORDS) | frozenset("fine good great okay going doing feeling sure ready glad sorry here back".split())


def _is_known_name(word: str) -> bool:
    first, last = name_lexicons()
    w = word.lower()
    return w in first or w in last


def _gazetteer_names(text: str) -> list[Span]:
    """Lowercase-robust name finder: gazetteer + context patterns."""
    first, last = name_lexicons()
    spans: list[Span] = []
    words = [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]
    lower = text.lower()
    for k, (s, e, w) in enumerate(words):
        wl = w.lower().strip("'-")
        if wl in NOT_NAMES or len(wl) < 2 or ("'" in wl and wl.split("'")[0] in _stop_words()):
            continue  # also "I'm", "it's", "you'll" - capitalised by Whisper but never names
        # "x-ray": the word right after "x-"
        if s >= 2 and lower[s - 2 : s] == "x-":
            continue
        # false start: "mira- miralax", "ale- , aleve"
        if text[e : e + 1] == "-" and k + 1 < len(words) and words[k + 1][2].lower().startswith(wl):
            continue
        # capitalisation is a name signal only mid-sentence (Whisper capitalises
        # sentence starts: "Lot of people...", "Day three...")
        cap = w[0].isupper() and not _at_sentence_start(text, s)
        # previous word, only if separated by whitespace ("allen . date" is not a name pair)
        prev = words[k - 1][2].lower() if k and not text[words[k - 1][1] : s].strip() else ""
        before = lower[max(0, s - 25) : s]
        nxt = words[k + 1][2].lower() if k + 1 < len(words) else ""
        titled = bool(re.search(rf"\b{_TITLE_RE}\s*$", before))
        abbr_titled = bool(re.search(rf"{_ABBR_TITLE_RE}\s*$", before))
        related = bool(re.search(rf"{_RELATION_RE}\s*,?\s*(?:is\s+)?$", before))
        appositive = bool(re.search(rf"{_RELATION_RE}\s*,\s*$", before))  # "my wife , ayesha siddiqui"
        greeted = bool(re.search(rf"(?:^|[.?!,]\s*)(?:{_GREETING_RE}\s*,?\s*)+$", before))  # "hey hi bobby"
        # "so , roger , i heard ..." / "so john , what's going on": a gazetteer word
        # boxed between a discourse marker and a comma is being addressed
        marker_vocative = bool(
            re.search(r"(?:^|[.?!,]\s*)(?:so|okay|ok|alright|all right)\s*,?\s*$", before)
            and re.match(r"\s*,", text[e : e + 3])
        )
        strong_named = bool(re.search(r"\b(?:my name is|my name's|name is|call me|meet|named|goes by)\s+$", before))
        # ACI dictation: "next patient is grace ross", "dictating on donald clark" -
        # unlocks gazetteer words only ("the patient is alert" must survive)
        dictation = bool(re.search(
            r"(?:\b(?:next|new) patient(?: is)?|(?<!the )(?<!this )(?<!our )\bpatient is|patient's name is|dictating on)\s*,?\s*$",
            before))
        weak_named = bool(re.search(r"\b(?:this is|i'm|i am|it's)\s+$", before))
        # vocative: "hi , oluwaseun ," / "thanks , ayesha ."
        after = text[e : e + 3]
        run_len = _vocative_run(words, k, text) if (greeted or appositive) else 0
        vocative = run_len > 0
        if run_len > 1 and wl not in AMBIGUOUS_NAMES and not (wl in _stop_words()):
            # multi-word name in the vocative slot: "hi , nguyen thi lan ." -> all words
            for j in range(k, k + run_len):
                spans.append(Span(words[j][0], words[j][1], "PERSON", 0.75, "vocative"))
            continue
        is_first, is_last = wl in first, wl in last
        # two ambiguous words are not a full name ("a lot more", "little wood")
        followed_by_surname = nxt in last and nxt not in NOT_NAMES and not (wl in AMBIGUOUS_NAMES and nxt in AMBIGUOUS_NAMES)
        turn_initial = not text[max(0, s - 3) : s].strip(" ,.") and (s < 3 or not text[:s].strip(" ,."))
        initial_vocative = (is_first or is_last) and turn_initial and bool(
            # "nelson , how are you ?" - ambiguous words need a greeting-like continuation
            re.match(r"\s*,", text[e : e + 3]) if wl not in AMBIGUOUS_NAMES
            else re.match(r"\s*,\s*(?:how|nice|good|great|thanks|thank|it's|i'm|come|have|please)\b", text[e : e + 20])
        )
        if marker_vocative and (is_first or is_last):
            initial_vocative = True
        unambiguous_name = (is_first or is_last) and wl not in AMBIGUOUS_NAMES
        named = (strong_named or initial_vocative or (dictation and (is_first or is_last))
                 or (weak_named and (unambiguous_name or (cap and wl not in AMBIGUOUS_NAMES))))
        if related and (is_first or is_last) and wl not in AMBIGUOUS_NAMES:
            named = True
        context = titled or greeted or named
        common_word = wl in _stop_words()
        # capitalisation alone never unlocks an ambiguous word (Whisper: "since May", "Labor Day")
        if is_first and (wl not in AMBIGUOUS_NAMES or context or followed_by_surname):
            spans.append(Span(s, e, "PERSON", 0.85, "gazetteer"))
        elif is_last and (titled or named or (prev in first and prev not in AMBIGUOUS_NAMES) or (cap and wl not in AMBIGUOUS_NAMES)):
            spans.append(Span(s, e, "PERSON", 0.8, "gazetteer"))
        elif abbr_titled and wl not in AMBIGUOUS_NAMES and len(wl) > 2 and wl not in _stop_words():
            # "dr. okonkwo": unknown word after a title is a surname
            spans.append(Span(s, e, "PERSON", 0.75, "title"))
        elif (strong_named or vocative) and len(wl) > 2 and not common_word and wl not in AMBIGUOUS_NAMES:
            spans.append(Span(s, e, "PERSON", 0.75, "context"))
        elif greeted and len(wl) > 2 and not common_word and (cap or is_first or is_last):
            spans.append(Span(s, e, "PERSON", 0.7, "context"))
        # surname right after a detected first name (unknown surname, e.g. "ayesha siddiqui")
        if spans and spans[-1].end == e and spans[-1].source != "pair" and k + 1 < len(words):
            ns, ne, nw = words[k + 1]
            nwl = nw.lower()
            gap = text[e:ns]
            # "my name is ayesha siddiqui" / "hi , olusegun bello ."
            unknown_ok = (strong_named or vocative) and nwl not in _stop_words()
            strong_ctx = strong_named or dictation or titled or vocative or related
            ambiguous_ok = nwl not in AMBIGUOUS_NAMES or (strong_ctx and nwl in last and nwl not in _stop_words())
            pronoun = nwl.split("'")[0] in _stop_words()  # "karen I'm ...", "sophia I see"
            if gap.strip() == "" and not pronoun and nwl not in NOT_NAMES and ambiguous_ok and (
                nwl in last or nw[0].isupper() or unknown_ok
            ):
                spans.append(Span(ns, ne, "PERSON", 0.75, "pair"))
    # "kelly wood . date of birth ...", "maria martin , mrn ..." (ACI dictation)
    first_l, last_l = first, last
    dob_rx = (r"\b([^\W\d_][\w'\-]+)(?:\s+([^\W\d_][\w'\-]+))?\s*[,.:]?\s*"
              r"(?:date of birth|birth date|dob|d o b|mrn|medical record)\b")
    for m in re.finditer(dob_rx, text, re.I):
        for g in (1, 2):
            tok = m.group(g)
            tl = tok.lower() if tok else ""
            if tok and (tl in first_l or tl in last_l) and tl not in NOT_NAMES and tl not in _stop_words():
                spans.append(Span(m.start(g), m.end(g), "PERSON", 0.75, "dob-pattern"))
    # "brian is a 58 year old" (ACI dictation style) for unknown names
    # "brian is a 58 year old", "edwards is a pleasant , 32-year-old", "beverly's a 24 year old"
    age_rx = (r"\b([^\W\d_][\w'\-]{2,}?)(?:\s+([^\W\d_][\w'\-]{2,}?))?(?:'s|\s+is)\s+(?:an?\s+)?"
              r"(?:(?:very|pleasant|nice|delightful|lovely|healthy|and|,)\s*){0,4}\d+[- ]year[- ]old")
    skip = NOT_NAMES | {"patient", "female", "male", "man", "woman", "boy", "girl", "child", "baby", "son", "daughter"}
    for m in re.finditer(age_rx, text, re.I):
        for g in (1, 2):
            tok = m.group(g)
            if tok and tok.lower() not in skip and tok.lower() not in _stop_words() and not tok.endswith("-"):
                spans.append(Span(m.start(g), m.end(g), "PERSON", 0.7, "age-pattern"))
    return spans


def _vocative_run(words, k: int, text: str, max_len: int = 4) -> int:
    """Length n of words[k : k+n] (n <= max_len, no stop words) followed by punctuation, else 0.

    Matches the name slot in "hi , olusegun bello ." without matching
    "hi , how are you".
    """
    for n in range(1, max_len + 1):
        if k + n > len(words):
            return False
        w = words[k + n - 1][2].lower()
        if w in _stop_words() or w in NOT_NAMES:
            return 0
        end = words[k + n - 1][1]
        if re.match(r"\s*(?:[,.!?]|$)", text[end : end + 3]):
            return n
        if k + n < len(words) and text[end : words[k + n][0]].strip():
            return 0  # something other than whitespace between words
    return 0


def _regex_spans(text: str) -> list[Span]:
    spans = []
    for ent, rx in REGEX_RULES:
        for m in rx.finditer(text):
            s, e = (m.start(1), m.end(1)) if m.groups() and m.group(1) else (m.start(), m.end())
            tok = text[s:e]
            if ent == "ID" and _CLINICAL_CODE_RE.match(tok):
                continue
            if _PLACEHOLDER_RE.fullmatch(tok):
                continue
            spans.append(Span(s, e, ent, 0.9 if ent != "ID" else 0.8, "regex"))
    return spans


# -------------------------------------------------------------- presidio glue
@lru_cache(maxsize=1)
def _presidio():
    """Build Presidio engines once. spaCy small model keeps RAM low."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    })
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    return analyzer, AnonymizerEngine()


_PRESIDIO_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN"]


def _presidio_spans(text: str) -> list[Span]:
    analyzer, _ = _presidio()
    spans = []
    for r in analyzer.analyze(text=text, language="en", entities=_PRESIDIO_ENTITIES, score_threshold=0.4):
        ent = "ID" if r.entity_type == "US_SSN" else r.entity_type
        frag = text[r.start : r.end]
        if ent == "PERSON":
            # spaCy on lowercase text tags clinical words ("palmar", "deep tendon").
            # Keep only the words of the span that look like names: capitalised,
            # or a non-ambiguous first name. Lowercase surname-only words
            # ("cherry", "farmer", "wood") are too often ordinary words.
            first, _last = name_lexicons()
            for m in _WORD_RE.finditer(frag):
                w, wl = m.group(), m.group().lower()
                if wl in NOT_NAMES or wl in _stop_words() or wl.split("'")[0] in _stop_words() or len(wl) < 2:
                    continue  # also "I'm", "I'll"
                # capitals count only mid-sentence (Whisper/sentence-cased text)
                cap = w[0].isupper() and not _at_sentence_start(text, r.start + m.start())
                if (cap and wl not in AMBIGUOUS_NAMES) or (wl in first and wl not in AMBIGUOUS_NAMES):
                    spans.append(Span(r.start + m.start(), r.start + m.end(), "PERSON", float(r.score), "presidio"))
            continue
        if ent == "PHONE_NUMBER" and len(re.sub(r"\D", "", frag)) < 7:
            continue
        spans.append(Span(r.start, r.end, ent, float(r.score), "presidio"))
    return spans


def _merge(spans: list[Span]) -> list[Span]:
    """Resolve overlaps: longest span wins, ties -> higher score."""
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start), -s.score))
    out: list[Span] = []
    for sp in spans:
        if out and sp.start < out[-1].end:
            last = out[-1]
            if sp.end > last.end and sp.entity == last.entity:
                last.end = sp.end
            continue
        out.append(sp)
    return out


# ------------------------------------------------------------------ public API
def find_pii(text: str, use_nlp: bool = True) -> list[Span]:
    spans = _regex_spans(text) + _gazetteer_names(text)
    if use_nlp:
        spans += _presidio_spans(text)
    # don't re-redact placeholders
    ph = [(m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(text)]
    spans = [s for s in spans if not any(a <= s.start < b for a, b in ph)]
    return _merge(spans)


def redact_text(text: str, use_nlp: bool = True) -> RedactionResult:
    """Return redacted text and the number of replacements.

    ``use_nlp=False`` skips spaCy (regex + gazetteer only) - used for logs.
    """
    if not text:
        return RedactionResult(text="", count=0, counts_by_type={})
    spans = find_pii(text, use_nlp=use_nlp)
    out, pos = [], 0
    for sp in spans:
        out.append(text[pos : sp.start])
        out.append(PLACEHOLDERS[sp.entity])
        pos = sp.end
    out.append(text[pos:])
    # "[NAME] [NAME]" (first + last name) is one person -> one replacement
    redacted = re.sub(r"\[NAME\](?:\s+\[NAME\])+", "[NAME]", "".join(out))
    counts: dict[str, int] = {}
    for ph in PLACEHOLDERS.values():
        n = redacted.count(ph) - text.count(ph)  # pre-existing placeholders don't count
        if n > 0:
            counts[ph] = n
    return RedactionResult(text=redacted, count=sum(counts.values()), counts_by_type=counts)


def redact_segments(segments: list[Segment], use_nlp: bool = True) -> tuple[list[Segment], RedactionResult]:
    """Redact every segment; IDs, timings and speakers are preserved."""
    out: list[Segment] = []
    total: dict[str, int] = {}
    for seg in segments:
        r = redact_text(seg.text, use_nlp=use_nlp)
        out.append(seg.model_copy(update={"text": r.text}))
        for k, v in r.counts_by_type.items():
            total[k] = total.get(k, 0) + v
    return out, RedactionResult(text="", count=sum(total.values()), counts_by_type=total)


class RedactingFilter(logging.Filter):
    """Logging filter that redacts PII from every record (regex + gazetteer).

    Install with :func:`install_log_redaction`. Defence in depth: modules are
    already written to log counts only.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        red = redact_text(msg, use_nlp=False)
        if red.count:
            record.msg, record.args = red.text, ()
        return True


_FACTORY_INSTALLED = False


def _install_record_factory() -> None:
    """Redact every LogRecord at creation time, whatever logger/handler it uses.

    Logger-level filters do not see records propagated from child loggers, so a
    record factory is the only hook that covers everything.
    """
    global _FACTORY_INSTALLED
    if _FACTORY_INSTALLED:
        return
    base = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        rec = base(*args, **kwargs)
        try:
            msg = rec.getMessage()
        except Exception:
            return rec
        red = redact_text(msg, use_nlp=False)
        if red.count:
            rec.msg, rec.args = red.text, ()
        return rec

    logging.setLogRecordFactory(factory)
    _FACTORY_INSTALLED = True


def install_log_redaction(logger: logging.Logger | None = None) -> None:
    """Redact PII in logs: global record factory + filter on ``logger`` and its handlers."""
    _install_record_factory()
    target = logger or logging.getLogger()
    if not any(isinstance(f, RedactingFilter) for f in target.filters):
        target.addFilter(RedactingFilter())
    for h in target.handlers:
        if not any(isinstance(f, RedactingFilter) for f in h.filters):
            h.addFilter(RedactingFilter())
