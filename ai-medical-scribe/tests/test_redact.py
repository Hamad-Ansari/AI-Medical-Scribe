"""Redaction tests (phase 3) - seeded fake PII, never real data."""
import logging
import random

import pytest
from faker import Faker

from app.redact import RedactingFilter, install_log_redaction, redact_segments, redact_text
from app.schema import Segment

SEED = 1234


@pytest.fixture(scope="module")
def fake():
    f = Faker("en_US")
    f.seed_instance(SEED)
    return f


def _gone(value: str, text: str) -> bool:
    return value.lower() not in text.lower()


# ---------------------------------------------------------------- names
@pytest.mark.parametrize("i", range(10))
def test_seeded_names_lowercase_greeting(fake, i):
    first, last = fake.first_name(), fake.last_name()
    r = redact_text(f"hi , {first.lower()} {last.lower()} . how are you today ?")
    assert _gone(first, r.text) and _gone(last, r.text), r.text
    assert r.text.startswith("hi , [NAME] .") and r.count == 1  # first+last = one person


@pytest.mark.parametrize("i", range(10))
def test_seeded_names_capitalised(fake, i):
    name = fake.name()
    r = redact_text(f"{name} is here for a follow-up visit.")
    for part in name.replace(".", " ").split():
        if len(part) > 2 and part.lower() not in {"mr", "mrs", "ms", "dr", "md", "phd", "dds", "dvm", "jr", "sr", "ii", "iii", "iv"}:
            assert _gone(part, r.text), (name, r.text)


@pytest.mark.parametrize("text,expected", [
    ("so , brian is a 58 year old male with heart failure .", "so , [NAME] is a 58 year old male with heart failure ."),
    ("hello , mr . watson . nice to meet you .", "hello , mr . [NAME] . nice to meet you ."),
    ("dr. okonkwo will review the scan .", "dr. [NAME] will review the scan ."),
    ("my name is ayesha siddiqui .", "my name is [NAME] ."),            # names outside the gazetteer
    ("hi , oluwaseun , come on in .", "hi , [NAME] , come on in ."),
    ("this is my wife , nguyen thi lan .", "this is my wife , [NAME] ."),
    ("nelson , how are you ?", "[NAME] , how are you ?"),
    ("edwards is a pleasant , 32-year-old female .", "[NAME] is a pleasant , 32-year-old female ."),
])
def test_contextual_names(text, expected):
    assert redact_text(text).text == expected


# --------------------------------------------------------- phones / emails / IDs
@pytest.mark.parametrize("fmt", ["(###) ###-####", "###-###-####", "###.###.####", "+1 ### ### ####"])
def test_seeded_phones(fake, fmt):
    ph = fake.numerify(fmt)
    r = redact_text(f"you can call me at {ph} anytime .")
    assert _gone(ph, r.text) and "[PHONE]" in r.text and r.counts_by_type == {"[PHONE]": 1}


def test_spoken_phone():
    r = redact_text("my number is five five five one two three four five six seven .")
    assert r.text == "my number is [PHONE] ."


@pytest.mark.parametrize("i", range(5))
def test_seeded_emails(fake, i):
    em = fake.email()
    r = redact_text(f"send it to {em} please")
    assert _gone(em, r.text) and r.counts_by_type == {"[EMAIL]": 1}


def test_spoken_email():
    assert redact_text("email me at jo dot smith at gmail dot com .").text == "email me at [EMAIL] ."


@pytest.mark.parametrize("text", [
    "my mrn is {d8} .", "medical record number {d8} .", "the member id is {code} .",
    "ssn {ssn} .", "my insurance id number is {code} .", "date of birth 04/12/1934 .",
])
def test_seeded_ids(fake, text):
    t = text.format(d8=fake.numerify("########"), code=fake.bothify("??-######").lower(), ssn=fake.ssn())
    r = redact_text(t)
    assert "[ID]" in r.text, (t, r.text)
    assert not any(ch.isdigit() for ch in r.text), r.text


# --------------------------------------------------------- clinical text survives
@pytest.mark.parametrize("text", [
    "your a1c was 6.5 and blood pressure is 150/90 .",
    "take lisinopril 20 mg daily and vitamin d 2000iu .",
    "disc bulge at l4-l5 , b12 is low , covid-19 negative .",
    "we will see you in may , i'll mark it in the chart .",
    "let's get an x-ray of the knee . hey dragon , show me the chest x-ray .",
    "the nurse told you about dax . the doctor said it was fine .",
    "deep tendon reflexes are 2+ . palmar erythema . clark level 2 .",
    "the patient is a 45 year old female .",
    "i am fine , this is going to hurt a bit .",
    "hi , how are you ? thanks , see you soon .",
])
def test_clinical_text_untouched(text):
    r = redact_text(text)
    assert r.text == text and r.count == 0, r.text


# --------------------------------------------------------- API behaviour
def test_mixed_counts(fake):
    t = f"Hi {fake.name()}, call {fake.numerify('###-###-####')} or {fake.email()}; mrn {fake.numerify('########')}."
    r = redact_text(t)
    assert r.counts_by_type == {"[NAME]": 1, "[PHONE]": 1, "[EMAIL]": 1, "[ID]": 1}
    assert r.count == 4


def test_idempotent():
    once = redact_text("hi , brian . call 555-123-4567 .").text
    twice = redact_text(once)
    assert twice.text == once and twice.count == 0


def test_empty_text():
    r = redact_text("")
    assert r.text == "" and r.count == 0


def test_redact_segments_preserves_structure():
    segs = [Segment(id=1, speaker="doctor", text="hi , brian .", start=0.0, end=1.2),
            Segment(id=2, speaker="patient", text="my knee hurts .", start=1.3, end=2.0)]
    out, res = redact_segments(segs)
    assert [s.id for s in out] == [1, 2] and out[0].start == 0.0 and out[0].speaker == "doctor"
    assert out[0].text == "hi , [NAME] ." and out[1].text == "my knee hurts ."
    assert res.count == 1 and res.counts_by_type == {"[NAME]": 1}
    assert segs[0].text == "hi , brian ."  # input not mutated


def test_log_filter_redacts(caplog, fake):
    logger = logging.getLogger("scribe.test")
    install_log_redaction(logger)
    phone, email = fake.numerify("###-###-####"), fake.email()
    with caplog.at_level(logging.INFO, logger="scribe.test"):
        logger.info("patient called from %s, email %s", phone, email)
    msg = caplog.records[-1].getMessage()
    assert phone not in msg and email not in msg and "[PHONE]" in msg and "[EMAIL]" in msg
    assert sum(isinstance(f, RedactingFilter) for f in logger.filters) == 1
    install_log_redaction(logger)  # idempotent
    assert sum(isinstance(f, RedactingFilter) for f in logger.filters) == 1


def test_random_seeded_sentences_never_leak_phone_or_email():
    f = Faker(["en_US", "en_GB"])
    f.seed_instance(SEED)
    rnd = random.Random(SEED)
    for _ in range(50):
        ph, em = f.numerify(rnd.choice(["(###) ###-####", "###-###-####"])), f.email()
        r = redact_text(f"{f.sentence()} reach me at {ph} or {em}. {f.sentence()}")
        assert ph not in r.text and em not in r.text


# --- over-redaction regressions: everyday words that are also Faker names ---
@pytest.mark.parametrize("text", [
    "i'm traveling a lot that week for work .",
    "anyone else sick in your family ?",
    "i think you're doing a good job .",
    "i've been drinking a lot more water .",
    "history of gout , gerd , and diabetes .",          # GERD is a diagnosis
    "your legs kick out ? it's sharp , and short of breath .",
    "mira- miralax will help . ale- , aleve is fine .",  # false starts
    "Uh, probably since Labor Day, so about five weeks or so.",  # Whisper casing
    "It started in May, then got worse in June.",
    "Lot of people say that. Anything else?",
])
def test_everyday_words_survive(text):
    r = redact_text(text)
    assert r.count == 0, r.text


@pytest.mark.parametrize("text, leaked", [
    ("so john , what's going on with your back ?", "john"),
    ("so , roger , i heard you went to the er .", "roger"),
    ("hey hi bobby how are you doing", "bobby"),
    ("nelson , how are you ?", "nelson"),
    ("next patient is grace ross . date of birth 8-23-97 .", "ross"),
    ("dr. jacob ward referred her .", "ward"),
    ("So, Roger, I heard you went to the ER.", "roger"),
])
def test_ambiguous_names_caught_in_context(text, leaked):
    assert leaked not in redact_text(text).text.lower()


def test_whisper_cased_pronouns_not_names():
    assert redact_text("Hey, I'm okay. Good to see you.").count == 0
    out = redact_text("So sophia I see that you hurt your knee. Alright karen I'm just gonna look.").text
    assert "I see" in out and "I'm just" in out and "sophia" not in out and "karen" not in out
