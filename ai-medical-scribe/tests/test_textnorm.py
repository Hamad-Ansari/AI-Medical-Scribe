from app.textnorm import normalize_for_wer

n = normalize_for_wer


def test_aci_and_whisper_styles_match():
    aci = "so , brian is a 58 year old male . i do n't know , mm-hmm . um , fifty eight . whats going on ? gon na get an x-ray ."
    whisper = "So, Brian is a 58-year-old male. I don't know, MMMM. Um, 58. What's going on? Gonna get an X-ray."
    assert n(aci) == n(whisper)


def test_decimals_and_numbers():
    assert n("a1c was six point five") != n("a1c was 6.5")  # "point" not converted - documented limitation
    assert n("one hundred and twenty over eighty") == "120 over 80"
    assert n("A1c was 6.5.") == "a1c was 6.5"


def test_glued_units_split():
    # Whisper writes "1000mg"; ACI writes "1000 mg" - must not count as a lost number
    assert normalize_for_wer("metformin to 1000mg twice a day") == normalize_for_wer("metformin to 1000 mg twice a day")
    assert normalize_for_wer("from 10mg to 20mg") == "from 10 mg to 20 mg"
