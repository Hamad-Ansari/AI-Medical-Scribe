"""Integration: redaction happens before the LLM ever sees the transcript."""
from app.pipeline import generate_and_verify, prepare_from_segments
from app.schema import Segment

from .test_verify import FakeEntailer


def test_llm_only_sees_redacted_text(fake_llm_factory, valid_note_json):
    segs = [
        Segment(id=1, speaker="doctor", text="hi , brian . call me at 555-201-3344 or brian.white@example.com ."),
        Segment(id=2, speaker="patient", text="my mrn is 48290017 . my right knee has been hurting for two weeks ."),
        Segment(id=3, speaker="doctor", text="any swelling ?"),
        Segment(id=4, speaker="patient", text="yes , it swells after i walk ."),
        Segment(id=5, speaker="doctor", text="on exam there is mild effusion of the right knee ."),
        Segment(id=6, speaker="doctor", text="i think this is a knee sprain . let's get an x-ray and start ibuprofen 400 mg ."),
    ]
    prepared = prepare_from_segments(segs)
    assert prepared.redaction.counts_by_type == {"[NAME]": 1, "[PHONE]": 1, "[EMAIL]": 1, "[ID]": 1}
    llm = fake_llm_factory([valid_note_json])
    result = generate_and_verify(prepared, client=llm, entailer=FakeEntailer())
    prompt = " ".join(m["content"] for m in llm.calls[0]["messages"])
    for secret in ("brian", "555-201-3344", "example.com", "48290017"):
        assert secret not in prompt.lower()
    assert "[NAME]" in prompt and "[PHONE]" in prompt
    assert result.verification.items  # verified against the same redacted segments
    assert segs[0].text.startswith("hi , brian")  # caller's data untouched


def test_redacted_logs(caplog, fake_llm_factory, valid_note_json):
    import logging

    with caplog.at_level(logging.INFO, logger="app"):
        logging.getLogger("app.anything").info("debug dump: patient phone 555-201-3344")
    assert "555-201-3344" not in caplog.text
