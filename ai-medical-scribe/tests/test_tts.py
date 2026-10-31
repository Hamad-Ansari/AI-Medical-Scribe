"""TTS tests (phase 3). Fast tests use a fake voice; the slow test uses real Piper."""
import wave

import numpy as np
import pytest

from app.schema import NoteItem, Segment, SOAPNote
from app.tts import TTSError, note_to_speech_text, read_note_aloud, speakable, synthesize_dialogue, voice_url


class FakeVoice:
    """0.01 s of audio per character, so timings are predictable."""

    def __init__(self, sample_rate=1000, level=1000):
        self.sample_rate, self.level, self.spoken = sample_rate, level, []

    def speak(self, text):
        self.spoken.append(text)
        return np.full(len(text) * 10, self.level, dtype=np.int16)


@pytest.fixture
def voices():
    return {"doctor": FakeVoice(level=1000), "patient": FakeVoice(level=-1000)}


def test_speakable_detokenises():
    assert speakable("i do n't know , i 'm gon na go .") == "I don't know, I'm gonna go."
    assert speakable("hi , [NAME] .") == "Hi, the patient."


def test_dialogue_timings_and_voices(tmp_path, voices):
    dlg = [("doctor", "hi ."), ("patient", "hello ."), ("patient_guest", "hi doc .")]
    out = synthesize_dialogue(dlg, tmp_path / "d.wav", voices=voices, pause_s=0.5)
    t = out.turns
    assert [x.speaker for x in t] == ["doctor", "patient", "patient_guest"]
    assert t[0].start == 0.0 and t[0].end == pytest.approx(0.03)          # "hi." = 3 chars
    assert t[1].start == pytest.approx(0.53)                               # + 0.5 s pause
    assert all(a.end <= b.start for a, b in zip(t, t[1:]))                 # no overlap
    assert voices["doctor"].spoken == ["Hi."]
    assert voices["patient"].spoken == ["Hello.", "Hi doc."]               # guest uses patient voice
    with wave.open(str(out.path)) as w:
        assert w.getframerate() == 1000 and w.getnchannels() == 1
        assert w.getnframes() / 1000 == pytest.approx(out.duration_s)


def test_dialogue_from_segments_keeps_ids(tmp_path, voices):
    segs = [Segment(id=7, speaker="doctor", text="hello ."), Segment(id=8, speaker="patient", text="hi .")]
    out = synthesize_dialogue(segs, tmp_path / "s.wav", voices=voices)
    assert [x.id for x in out.turns] == [7, 8] and out.turns[0].text == "hello ."


def test_empty_dialogue_rejected(tmp_path, voices):
    with pytest.raises(TTSError, match="empty"):
        synthesize_dialogue([], tmp_path / "x.wav", voices=voices)


def test_mismatched_sample_rates_rejected(tmp_path):
    with pytest.raises(TTSError, match="sample rate"):
        synthesize_dialogue([("doctor", "hi")], tmp_path / "x.wav",
                            voices={"doctor": FakeVoice(1000), "patient": FakeVoice(2000)})


def test_note_to_speech_text():
    note = SOAPNote(subjective=[NoteItem(text="Knee pain for 2 weeks", evidence=[1])], objective=[],
                    assessment=[NoteItem(text="Knee sprain.", evidence=[2])], plan=[NoteItem(text="Ibuprofen 400 mg", evidence=[2])])
    txt = note_to_speech_text(note)
    assert txt == ("Subjective. Knee pain for 2 weeks. Objective. Nothing documented. "
                   "Assessment. Knee sprain. Plan. Ibuprofen 400 mg.")


def test_read_note_aloud_writes_wav(tmp_path, voices, valid_note_dict):
    p = read_note_aloud(SOAPNote.model_validate(valid_note_dict), tmp_path / "note.wav", voices=voices)
    with wave.open(str(p)) as w:
        assert w.getnframes() > 0
    assert voices["doctor"].spoken[0] == "Subjective." and voices["patient"].spoken == []


def test_voice_url():
    onnx, js = voice_url("en_US-ryan-medium")
    assert onnx.endswith("/en/en_US/ryan/medium/en_US-ryan-medium.onnx") and js.endswith(".onnx.json")


@pytest.mark.slow
def test_real_piper_two_voices(tmp_path):
    from app.config import get_settings
    from app.tts import voice_path

    s = get_settings()
    if not voice_path(s.doctor_voice).exists():
        pytest.skip("Piper voices not downloaded")
    out = synthesize_dialogue([("doctor", "how is your knee ?"), ("patient", "it still hurts .")], tmp_path / "r.wav")
    assert out.sample_rate == 22050 and 1.0 < out.duration_s < 6.0
    assert out.turns[1].start > out.turns[0].end


# --- regression: lowercase ACI text made espeak treat a whole turn as one
# sentence, and Piper silently dropped "...metformin to 1000 mg twice a day".
@pytest.mark.parametrize("raw, expected", [
    ("so i want to increase it . and i 'm going to wait .", "So I want to increase it. And I'm going to wait."),
    ("y-you know , i-i think th-the knee hurts .", "You know, I think the knee hurts."),
    ("x-ray of the knee , well-known .", "X-ray of the knee, well-known."),  # real hyphens kept
    ("it's a little under- out of control .", "It's a little under out of control."),
    ("and um ... um ... yes", "And um, um, yes"),
    ("is it ? yes ! ok .", "Is it? Yes! Ok."),
])
def test_speakable_sentence_casing(raw, expected):
    assert speakable(raw) == expected


def test_speakable_keeps_words_inside_other_words():
    assert speakable("the iliac crest . ibuprofen .") == "The iliac crest. Ibuprofen."


def test_backchannels_made_speakable_but_units_untouched():
    # Piper's "mm-hmm" came back from Whisper as "I don't know." (meaning flipped)
    assert speakable("mm-hmm .") == "Uh-huh."
    assert speakable("mhm , yes .") == "Uh-huh, yes."
    assert speakable("a 5 mm lesion .") == "A 5 mm lesion."
    assert speakable("mm-mm , no pain .") == "Uh-uh, no pain."  # negative stays a negative
