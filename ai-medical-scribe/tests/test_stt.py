"""STT tests (phase 3): speaker alignment, segmenting, validation, and a real
Piper -> Whisper round trip (slow)."""
import wave

import numpy as np
import pytest

from app.stt import (
    AudioValidationError, SpeakerTurn, Word, assign_speakers, load_audio, validate_samples, words_to_segments,
)


def _w(s, e, t):
    return Word(s, e, t)


TURNS = [SpeakerTurn(0.0, 2.0, "doctor", 1), SpeakerTurn(2.4, 5.0, "patient", 2), SpeakerTurn(5.3, 6.0, "doctor", 3)]


def test_assign_by_overlap():
    words = [_w(0.1, 0.5, " Hi"), _w(1.8, 2.5, " there"), _w(2.6, 3.0, " my"), _w(5.4, 5.8, " ok")]
    owners = assign_speakers(words, TURNS)
    # " there" overlaps doctor 0.2 s vs patient 0.1 s -> doctor
    assert [o.turn_id for o in owners] == [1, 1, 2, 3]


def test_word_in_pause_goes_to_nearest_turn():
    owners = assign_speakers([_w(2.05, 2.15, " um"), _w(5.2, 5.25, " so")], TURNS)
    assert owners[0].turn_id == 1 and owners[1].turn_id == 3


def test_no_turns_gives_none():
    assert assign_speakers([_w(0, 1, " hi")], []) == [None]


def test_words_to_segments_splits_on_speaker_change():
    words = [_w(0.0, 0.4, " Hi,"), _w(0.4, 0.9, " Brian."), _w(2.5, 2.9, " Hello"), _w(2.9, 3.2, " doc."), _w(5.4, 5.9, " Okay.")]
    segs = words_to_segments(words, assign_speakers(words, TURNS))
    assert [(s.id, s.speaker, s.text) for s in segs] == [(1, "doctor", "Hi, Brian."), (2, "patient", "Hello doc."), (3, "doctor", "Okay.")]
    assert segs[0].start == 0.0 and segs[0].end == 0.9 and segs[1].start == 2.5


def test_unknown_speaker_without_turns():
    segs = words_to_segments([_w(0, 1, " Hi.")], [None])
    assert segs[0].speaker == "unknown"


def test_long_group_split_keeps_times():
    words = [_w(i, i + 0.5, f" w{i}.") for i in range(12)]
    segs = words_to_segments(words, [SpeakerTurn(0, 20, "doctor", 1)] * 12, max_words=5)
    assert len(segs) >= 2 and segs[0].start == 0 and segs[-1].end == 11.5
    assert [s.id for s in segs] == list(range(1, len(segs) + 1))


# ---- validation: clear errors for bad input -----------------------------------
def _write_wav(path, samples, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return path


def test_missing_file(tmp_path):
    with pytest.raises(AudioValidationError, match="not found"):
        load_audio(tmp_path / "nope.wav")


def test_empty_file(tmp_path):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(AudioValidationError, match="empty"):
        load_audio(p)


def test_corrupt_file(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"this is not audio" * 100)
    with pytest.raises(AudioValidationError, match="could not decode|no samples"):
        load_audio(p)


def test_too_short(tmp_path):
    p = _write_wav(tmp_path / "short.wav", np.full(1600, 3000))  # 0.1 s
    with pytest.raises(AudioValidationError, match="too short"):
        load_audio(p)


def test_silent(tmp_path):
    p = _write_wav(tmp_path / "silent.wav", np.zeros(16000))
    with pytest.raises(AudioValidationError, match="silent"):
        load_audio(p)


def test_validate_samples_ok():
    out = validate_samples(np.sin(np.linspace(0, 1000, 16000)).astype(np.float64))
    assert out.dtype == np.float32


# ---- integration: Piper -> Whisper (slow) -------------------------------------
@pytest.mark.slow
def test_piper_whisper_round_trip(tmp_path):
    import jiwer

    from app.config import get_settings
    from app.stt import transcribe
    from app.textnorm import normalize_for_wer
    from app.tts import synthesize_dialogue, voice_path

    if not voice_path(get_settings().doctor_voice).exists():
        pytest.skip("Piper voices not downloaded")
    dlg = [("doctor", "what brings you in today ?"),
           ("patient", "my right knee has been hurting for about two weeks ."),
           ("doctor", "okay . i think this is a sprain , let's get an x-ray and start ibuprofen .")]
    audio = synthesize_dialogue(dlg, tmp_path / "rt.wav")
    res = transcribe(audio.path, reference_turns=audio.turns)
    assert [s.speaker for s in res.segments] == ["doctor", "patient", "doctor"]
    assert all(s.start is not None and s.end > s.start for s in res.segments)
    ref = normalize_for_wer(" ".join(t for _, t in dlg))
    hyp = normalize_for_wer(" ".join(s.text for s in res.segments))
    assert jiwer.wer(ref, hyp) < 0.15, (ref, hyp)
