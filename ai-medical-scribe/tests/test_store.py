"""Approval gate + audit log at the storage layer (incl. SQLite triggers)."""
import sqlite3

import pytest

from app.schema import SOAPNote
from app.store import NoteDocument, NoteLocked, NoteNotFound, NoteStatus, NoteStore, NotApproved


@pytest.fixture
def store(tmp_path):
    s = NoteStore(tmp_path / "t.db")
    yield s
    s.close()


@pytest.fixture
def doc(segments, valid_note_dict):
    return NoteDocument(note=SOAPNote(**valid_note_dict), segments=segments, source={"encounter_id": "TEST"})


def _edited(note: SOAPNote) -> SOAPNote:
    d = note.model_dump()
    d["plan"][1]["text"] = "Ibuprofen 400 mg three times daily"
    return SOAPNote(**d)


def test_create_is_draft_and_audited(store, doc):
    rec = store.create(doc)
    assert rec.status is NoteStatus.draft and rec.doc == doc
    [a] = store.audit(rec.id)
    assert a.action == "created" and a.before is None and a.after["note"]["plan"][1]["text"] == "Ibuprofen 400 mg"


def test_edit_sets_edited_and_records_before_after(store, doc):
    rec = store.create(doc)
    new = store.update(rec.id, _edited(doc.note), None, actor="dr_test")
    assert new.status is NoteStatus.edited
    a = store.audit(rec.id)[-1]
    assert a.action == "edited" and a.after["by"] == "dr_test"
    assert a.before["note"]["plan"][1]["text"] == "Ibuprofen 400 mg"
    assert a.after["note"]["plan"][1]["text"] == "Ibuprofen 400 mg three times daily"


def test_unchanged_save_is_a_no_op(store, doc):
    rec = store.create(doc)
    assert store.update(rec.id, doc.note, None).status is NoteStatus.draft
    assert [a.action for a in store.audit(rec.id)] == ["created"]


def test_approval_gate(store, doc):
    rec = store.create(doc)
    with pytest.raises(NotApproved):
        store.final(rec.id)  # drafts never reach the final view
    store.update(rec.id, _edited(doc.note), None)
    with pytest.raises(NotApproved):
        store.final(rec.id)  # nor do edited notes
    ok = store.approve(rec.id, actor="dr_test")
    assert ok.status is NoteStatus.approved
    assert store.final(rec.id).doc.note == _edited(doc.note)
    with pytest.raises(NoteLocked):
        store.update(rec.id, doc.note, None)
    with pytest.raises(NoteLocked):
        store.approve(rec.id)
    assert [a.action for a in store.audit(rec.id)] == ["created", "edited", "approved"]
    assert store.audit(rec.id)[-1].after["by"] == "dr_test"


def test_database_triggers_protect_approved_notes_and_audit(store, doc):
    rec = store.create(doc)
    store.approve(rec.id)
    raw = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        raw.execute("UPDATE notes SET json = '{}' WHERE id = ?", (rec.id,))
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        raw.execute("UPDATE notes SET status = 'draft' WHERE id = ?", (rec.id,))
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        raw.execute("DELETE FROM notes WHERE id = ?", (rec.id,))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        raw.execute("DELETE FROM audit_log")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        raw.execute("UPDATE audit_log SET action = 'x'")
    raw.close()


def test_status_check_constraint(store, doc):
    rec = store.create(doc)
    raw = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("UPDATE notes SET status = 'final' WHERE id = ?", (rec.id,))
    raw.close()


def test_persistence_and_listing(tmp_path, doc):
    s1 = NoteStore(tmp_path / "p.db")
    a, b = s1.create(doc), s1.create(doc)
    s1.approve(b.id)
    s1.close()
    s2 = NoteStore(tmp_path / "p.db")
    assert [r.id for r in s2.list()] == [b.id, a.id]
    assert [r.id for r in s2.list("approved")] == [b.id]
    assert [r.id for r in s2.list(NoteStatus.draft)] == [a.id]
    s2.close()


def test_missing_note(store):
    with pytest.raises(NoteNotFound):
        store.get(999)
    with pytest.raises(NoteNotFound):
        store.audit(999)
