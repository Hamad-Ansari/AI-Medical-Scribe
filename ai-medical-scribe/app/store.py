"""SQLite persistence: notes + append-only audit log.

Tables (as specified)::

    notes(id, status draft|edited|approved, json, created_at)
    audit_log(note_id, action, before, after, ts)

``notes.json`` holds a :class:`NoteDocument` - the SOAP note *and* the redacted
transcript and verification report it was checked against, so the review UI
can show evidence for every line. ``audit_log.before/after`` hold the SOAP note
JSON (or a small status dict for approvals).

Guarantees enforced by the database itself (SQLite triggers), not just by
Python code, so a bug elsewhere cannot silently break them:

* an approved note can never be modified or deleted;
* audit-log rows can never be modified or deleted.

Callers must pass **redacted** content (``app.pipeline`` and ``app.main`` do);
the store does not see raw transcripts.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.schema import Segment, SOAPNote
from app.verify import VerificationReport

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    status      TEXT NOT NULL CHECK (status IN ('draft', 'edited', 'approved')),
    json        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    note_id  INTEGER NOT NULL REFERENCES notes(id),
    action   TEXT NOT NULL,
    before   TEXT,
    after    TEXT,
    ts       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_note ON audit_log(note_id);

CREATE TRIGGER IF NOT EXISTS approved_notes_are_final
BEFORE UPDATE ON notes WHEN OLD.status = 'approved'
BEGIN SELECT RAISE(ABORT, 'approved notes are immutable'); END;

CREATE TRIGGER IF NOT EXISTS approved_notes_not_deleted
BEFORE DELETE ON notes WHEN OLD.status = 'approved'
BEGIN SELECT RAISE(ABORT, 'approved notes are immutable'); END;

CREATE TRIGGER IF NOT EXISTS audit_no_update
BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;

CREATE TRIGGER IF NOT EXISTS audit_no_delete
BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""


class NoteStatus(str, Enum):
    draft = "draft"
    edited = "edited"
    approved = "approved"


class NoteDocument(BaseModel):
    """Everything the reviewer needs, all of it already redacted."""

    note: SOAPNote
    segments: list[Segment]
    verification: VerificationReport | None = None
    source: dict[str, Any] = Field(default_factory=dict, description="case id, input type, model, redaction counts...")


class NoteRecord(BaseModel):
    id: int
    status: NoteStatus
    created_at: str
    doc: NoteDocument


class AuditEntry(BaseModel):
    seq: int = Field(description="insertion order (SQLite rowid)")
    note_id: int
    action: str
    before: Any = None
    after: Any = None
    ts: str


class StoreError(Exception):
    pass


class NoteNotFound(StoreError):
    pass


class NoteLocked(StoreError):
    """The note is approved: it can be read, never changed."""


class NotApproved(StoreError):
    """The final view only shows approved notes."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dump(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()
    return json.dumps(obj)


class NoteStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()

    # ------------------------------------------------------------ internals
    def _audit(self, note_id: int, action: str, before: Any, after: Any) -> None:
        self._conn.execute(
            "INSERT INTO audit_log(note_id, action, before, after, ts) VALUES (?, ?, ?, ?, ?)",
            (note_id, action, _dump(before), _dump(after), _now()),
        )

    def _row(self, note_id: int) -> sqlite3.Row:
        row = self._conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if row is None:
            raise NoteNotFound(f"note {note_id} not found")
        return row

    @staticmethod
    def _record(row: sqlite3.Row) -> NoteRecord:
        return NoteRecord(id=row["id"], status=row["status"], created_at=row["created_at"],
                          doc=NoteDocument.model_validate_json(row["json"]))

    # ------------------------------------------------------------ public API
    def create(self, doc: NoteDocument, actor: str = "system") -> NoteRecord:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                cur = self._conn.execute(
                    "INSERT INTO notes(status, json, created_at) VALUES ('draft', ?, ?)",
                    (doc.model_dump_json(), _now()),
                )
                nid = cur.lastrowid
                self._audit(nid, "created", None, {"by": actor, "note": doc.note.model_dump(), "source": doc.source})
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            return self.get(nid)

    def get(self, note_id: int) -> NoteRecord:
        with self._lock:
            return self._record(self._row(note_id))

    def list(self, status: NoteStatus | str | None = None) -> list[NoteRecord]:
        with self._lock:
            if status:
                rows = self._conn.execute("SELECT * FROM notes WHERE status = ? ORDER BY id DESC", (str(NoteStatus(status).value),))
            else:
                rows = self._conn.execute("SELECT * FROM notes ORDER BY id DESC")
            return [self._record(r) for r in rows.fetchall()]

    def update(self, note_id: int, note: SOAPNote, verification: VerificationReport | None,
               actor: str = "doctor") -> NoteRecord:
        """Save the doctor's edits. No-op (no audit row) if nothing changed."""
        with self._lock:
            rec = self._record(self._row(note_id))
            if rec.status is NoteStatus.approved:
                raise NoteLocked(f"note {note_id} is approved and can no longer be edited")
            if rec.doc.note == note:
                return rec
            new_doc = rec.doc.model_copy(update={"note": note, "verification": verification})
            self._conn.execute("BEGIN")
            try:
                self._conn.execute("UPDATE notes SET status = 'edited', json = ? WHERE id = ?",
                                   (new_doc.model_dump_json(), note_id))
                self._audit(note_id, "edited", {"by": actor, "note": rec.doc.note.model_dump()},
                            {"by": actor, "note": note.model_dump()})
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            return self.get(note_id)

    def approve(self, note_id: int, actor: str = "doctor") -> NoteRecord:
        with self._lock:
            rec = self._record(self._row(note_id))
            if rec.status is NoteStatus.approved:
                raise NoteLocked(f"note {note_id} is already approved")
            counts = rec.doc.verification.counts if rec.doc.verification else {}
            self._conn.execute("BEGIN")
            try:
                self._conn.execute("UPDATE notes SET status = 'approved' WHERE id = ?", (note_id,))
                self._audit(note_id, "approved", {"status": rec.status.value},
                            {"status": "approved", "by": actor, "line_status_at_approval": counts})
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            return self.get(note_id)

    def final(self, note_id: int) -> NoteRecord:
        rec = self.get(note_id)
        if rec.status is not NoteStatus.approved:
            raise NotApproved(f"note {note_id} is {rec.status.value}; only approved notes are final")
        return rec

    def log(self, note_id: int, action: str, detail: Any = None) -> None:
        """Audit a non-modifying action (e.g. 'read_aloud')."""
        with self._lock:
            self._row(note_id)
            self._audit(note_id, action, None, detail)

    def audit(self, note_id: int) -> list[AuditEntry]:
        with self._lock:
            self._row(note_id)
            rows = self._conn.execute(
                "SELECT rowid AS seq, * FROM audit_log WHERE note_id = ? ORDER BY rowid", (note_id,)
            ).fetchall()
        return [AuditEntry(seq=r["seq"], note_id=r["note_id"], action=r["action"],
                           before=json.loads(r["before"]) if r["before"] else None,
                           after=json.loads(r["after"]) if r["after"] else None, ts=r["ts"]) for r in rows]

    def close(self) -> None:
        self._conn.close()
