"""Pydantic v2 data models shared by every module.

Design notes
------------
* ``NoteItem`` / ``SOAPNote`` are the *LLM output contract*. They are strict on
  types and shape (``extra="forbid"``) but deliberately lenient on semantics:
  an item with empty evidence is still *valid* JSON - it is the verifier's job
  (phase 2) to flag it, not the parser's job to reject it. Rejecting it here
  would just make the LLM retry loop hide problems from the doctor.
* ``Segment`` is the unit of evidence. IDs are 1-based and stable for a given
  transcript; every note item cites them.
"""
from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_ITEMS_PER_SECTION = 30
MAX_EVIDENCE_PER_ITEM = 10
MAX_ITEM_CHARS = 300

SECTIONS: tuple[str, ...] = ("subjective", "objective", "assessment", "plan")
Section = Literal["subjective", "objective", "assessment", "plan"]


# --------------------------------------------------------------------------
# Transcript
# --------------------------------------------------------------------------
class Segment(BaseModel):
    """One numbered transcript segment (a speaker turn or part of one)."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1)
    start: float | None = Field(default=None, ge=0, description="seconds; None for text-only cases")
    end: float | None = Field(default=None, ge=0)
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("text", "speaker")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


def segments_to_prompt(segments: list[Segment]) -> str:
    """Render segments as ``[id] SPEAKER: text`` lines - the only text the LLM sees."""
    if not segments:
        raise ValueError("transcript has no segments")
    return "\n".join(f"[{s.id}] {s.speaker.upper()}: {s.text}" for s in segments)


# --------------------------------------------------------------------------
# SOAP note (LLM output contract)
# --------------------------------------------------------------------------
class NoteItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_ITEM_CHARS, description="One short, atomic clinical statement.")
    evidence: list[int] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_PER_ITEM,
        description="IDs of transcript segments supporting the statement.",
    )
    unsupported: bool = Field(default=False, description="True if the transcript does not clearly support the statement.")

    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("note item text must not be blank")
        return v

    @field_validator("evidence")
    @classmethod
    def _dedupe_evidence(cls, v: list[int]) -> list[int]:
        # keep order of first appearance, drop duplicates
        return list(dict.fromkeys(v))


class SOAPNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # maxItems bounds also go into the decoding grammar, which stops small
    # models from looping forever inside an array.
    subjective: list[NoteItem] = Field(max_length=MAX_ITEMS_PER_SECTION)
    objective: list[NoteItem] = Field(max_length=MAX_ITEMS_PER_SECTION)
    assessment: list[NoteItem] = Field(max_length=MAX_ITEMS_PER_SECTION)
    plan: list[NoteItem] = Field(max_length=MAX_ITEMS_PER_SECTION)

    def items(self):
        """Yield ``(section, index, item)`` for every line in the note."""
        for sec in SECTIONS:
            for i, it in enumerate(getattr(self, sec)):
                yield sec, i, it

    def section_text(self, section: str) -> str:
        return "\n".join(it.text for it in getattr(self, section))

    def n_items(self) -> int:
        return sum(len(getattr(self, s)) for s in SECTIONS)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref``/``$defs`` so the schema is self-contained.

    Ollama converts the schema to a decoding grammar; a flat schema is the most
    portable form across Ollama versions.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return resolve(copy.deepcopy(defs[name]))
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


def soap_json_schema() -> dict[str, Any]:
    """JSON schema passed to Ollama's ``format`` field."""
    return _inline_refs(SOAPNote.model_json_schema())
