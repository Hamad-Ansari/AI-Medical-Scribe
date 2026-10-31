"""Doctor review UI (Streamlit). Talks to the FastAPI service over HTTP only.

    SCRIBE_API_URL=http://localhost:8000 streamlit run ui/review.py

Left: redacted transcript with speaker labels. Right: SOAP note, one row per
line, coloured by verification status (green supported / amber flagged / red
unsupported, with the reason). "Evidence" on a line highlights the segments it
cites in yellow. The doctor can edit, delete and add lines, then approve; only
approved notes appear under "Final notes" and can be read aloud.
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime

import httpx
import streamlit as st

API = os.environ.get("SCRIBE_API_URL", "http://localhost:8000").rstrip("/")
SECTIONS = ("subjective", "objective", "assessment", "plan")
TITLES = {"subjective": "Subjective", "objective": "Objective", "assessment": "Assessment", "plan": "Plan"}
ICON = {"supported": '<span class="dot d-supported"></span>', "flagged": '<span class="dot d-flagged"></span>',
        "unsupported": '<span class="dot d-unsupported"></span>'}
WORD = {"supported": "supported", "flagged": "flagged", "unsupported": "unsupported"}
STATUS_LABEL = {"draft": "Draft", "edited": "Edited", "approved": "Approved"}

st.set_page_config(page_title="AI Medical Scribe · review", page_icon="🩺", layout="wide")

CSS = """
<style>
.banner{background:#fff4e5;border:1px solid #f5c26b;border-radius:6px;padding:6px 12px;font-size:.85rem;margin-bottom:.6rem}
.seg{padding:4px 8px;margin:2px 0;border-radius:4px;line-height:1.45;font-size:.92rem;border-left:3px solid transparent}
.seg-hl{background:#fde047;border-left-color:#ca8a04}
.seg-ctx{background:#fef9c3;border-left-color:#facc15}
.sid{color:#6b7280;font-family:monospace;font-size:.8rem;margin-right:4px}
.spk{font-weight:600;font-size:.75rem;text-transform:uppercase;margin-right:6px;padding:1px 5px;border-radius:3px}
.spk-doctor{background:#dbeafe;color:#1e40af}.spk-patient,.spk-patient_guest{background:#dcfce7;color:#166534}
.spk-unknown,.spk-other{background:#e5e7eb;color:#374151}
.item{padding:6px 10px;margin:3px 0;border-radius:5px;border-left:5px solid #d1d5db;background:#fafafa}
.item.st-supported{border-left-color:#16a34a}
.item.st-flagged{border-left-color:#f59e0b;background:#fffbeb}
.item.st-unsupported{border-left-color:#dc2626;background:#fef2f2}
.item.st-unsupported .itext{color:#b91c1c;font-weight:600}
.item.selected{outline:2px solid #ca8a04}
.ev{color:#6b7280;font-family:monospace;font-size:.78rem;margin-left:6px}
.reason{font-size:.8rem;margin-top:2px}
.st-unsupported .reason{color:#b91c1c}.st-flagged .reason{color:#b45309}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600;margin-left:8px;vertical-align:middle}
.b-draft{background:#e5e7eb;color:#374151}.b-edited{background:#dbeafe;color:#1e40af}.b-approved{background:#dcfce7;color:#166534}
.dot{display:inline-block;width:.7rem;height:.7rem;border-radius:50%;margin-right:5px;vertical-align:baseline}
.d-supported{background:#16a34a}.d-flagged{background:#f59e0b}.d-unsupported{background:#dc2626}
.legend{font-size:.78rem;color:#4b5563}.legend span.sw{display:inline-block;width:.9rem;height:.8rem;border-radius:2px;vertical-align:middle;margin:0 3px 0 8px}
.quote{border-left:3px solid #fde047;padding:2px 8px;margin:2px 0 2px 14px;font-size:.85rem;color:#374151}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ API client
class ApiError(Exception):
    pass


def api(method: str, path: str, timeout: float = 30, **kw):
    try:
        r = httpx.request(method, f"{API}{path}", timeout=timeout, **kw)
    except httpx.TransportError as e:
        raise ApiError(f"API not reachable at {API} ({type(e).__name__}). The API is not running: stop this and start everything with `python run.py` (or `python run.py --demo` without Ollama).")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        if isinstance(detail, list):
            detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in detail)
        raise ApiError(f"{r.status_code}: {detail}")
    return r


def fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return ts


def badge(status: str) -> str:
    return f'<span class="badge b-{status}" data-testid="note-status">{STATUS_LABEL.get(status, status)}</span>'


def counts_text(c: dict) -> str:
    """Plain text (works in widget labels and tables)."""
    return " · ".join(f"{c.get(k, 0)} {WORD[k]}" for k in ("unsupported", "flagged", "supported"))


def counts_html(c: dict) -> str:
    return " ".join(f"{ICON[k]}{c.get(k, 0)} {WORD[k]}" for k in ("unsupported", "flagged", "supported"))


# ------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault("sel", None)          # (section, index) of the line whose evidence is shown
ss.setdefault("flash", None)        # one-shot success message
qp = st.query_params
if "note" in qp and qp["note"].isdigit() and ss.get("note_id") is None:
    ss.note_id = int(qp["note"])
ss.setdefault("note_id", None)
ss.setdefault("view", qp.get("view", "Review"))


def select_note(nid: int | None):
    ss.note_id, ss.sel, ss.audio = nid, None, None
    if nid is None:
        st.query_params.clear()
    else:
        st.query_params["note"] = str(nid)


# ------------------------------------------------------------------ sidebar
st.markdown('<div class="banner">⚠️ Research demo on <b>synthetic</b> ACI-Bench consultations. '
            'Not a clinical tool. Drafts are not final until a doctor approves them.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("🩺 AI Medical Scribe")
    view = st.radio("View", ["Review", "Final notes"], key="view", horizontal=True,
                    help="Final notes shows approved notes only.")
    st.divider()
    with st.expander("➕ New draft", expanded=False):
        t_case, t_paste, t_upload = st.tabs(["ACI case", "Paste", "Upload audio"])
        with t_case:
            split = st.selectbox("Split", ["valid", "train", "test1", "test2", "test3"], key="split")
            try:
                case_ids = api("GET", f"/cases?split={split}").json()
            except ApiError as e:
                case_ids = []
                st.caption(str(e))
            enc = st.selectbox("Encounter", case_ids, key="enc")
            mode = st.radio("Input", ["text", "audio"], horizontal=True, key="mode",
                            help="audio = synthesise with Piper, transcribe with Whisper")
            max_turns = st.number_input("Max turns (0 = all)", 0, 200, 0, key="max_turns")
            if st.button("Generate draft", key="gen_case", disabled=not enc, type="primary"):
                body = {"encounter_id": enc, "split": split, "input": mode}
                if max_turns:
                    body["max_turns"] = int(max_turns)
                with st.spinner("Transcribing, redacting, generating and verifying… (minutes on CPU)"):
                    try:
                        nid = api("POST", "/process", json=body, timeout=1800).json()["id"]
                        select_note(nid)
                        ss.flash = f"Draft #{nid} created."
                        st.rerun()
                    except ApiError as e:
                        st.error(str(e))
        with t_paste:
            txt = st.text_area("Dialogue ([doctor] / [patient] lines, synthetic only)", key="paste", height=150)
            # not disabled: a text_area only commits on blur, so a disabled button would swallow the first click
            if st.button("Generate draft", key="gen_paste") and not txt.strip():
                st.error("Paste a dialogue first.")
            elif ss.get("gen_paste") and txt.strip():
                with st.spinner("Redacting, generating and verifying…"):
                    try:
                        nid = api("POST", "/process", json={"dialogue": txt}, timeout=1800).json()["id"]
                        select_note(nid)
                        ss.flash = f"Draft #{nid} created."
                        st.rerun()
                    except ApiError as e:
                        st.error(str(e))
        with t_upload:
            up = st.file_uploader("Audio (wav/mp3/m4a)", type=["wav", "mp3", "m4a", "flac", "ogg"], key="upload")
            st.caption("Speakers are labelled 'unknown' unless diarization is enabled (SCRIBE_DIARIZE=1).")
            if st.button("Transcribe + draft", key="gen_upload", disabled=up is None):
                with st.spinner("Transcribing…"):
                    try:
                        nid = api("POST", "/process", files={"audio": (up.name, up.getvalue(), up.type or "audio/wav")},
                                  timeout=1800).json()["id"]
                        select_note(nid)
                        ss.flash = f"Draft #{nid} created."
                        st.rerun()
                    except ApiError as e:
                        st.error(str(e))

    st.subheader("Notes")
    try:
        notes = api("GET", "/final" if view == "Final notes" else "/notes").json()
    except ApiError as e:
        notes = []
        st.error(str(e))
    if not notes:
        st.caption("No approved notes yet." if view == "Final notes" else "No notes yet - create a draft above.")
    for n in notes:
        label = f"#{n['id']} · {n.get('encounter_id') or n.get('input') or 'note'} · {STATUS_LABEL[n['status']]}"
        if view == "Review" and n["counts"]:
            label += f" · {n['counts'].get('unsupported', 0)} red, {n['counts'].get('flagged', 0)} amber"
        if st.button(label, key=f"pick-{n['id']}", use_container_width=True,
                     type="primary" if n["id"] == ss.note_id else "secondary"):
            select_note(n["id"])
            st.rerun()


# ------------------------------------------------------------------ helpers
def load_note(nid: int, final: bool):
    return api("GET", f"/final/{nid}" if final else f"/note/{nid}").json()


def ver_items(note_json) -> dict[tuple[str, int], dict]:
    v = note_json.get("verification") or {"items": []}
    return {(i["section"], i["index"]): i for i in v["items"]}


def render_transcript(segments, cited: set[int], context: set[int]):
    rows = []
    for s in segments:
        cls = "seg seg-hl" if s["id"] in cited else ("seg seg-ctx" if s["id"] in context else "seg")
        spk = s["speaker"] if s["speaker"] in ("doctor", "patient", "patient_guest", "unknown") else "other"
        ts = f' <span class="sid">{s["start"]:.0f}s</span>' if s.get("start") is not None else ""
        rows.append(f'<div class="{cls}" data-seg="{s["id"]}"><span class="sid">[{s["id"]}]</span>'
                    f'<span class="spk spk-{spk}">{html.escape(s["speaker"])}</span>{html.escape(s["text"])}{ts}</div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


def audit_summary(a) -> tuple[str, str]:
    by = (a.get("after") or {}).get("by", "") if isinstance(a.get("after"), dict) else ""
    if a["action"] == "edited":
        def lines(x):
            return {(s, it["text"]) for s in SECTIONS for it in (x or {}).get("note", {}).get(s, [])}
        b, af = lines(a.get("before")), lines(a.get("after"))
        return by, f"+{len(af - b)} line(s) / -{len(b - af)} line(s)"
    if a["action"] == "approved":
        c = a["after"].get("line_status_at_approval") or {}
        return by, f"approved with {counts_text(c)}" if c else "approved"
    if a["action"] == "created":
        src = a["after"].get("source", {})
        return by, f"draft from {src.get('encounter_id') or src.get('input')} ({src.get('model', '?')})"
    return by, json.dumps(a.get("after"))[:80] if a.get("after") else ""


def read_aloud(nid: int):
    if st.button("🔊 Read approved note aloud", key="read_aloud"):
        with st.spinner("Synthesising with Piper…"):
            try:
                ss.audio = (nid, api("GET", f"/note/{nid}/audio", timeout=600).content)
                st.rerun()  # refresh the audit history (read_aloud is logged)
            except ApiError as e:
                st.error(str(e))
    if ss.get("audio") and ss.audio[0] == nid:
        st.audio(ss.audio[1], format="audio/wav")


def note_markdown(n) -> str:
    out = [f"# SOAP note #{n['id']} ({n['source'].get('encounter_id') or 'note'})", ""]
    for sec in SECTIONS:
        out.append(f"## {TITLES[sec]}")
        out += [f"- {it['text']}  [{', '.join(map(str, it['evidence']))}]" for it in n["note"][sec]] or ["- (none)"]
        out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------ edit form
COLS = [0.56, 0.16, 0.16, 0.12]


def edit_form(n):
    nid = n["id"]
    version = len(api("GET", f"/note/{nid}/audit").json())  # new widget keys after each save
    with st.form(key=f"edit-{nid}-{version}"):
        st.caption("Edit text or evidence (comma-separated segment IDs), tick **delete** to remove a line, "
                   "or fill the empty row to add one. Edited and new lines are re-verified on save.")
        new = {}
        for sec in SECTIONS:
            st.markdown(f"**{TITLES[sec]}**")
            h = st.columns(COLS)
            for col, name in zip(h, ["line", "evidence IDs", "unsupported", "delete"]):
                col.markdown(f'<div class="legend">{name}</div>', unsafe_allow_html=True)
            rows = list(n["note"][sec]) + [{"text": "", "evidence": [], "unsupported": False, "_new": True}]
            out = []
            for i, it in enumerate(rows):
                c1, c2, c3, c4 = st.columns(COLS, vertical_alignment="center")
                tag = "new line" if it.get("_new") else f"line {i + 1}"
                text = c1.text_input(f"{TITLES[sec]} {tag} text", it["text"], key=f"t-{nid}-{version}-{sec}-{i}",
                                     label_visibility="collapsed", placeholder="add a line…" if it.get("_new") else None)
                ev = c2.text_input(f"{TITLES[sec]} {tag} evidence", ", ".join(map(str, it["evidence"])),
                                   key=f"e-{nid}-{version}-{sec}-{i}", label_visibility="collapsed", placeholder="IDs")
                uns = c3.checkbox(f"{TITLES[sec]} {tag} unsupported", it["unsupported"], key=f"u-{nid}-{version}-{sec}-{i}",
                                  label_visibility="collapsed")
                dele = False if it.get("_new") else c4.checkbox(f"{TITLES[sec]} {tag} delete", False,
                                                                key=f"d-{nid}-{version}-{sec}-{i}", label_visibility="collapsed")
                out.append((text, ev, uns, dele, it.get("_new", False)))
            new[sec] = out
        actor = st.text_input("Reviewer", ss.get("actor", "doctor"), key=f"actor-edit-{nid}-{version}")
        saved = st.form_submit_button("💾 Save edits", type="primary")
    if saved:
        note, errors = {}, []
        for sec, rows in new.items():
            items = []
            for k, (text, ev, uns, dele, is_new) in enumerate(rows):
                if dele or (is_new and not text.strip()):
                    continue
                if not text.strip():
                    errors.append(f"{TITLES[sec]} line {k + 1}: text is empty (tick Delete to remove it)")
                    continue
                try:
                    ids = [int(x) for x in ev.replace(";", ",").split(",") if x.strip()]
                except ValueError:
                    errors.append(f"{TITLES[sec]} line {k + 1}: evidence must be segment numbers like '2, 6'")
                    continue
                items.append({"text": text.strip(), "evidence": ids, "unsupported": uns})
            note[sec] = items
        if errors:
            for e in errors:
                st.error(e)
            return
        ss.actor = actor
        try:
            r = api("PUT", f"/note/{nid}", json={"note": note, "actor": actor}, timeout=600).json()
            ss.flash = (f"Saved - {r['rechecked']} line(s) re-verified"
                        + (f", {r['redactions']} PII replacement(s) in your edits" if r["redactions"] else "") + ".")
            ss.sel = None
            ss.close_editor = nid  # applied before the toggle is drawn on the next run
            st.rerun()
        except ApiError as e:
            st.error(str(e))


# ------------------------------------------------------------------ main panel
if ss.flash:
    st.success(ss.flash)
    ss.flash = None

if ss.note_id is None:
    st.header("Review")
    st.info("Pick a note on the left, or create a new draft.")
    st.stop()

final_view = view == "Final notes"
try:
    n = load_note(ss.note_id, final_view)
except ApiError as e:
    st.warning(str(e) if not final_view else f"Note #{ss.note_id} is not approved, so it is not in the final view.")
    st.stop()

nid, status, src = n["id"], n["status"], n["source"]
approved = status == "approved"
items = ver_items(n)
audit = api("GET", f"/note/{nid}/audit").json()
approval = next((a for a in reversed(audit) if a["action"] == "approved"), None)

st.markdown(f"<h2>Note #{nid} · {html.escape(src.get('encounter_id') or src.get('filename') or 'pasted dialogue')}"
            f"{badge(status)}</h2>", unsafe_allow_html=True)
meta = [f"input: {src.get('input')}", f"model: {src.get('model', '?')}", f"verifier: {src.get('verifier', '?')}",
        f"redactions: {sum((src.get('redactions') or {}).values())}"]
if src.get("speaker_source"):
    meta.append(f"speakers: {src['speaker_source']}")
st.markdown(f'<div class="legend">{" · ".join(meta)} &nbsp;|&nbsp; {counts_html(n["counts"])}</div>', unsafe_allow_html=True)
if approved and approval:
    st.success(f"Approved by **{approval['after'].get('by', '?')}** at {fmt_ts(approval['ts'])}. This note is final and read-only.")

# ---- final view: clean note, no editing
if final_view:
    for sec in SECTIONS:
        st.subheader(TITLES[sec])
        for it in n["note"][sec] or [{"text": "(none)", "evidence": []}]:
            st.markdown(f"- {html.escape(it['text'])} <span class='ev'>[{', '.join(map(str, it['evidence']))}]</span>",
                        unsafe_allow_html=True)
    read_aloud(nid)
    st.download_button("⬇️ Download (Markdown)", note_markdown(n), file_name=f"soap_note_{nid}.md")
    st.stop()

# ---- review view
sel = ss.sel if ss.sel in items else None
cited = set(items[sel]["valid_ids"]) if sel else set()
context = set(items[sel]["context_ids"]) - cited if sel else set()

left, right = st.columns([5, 6], gap="large")
with left:
    st.subheader("Transcript (redacted)")
    if sel:
        st.caption(f"Highlighting evidence for {TITLES[sel[0]]} line {sel[1] + 1}: "
                   + (", ".join(f"[{i}]" for i in sorted(cited)) or "no valid segments cited"))
    st.markdown('<div class="legend">Click <b>Evidence</b> on a note line:<span class="sw" style="background:#fde047"></span>cited segment'
                '<span class="sw" style="background:#fef9c3"></span>context added for a short cited turn</div>', unsafe_allow_html=True)
    with st.container(height=680):
        render_transcript(n["segments"], cited, context)

with right:
    if ss.pop("close_editor", None) == nid:
        ss[f"editing-{nid}"] = False
    editing = st.toggle("✏️ Edit note", key=f"editing-{nid}", disabled=approved,
                        help="Approved notes are read-only." if approved else None)
    if editing and not approved:
        edit_form(n)
    else:
        st.subheader("SOAP note")
        by_id = {s["id"]: s for s in n["segments"]}
        for sec in SECTIONS:
            st.markdown(f"**{TITLES[sec]}**")
            if not n["note"][sec]:
                st.caption("Nothing documented.")
            for i, it in enumerate(n["note"][sec]):
                v = items.get((sec, i), {"status": "flagged", "reasons": ["not verified"], "valid_ids": []})
                is_sel = sel == (sec, i)
                reasons = "" if v["status"] == "supported" else "".join(
                    f'<div class="reason">{"✖" if v["status"] == "unsupported" else "!"} {html.escape(r)}</div>'
                    for r in v["reasons"])
                if v["status"] != "supported" and v.get("suggested_evidence"):
                    reasons += f'<div class="reason">hint: maybe see segments {", ".join(map(str, v["suggested_evidence"]))}</div>'
                c1, c2 = st.columns([0.79, 0.21])
                c1.markdown(
                    f'<div class="item st-{v["status"]}{" selected" if is_sel else ""}" data-status="{v["status"]}">'
                    f'{ICON[v["status"]]}<span class="itext">{html.escape(it["text"])}</span>'
                    f'<span class="ev">[{", ".join(map(str, it["evidence"])) or "no evidence"}]</span>{reasons}</div>',
                    unsafe_allow_html=True)
                if c2.button("Hide" if is_sel else "Evidence", key=f"ev-{sec}-{i}", help="highlight cited transcript lines"):
                    ss.sel = None if is_sel else (sec, i)
                    st.rerun()
                if is_sel:
                    quotes = [by_id[k] for k in v["valid_ids"] if k in by_id]
                    st.markdown("".join(f'<div class="quote"><b>[{q["id"]}] {html.escape(q["speaker"])}:</b> '
                                        f'{html.escape(q["text"])}</div>' for q in quotes)
                                or '<div class="quote">No transcript line supports this statement.</div>',
                                unsafe_allow_html=True)

    st.divider()
    if approved:
        read_aloud(nid)
    else:
        red = n["counts"].get("unsupported", 0)
        if red:
            st.warning(f"{red} line(s) are not supported by the transcript (red). Review them before approving; "
                       "the decision is yours.")
        actor = st.text_input("Reviewer name", ss.get("actor", "doctor"), key="actor-approve")
        if st.button("✅ Approve note", key="approve", type="primary", disabled=editing):
            try:
                api("POST", f"/note/{nid}/approve", json={"actor": actor})
                ss.actor = actor
                ss.flash = f"Note #{nid} approved."
                st.rerun()
            except ApiError as e:
                st.error(str(e))
        if editing:
            st.caption("Save or close the editor before approving.")

with st.expander(f"📜 Audit history ({len(audit)})", expanded=False):
    st.table([{"time": fmt_ts(a["ts"]), "action": a["action"], "by": audit_summary(a)[0],
               "details": audit_summary(a)[1]} for a in audit])
