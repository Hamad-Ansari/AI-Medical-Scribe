"""End-to-end: a doctor reviews, edits and approves a draft in the Streamlit UI."""
import re

import httpx
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def open_note(page: Page, ui_url: str, nid: int):
    page.goto(f"{ui_url}/?note={nid}")
    expect(page.get_by_text("Transcript (redacted)")).to_be_visible(timeout=30_000)


def item(page: Page, text: str):
    return page.locator(".item", has_text=text)


def test_transcript_is_redacted_and_lines_are_coloured(page: Page, ui_url, new_note):
    open_note(page, ui_url, new_note())
    segs = page.locator(".seg")
    expect(segs).to_have_count(6)
    expect(segs.first).to_contain_text("[NAME]")
    expect(page.locator("body")).not_to_contain_text("brian")
    expect(segs.nth(1).locator(".spk")).to_have_text("patient")
    red = item(page, "Start metformin 500 mg")
    expect(red).to_have_attribute("data-status", "unsupported")
    expect(red).to_contain_text("No transcript segment cited")
    expect(item(page, "Swelling after walking")).to_have_attribute("data-status", "flagged")
    expect(item(page, "Right knee pain for two weeks")).to_have_attribute("data-status", "supported")
    expect(page.get_by_test_id("note-status")).to_have_text("Draft")


def test_evidence_button_highlights_cited_segments(page: Page, ui_url, new_note):
    open_note(page, ui_url, new_note())
    page.get_by_role("button", name="Evidence").nth(1).click()  # "Swelling after walking" cites [3, 4]
    expect(page.locator(".seg-hl")).to_have_count(2)
    expect(page.locator(".seg-hl").first).to_contain_text("any swelling")
    expect(page.locator(".seg-ctx")).to_contain_text("knee has been hurting")  # context for the short turn [3]
    expect(item(page, "Swelling after walking")).to_have_class(re.compile(r"\bselected\b"))
    page.get_by_role("button", name="Hide").click()
    expect(page.locator(".seg-hl")).to_have_count(0)


def test_unsupported_line_explains_missing_transcript_support(page: Page, ui_url, new_note):
    open_note(page, ui_url, new_note())
    page.get_by_role("button", name="Evidence").nth(5).click()  # the red metformin line
    expect(page.locator(".seg-hl")).to_have_count(0)
    expect(page.locator(".quote")).to_have_text("No transcript line supports this statement.")


def test_edit_delete_add_approve_and_read_aloud(page: Page, ui_url, api_url, new_note):
    nid = new_note()
    open_note(page, ui_url, nid)
    page.get_by_text("Edit note").click()
    page.get_by_label("Plan line 2 delete").check(force=True)                    # delete the red line
    page.get_by_label("Subjective line 1 text").fill("Right knee pain for about two weeks")   # edit
    page.get_by_label("Plan new line text").fill("Start ibuprofen 400 mg")           # add
    page.get_by_label("Plan new line evidence").fill("6")
    page.get_by_role("button", name="Save edits").click()
    expect(page.get_by_text(re.compile(r"Saved - 2 line\(s\) re-verified"))).to_be_visible(timeout=20_000)
    expect(page.get_by_test_id("note-status")).to_have_text("Edited")
    expect(item(page, "Start ibuprofen 400 mg")).to_have_attribute("data-status", "supported")
    expect(page.locator(".item", has_text="metformin")).to_have_count(0)
    expect(item(page, "Right knee pain for about two weeks")).to_be_visible()

    page.get_by_label("Reviewer name").fill("dr_e2e")
    page.get_by_role("button", name="Approve note").click()
    expect(page.get_by_test_id("note-status")).to_have_text("Approved", timeout=20_000)
    expect(page.get_by_text("Approved by dr_e2e")).to_be_visible()
    expect(page.get_by_role("switch", name="Edit note")).to_be_disabled()
    expect(page.get_by_role("button", name="Approve note")).to_have_count(0)

    page.get_by_role("button", name="Read approved note aloud").click()
    expect(page.locator("audio")).to_have_count(1, timeout=20_000)

    page.get_by_text("Audit history").click()
    table = page.locator("[data-testid=stTable]")
    for action in ("created", "edited", "approved", "read_aloud"):
        expect(table).to_contain_text(action)
    expect(table).to_contain_text("dr_e2e")

    # the server agrees: approved and locked
    assert httpx.get(f"{api_url}/final/{nid}").status_code == 200
    assert httpx.put(f"{api_url}/note/{nid}", json={"note": httpx.get(f"{api_url}/note/{nid}").json()["note"]}).status_code == 409


def test_final_view_only_lists_approved_notes(page: Page, ui_url, api_url, new_note):
    draft, approved = new_note(), new_note()
    httpx.post(f"{api_url}/note/{approved}/approve", json={"actor": "dr_e2e"}).raise_for_status()
    page.goto(f"{ui_url}/?view=Final%20notes")
    side = page.locator("[data-testid=stSidebar]")
    expect(side.get_by_role("button", name=re.compile(rf"^#{approved} "))).to_be_visible(timeout=30_000)
    expect(side.get_by_role("button", name=re.compile(rf"^#{draft} "))).to_have_count(0)
    page.goto(f"{ui_url}/?view=Final%20notes&note={draft}")
    expect(page.get_by_text(f"Note #{draft} is not approved")).to_be_visible(timeout=30_000)


def test_create_draft_from_pasted_dialogue(page: Page, ui_url):
    from tests.e2e.fake_api import DIALOGUE

    page.goto(ui_url)
    expect(page.get_by_text("Pick a note on the left")).to_be_visible(timeout=30_000)
    page.locator("[data-testid=stSidebar]").get_by_text("New draft").click()
    page.get_by_role("tab", name="Paste").click()
    page.get_by_label("Dialogue ([doctor] / [patient] lines, synthetic only)").fill(DIALOGUE)
    page.locator("[data-testid=stSidebar]").get_by_role("button", name="Generate draft").last.click()
    expect(page.get_by_text(re.compile(r"Draft #\d+ created"))).to_be_visible(timeout=30_000)
    expect(page.get_by_text("pasted dialogue")).to_be_visible()
    expect(page.locator(".seg")).to_have_count(6)
