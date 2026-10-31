"""Copy the note-quality tables from eval/results/eval_summary.md into README.md.

    python eval/update_readme.py            # after eval/run_eval.py

Replaces everything between the EVAL_TABLES_START / EVAL_TABLES_END markers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START = re.compile(r"<!-- EVAL_TABLES_START[^>]*-->\n")
END = "<!-- EVAL_TABLES_END -->"


def tables_from_summary(md: str) -> str:
    """The tables from eval_summary.md: from the first table down to the footnotes."""
    start = md.index("| Model |")
    stops = [md.find(s, start) for s in ("\nSpeech-to-text", "\nInvented = note line")]
    stop = min(s for s in stops if s != -1) if any(s != -1 for s in stops) else len(md)
    return md[start:stop].strip() + "\n"


def update(readme: Path, summary: Path) -> str:
    text = readme.read_text()
    m = START.search(text)
    if not m or END not in text:
        raise SystemExit(f"markers not found in {readme}")
    new = text[: m.end()] + tables_from_summary(summary.read_text()) + text[text.index(END):]
    readme.write_text(new)
    return new


if __name__ == "__main__":
    summary = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval" / "results" / "eval_summary.md"
    update(ROOT / "README.md", summary)
    rows = [ln.split("|")[1].strip() for ln in tables_from_summary(summary.read_text()).splitlines()
            if ln.startswith("| ") and not ln.startswith("| Model")][:10]
    print(f"README.md updated from {summary}: {', '.join(dict.fromkeys(rows))}")
