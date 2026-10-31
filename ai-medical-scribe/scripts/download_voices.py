"""Download the two Piper voices (doctor, patient) into models/piper/."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.tts import iter_voice_files  # noqa: E402

for url, dest in iter_voice_files():
    if dest.exists() and dest.stat().st_size > 0:
        print(f"✓ {dest.name} (cached)")
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    print(f"↓ {dest.name} ({dest.stat().st_size // 1024} KB)")
