"""Download the ACI-Bench challenge CSVs (synthetic, CC BY 4.0) into data/aci-bench/."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import get_settings  # noqa: E402
from app.loader import ACI_BASE_URL, SPLITS  # noqa: E402


def main() -> None:
    out = get_settings().data_dir
    out.mkdir(parents=True, exist_ok=True)
    for split, fname in SPLITS.items():
        dest = out / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"✓ {split:6s} {dest} (cached)")
            continue
        r = httpx.get(ACI_BASE_URL + fname, timeout=60, follow_redirects=True)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"↓ {split:6s} {dest} ({len(r.content)//1024} KB)")


if __name__ == "__main__":
    main()
