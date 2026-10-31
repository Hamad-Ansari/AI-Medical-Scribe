#!/usr/bin/env bash
# Install Playwright's headless Chromium WITHOUT root (sandboxes / CI without sudo).
# Normal machines: `playwright install --with-deps chromium` is all you need.
# Here the missing shared libraries are unpacked from Debian .debs into
# ~/.cache/pw-libs; tests/e2e/conftest.py adds that dir to LD_LIBRARY_PATH.
set -euo pipefail
PY=${PY:-.venv/bin/python}
LIBDIR=${PW_LIBDIR:-$HOME/.cache/pw-libs}
$PY -m playwright install chromium
BIN=$(ls -d "$HOME"/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell | tail -1)
if ! ldd "$BIN" | grep -q "not found"; then echo "chromium: all libraries present"; exit 0; fi
mkdir -p "$LIBDIR/debs" && cd "$LIBDIR/debs"
PKGS="libnspr4 libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 libxdamage1 libxkbcommon0 libasound2t64 libatspi2.0-0t64"
apt-get download $PKGS 2>/dev/null || { echo "apt-get download failed; run 'apt-get update' (or install the libs with sudo)"; exit 1; }
for d in *.deb; do dpkg-deb -x "$d" "$LIBDIR/root"; done
mkdir -p "$LIBDIR/lib"
find "$LIBDIR/root" -name "*.so*" -exec cp -P {} "$LIBDIR/lib/" \;
echo "still missing:"; LD_LIBRARY_PATH="$LIBDIR/lib" ldd "$BIN" | grep "not found" || echo "  nothing - chromium ready (LD_LIBRARY_PATH=$LIBDIR/lib)"
