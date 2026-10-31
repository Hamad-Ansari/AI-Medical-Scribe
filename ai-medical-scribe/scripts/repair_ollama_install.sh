#!/usr/bin/env bash
# Repair a sandbox Ollama install after a workspace snapshot restore.
# Snapshots drop exec bits and symlinks, which breaks the llama-server runner
# ("libggml-base.so.0 => not found"). This restores both. Idempotent.
# Not needed on a normal machine where Ollama is installed system-wide.
set -euo pipefail
ROOT="${OLLAMA_ROOT:-/home/user/.ollama-bin}"
LIB="$ROOT/lib/ollama"
chmod +x "$ROOT/bin/ollama" "$LIB/llama-server" "$LIB/llama-quantize" 2>/dev/null || true
find "$LIB" -name "*.so*" -exec chmod +x {} +
# lib<name>.so.<major>.<minor>.<patch> -> lib<name>.so.<major>  (real copies survive snapshots)
for f in "$LIB"/*.so.*.*.*; do
  [ -e "$f" ] || continue
  base="$(basename "$f")"
  soname="$(echo "$base" | sed -E 's/^(.*\.so\.[0-9]+)\..*$/\1/')"
  [ -e "$LIB/$soname" ] || cp "$f" "$LIB/$soname"
done
chmod +x /home/user/ai-medical-scribe/scripts/*.sh 2>/dev/null || true
missing="$(LD_LIBRARY_PATH="$LIB" ldd "$LIB/llama-server" | grep 'not found' || true)"
if [ -n "$missing" ]; then echo "still missing:"; echo "$missing"; exit 1; fi
echo "ollama install OK: $("$ROOT/bin/ollama" --version 2>&1 | tail -1)"
