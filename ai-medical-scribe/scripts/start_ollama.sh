#!/usr/bin/env bash
# Start a local Ollama server reachable from the sandbox preview proxy.
#   - binds 0.0.0.0 (127.0.0.1 is invisible to the preview)
#   - OLLAMA_ORIGINS="*" so browser requests from the preview origin aren't 403'd
# On your own machine plain `ollama serve` is fine.
set -euo pipefail
# sandbox only: restore exec bits / sonames lost in snapshot restores
[ -d /home/user/.ollama-bin ] && "$(dirname "$0")/repair_ollama_install.sh" >/dev/null
BIN="${OLLAMA_BIN:-$(command -v ollama || echo /home/user/.ollama-bin/bin/ollama)}"
export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
export OLLAMA_ORIGINS="${OLLAMA_ORIGINS:-*}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/home/user/.ollama-models}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
exec "$BIN" serve
