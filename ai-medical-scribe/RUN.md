# How to run AI Medical Scribe on your own computer

Everything runs locally and is free. Only synthetic ACI-Bench data is used. This is **not a clinical tool**.

## What you need

| | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB (small model, low-memory mode) | 16 GB (for `qwen2.5:7b`) |
| Disk | ~6 GB free | ~10 GB |
| OS | Windows 10/11, macOS, or Linux | |
| Internet | only for the first setup (downloads) | |

---

## Step 1: Install two programs (one time)

1. **Ollama** (runs the LLM locally): download it from <https://ollama.com/download> and install it.
   On Windows and macOS it starts by itself in the background.
2. **uv** (installs Python 3.11 and the packages for you). Open a terminal (on Windows, use **PowerShell**) and run:
   ```bash
   pip install uv
   ```
   No Python yet? Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` · macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Step 2: Get the project and install packages (one time)

Unzip `ai-medical-scribe.zip`, then open a terminal **inside the `ai-medical-scribe` folder**.

**Windows (PowerShell):**
```powershell
uv venv -p 3.11 .venv
.venv\Scripts\activate
uv pip install torch
uv pip install -r requirements.txt
```

**macOS / Linux:**
```bash
uv venv -p 3.11 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, avoids a 2 GB CUDA download
uv pip install -r requirements.txt
```

## Step 3: Download the data, voices and LLM (one time)

```bash
python scripts/download_aci_bench.py    # synthetic consultations (already in the zip; safe to re-run)
python scripts/download_voices.py       # 2 Piper voices into models/piper/ (~120 MB)
ollama pull qwen2.5:7b                  # the LLM (~4.7 GB)
```
On a **weaker PC** (8 GB RAM or less), pull a smaller model instead: `ollama pull qwen2.5:3b` (8 GB RAM) or `ollama pull qwen2.5:0.5b` (4 GB RAM; fast, but it often cites the wrong segments).

## Step 4: Run it (every time)

Open a terminal in the project folder, activate the venv
(`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux), then run:
```bash
python run.py
```
This **one command** checks Ollama and starts the API, waits until it is ready, and then opens the review screen at
**http://localhost:8501**. Press **Ctrl+C** to stop everything.

In the sidebar, open **New draft**, then pick an ACI case (or paste a dialogue, or upload audio) and click **Generate draft**.
Then click **Evidence** on lines, use **Edit note**, enter your name, **Approve**, and **Read approved note aloud**.

> The first draft is slow because Whisper (~150 MB) and the NLI verifier (~700 MB) download once.
> With `qwen2.5:7b` on a CPU, expect about 1–3 minutes per note.

<details><summary>Prefer two terminals? (manual way)</summary>

```bash
python -m uvicorn app.main:app --port 8000      # terminal 1: API, keep it open
python -m streamlit run ui/review.py            # terminal 2: review screen
```
</details>

### If you used a smaller model or have little RAM
Set these in the terminal **before** `python run.py`:

| | Windows (PowerShell) | macOS / Linux |
|---|---|---|
| Use another model | `$env:SCRIBE_MODEL="qwen2.5:3b"` | `export SCRIBE_MODEL=qwen2.5:3b` |
| 4 GB RAM machines | `$env:SCRIBE_LOW_MEMORY="1"; $env:SCRIBE_KEEP_ALIVE="0"; $env:SCRIBE_NLI_MODEL="cross-encoder/nli-deberta-v3-xsmall"` | `export SCRIBE_LOW_MEMORY=1 SCRIBE_KEEP_ALIVE=0 SCRIBE_NLI_MODEL=cross-encoder/nli-deberta-v3-xsmall` |

## Quick demo: no Ollama, no model downloads

Want to see the UI first? This starts a demo API with scripted models (one fixed sample note):
```bash
python run.py --demo
```
In **New draft → Paste**, click **Generate draft** with any `[doctor] … / [patient] …` text.

## Run the tests

```bash
python -m pytest                       # 230 fast tests (no Ollama needed)
python -m pytest -m slow               # real Whisper / Piper / NLI models (downloads them)
playwright install --with-deps chromium   # once, for the browser tests
python -m pytest -m e2e tests/e2e      # 6 Playwright end-to-end tests
```
macOS/Linux users with `make` can use the shortcuts instead: `make test`, `make api`, `make ui`, `make demo`, `make e2e`.

## Common problems

| Problem | Fix |
|---|---|
| Error **"Ollama is not reachable"** (HTTP 503) | Start the Ollama app, or run `ollama serve`. Check with `ollama list` |
| `model "qwen2.5:7b" not found` | `ollama pull qwen2.5:7b`, or set `SCRIBE_MODEL` to the model you pulled |
| UI says **"API not reachable"** | The API is not running. Close the UI and start with `python run.py`: it starts both and shows the API's real error if it fails |
| Port already in use | Close the other copy, or `python run.py --api-port 8001 --ui-port 8502` |
| PC freezes or the process is "Killed" | Use a smaller model plus the 4 GB settings above |
| `.venv\Scripts\activate` is blocked on Windows | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again |
| Voices missing (read-aloud error) | `python scripts/download_voices.py` |
