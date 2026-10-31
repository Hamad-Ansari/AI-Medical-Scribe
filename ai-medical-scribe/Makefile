PY ?= python

.PHONY: setup data test smoke
setup:            ## create venv + install deps (needs uv or python3.11)
	uv venv -p 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
data:             ## download ACI-Bench (synthetic, CC BY 4.0)
	$(PY) scripts/download_aci_bench.py
test:             ## all tests that don't need Ollama
	$(PY) -m pytest -m "not slow and not ollama and not e2e"
smoke:            ## real generation on one case (needs `ollama serve`)
	$(PY) scripts/smoke_generate.py --split valid --index 0

.PHONY: test-slow verifier-eval smoke-verify
test-slow:        ## real models: NLI gold set, Piper voices, Piper->Whisper round trip
	$(PY) -m pytest -m slow
verifier-eval:    ## per-line verifier results on the gold set
	$(PY) eval/verifier_eval.py
smoke-verify:     ## generate + verify one case (needs `ollama serve`)
	$(PY) scripts/smoke_verify.py --split valid --index 0

.PHONY: voices smoke-audio stt-eval redaction-eval
voices:           ## download the two Piper voices into models/piper/
	$(PY) scripts/download_voices.py
smoke-audio:      ## TTS -> STT -> redaction on the first 16 turns of one case
	$(PY) scripts/smoke_audio.py --split valid --index 0 --turns 16
stt-eval:         ## WER on 5 full synthesised consultations -> eval/results/stt_wer.csv
	$(PY) eval/stt_eval.py --n 5 --max-turns 0
redaction-eval:   ## name recall on ACI-Bench + seeded fake PII + over-redaction
	$(PY) eval/redaction_eval.py

.PHONY: api ui demo e2e playwright
api:              ## FastAPI on :8000 (needs `ollama serve`; DB = $$SCRIBE_DB or data/scribe.db)
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000
ui:               ## Streamlit review UI on :8501, talking to $$SCRIBE_API_URL (default http://127.0.0.1:8000)
	$(PY) -m streamlit run ui/review.py --server.port 8501 --browser.gatherUsageStats false
demo:             ## API with scripted models on :8000 (no Ollama / NLI / Piper) -> then `make ui`
	$(PY) -m tests.e2e.fake_api --port 8000 --db /tmp/scribe-demo.db
e2e:              ## Playwright: real Streamlit UI + real FastAPI with scripted models
	$(PY) -m pytest -m e2e tests/e2e
playwright:       ## install headless Chromium for `make e2e` (no root needed)
	bash scripts/install_playwright_local.sh

.PHONY: eval eval-small
eval:             ## note quality on 20 validation cases with $$SCRIBE_MODEL (default qwen2.5:7b) -> eval/results/
	$(PY) eval/run_eval.py --split valid --n 20
eval-small:       ## the README run: qwen2.5:0.5b + 1.5b on 20 validation cases (fits ~2 GB RAM)
	SCRIBE_REPEAT_PENALTY=1.15 SCRIBE_REPEAT_LAST_N=256 $(PY) eval/run_eval.py --models qwen2.5:0.5b,qwen2.5:1.5b --split valid --n 20

.PHONY: eval-7b
eval-7b:          ## add qwen2.5:7b to the README table (cached 0.5b/1.5b notes are reused; needs ~8 GB RAM, `ollama pull qwen2.5:7b`)
	$(PY) eval/run_eval.py --models qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:7b --split valid --n 20
	$(PY) eval/update_readme.py
