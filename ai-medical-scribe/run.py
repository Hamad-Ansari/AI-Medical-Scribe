"""One-command launcher: starts the API and the review UI together.

    python run.py            # real models (needs Ollama running + the model pulled)
    python run.py --demo     # scripted demo models: no Ollama, no downloads

Works on Windows, macOS and Linux. Press Ctrl+C to stop both.
It checks Ollama first, waits until the API answers before opening the UI,
and if the API fails to start it prints the API's own error message.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def say(msg: str) -> None:
    print(f"[run] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"\n[run] ERROR: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(1)


def get_json(url: str, timeout: float = 3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def check_packages() -> None:
    missing = []
    for mod in ("fastapi", "uvicorn", "streamlit", "httpx", "pydantic"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        fail(f"missing Python packages: {', '.join(missing)}\n"
             f"  You are using: {sys.executable}\n"
             "  Activate the project venv first, then install:\n"
             "    Windows:      .venv\\Scripts\\activate\n"
             "    macOS/Linux:  source .venv/bin/activate\n"
             "    then:         uv pip install -r requirements.txt")


def check_ollama() -> None:
    host = os.environ.get("SCRIBE_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.environ.get("SCRIBE_MODEL", "qwen2.5:7b")
    try:
        tags = get_json(f"{host}/api/tags")
    except (urllib.error.URLError, OSError):
        fail(f"Ollama is not running at {host}.\n"
             "  Start the Ollama app (Windows/macOS) or run `ollama serve` in another terminal.\n"
             "  Or try the UI without any models:  python run.py --demo")
    names = {m.get("name", "") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        have = ", ".join(sorted(names)) or "none"
        fail(f"Ollama is running but model '{model}' is not downloaded (you have: {have}).\n"
             f"  Download it:        ollama pull {model}\n"
             "  Or use one you have: set SCRIBE_MODEL first, e.g.\n"
             '    Windows:      $env:SCRIBE_MODEL="qwen2.5:3b"\n'
             "    macOS/Linux:  export SCRIBE_MODEL=qwen2.5:3b")
    say(f"Ollama OK ({host}), model {model}")


def wait_for_api(proc: subprocess.Popen, url: str, log: Path, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            tail = log.read_text(errors="replace")[-3000:]
            fail(f"the API stopped while starting (exit code {proc.returncode}). Its output:\n"
                 f"{'-' * 70}\n{tail}\n{'-' * 70}")
        try:
            get_json(f"{url}/health")
            return
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.5)
    fail(f"the API did not answer at {url}/health within {timeout:.0f}s. Log: {log}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Start the AI Medical Scribe API + review UI")
    ap.add_argument("--demo", action="store_true", help="scripted demo models (no Ollama / downloads)")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--ui-port", type=int, default=8501)
    ap.add_argument("--ui-host", default="localhost", help="0.0.0.0 to allow other devices on your network")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    a = ap.parse_args()

    os.chdir(ROOT)  # works no matter which folder you start it from
    check_packages()
    if not a.demo:
        check_ollama()

    api_url = f"http://127.0.0.1:{a.api_port}"  # 127.0.0.1, not localhost: avoids IPv6 (::1) mix-ups on Windows
    for port, what, flag in ((a.api_port, "API", "--api-port"), (a.ui_port, "UI", "--ui-port")):
        if not port_free(port):
            fail(f"port {port} ({what}) is already in use - is it already running in another terminal?\n"
                 f"  Close that one, or pick another port:  python run.py {flag} {port + 1}")

    log = Path(tempfile.gettempdir()) / "scribe-api.log"
    if a.demo:
        db = Path(tempfile.gettempdir()) / "scribe-demo.db"
        api_cmd = [sys.executable, "-m", "tests.e2e.fake_api", "--port", str(a.api_port), "--db", str(db)]
        say("starting DEMO API (scripted models, sample note) ...")
    else:
        api_cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(a.api_port)]
        say("starting API ...")
    with open(log, "w") as fh:
        api = subprocess.Popen(api_cmd, stdout=fh, stderr=subprocess.STDOUT)
    procs = [api]
    try:
        wait_for_api(api, api_url, log)
        say(f"API ready at {api_url}  (log: {log})")
        ui_cmd = [sys.executable, "-m", "streamlit", "run", "ui/review.py",
                  "--server.port", str(a.ui_port), "--server.address", a.ui_host,
                  "--browser.gatherUsageStats", "false"]
        if a.no_browser:
            ui_cmd += ["--server.headless", "true"]
        ui = subprocess.Popen(ui_cmd, env={**os.environ, "SCRIBE_API_URL": api_url})
        procs.append(ui)
        say(f"review UI: http://localhost:{a.ui_port}   (Ctrl+C to stop)")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        if api.poll() is not None:
            say(f"the API stopped (exit code {api.returncode}). Last lines of {log}:")
            print(log.read_text(errors="replace")[-2000:])
    except KeyboardInterrupt:
        say("stopping ...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
