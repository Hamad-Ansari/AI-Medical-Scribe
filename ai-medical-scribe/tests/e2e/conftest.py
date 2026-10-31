"""Playwright E2E fixtures: the real Streamlit UI against the real FastAPI app
(scripted LLM / NLI / voices), each on a free port with a fresh database."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
_LIBS = Path.home() / ".cache" / "pw-libs" / "lib"  # scripts/install_playwright_local.sh (no-root installs)
if _LIBS.is_dir():
    os.environ["LD_LIBRARY_PATH"] = f"{_LIBS}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str, proc: subprocess.Popen, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{proc.args} exited with {proc.returncode}")
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.3)
    raise TimeoutError(url)


@pytest.fixture(scope="session")
def api_url(tmp_path_factory):
    port = _free_port()
    db = tmp_path_factory.mktemp("e2e") / "e2e.db"
    proc = subprocess.Popen([sys.executable, "-m", "tests.e2e.fake_api", "--port", str(port), "--db", str(db)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}"
    _wait(f"{url}/health", proc)
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="session")
def ui_url(api_url):
    port = _free_port()
    env = {**os.environ, "SCRIBE_API_URL": api_url}
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "ui/review.py", "--server.port", str(port), "--server.address",
         "127.0.0.1", "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    _wait(f"{url}/_stcore/health", proc)
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.fixture
def new_note(api_url):
    """Create a draft through the API; returns its id."""
    from tests.e2e.fake_api import DIALOGUE

    def _make() -> int:
        r = httpx.post(f"{api_url}/process", json={"dialogue": DIALOGUE}, timeout=60)
        r.raise_for_status()
        return r.json()["id"]
    return _make
