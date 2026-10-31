"""The real FastAPI app with scripted models, for UI demos and Playwright E2E.

    python -m tests.e2e.fake_api --port 8765 --db /tmp/e2e.db

Same routes, store, redaction and verifier logic as production; only the LLM,
the NLI model and the Piper voices are replaced by deterministic fakes, so the
E2E run needs no Ollama and no model downloads.
"""
from __future__ import annotations

import argparse
import json

import uvicorn

from app.main import Services, create_app
from app.store import NoteStore
from app.verify import Judgement

DIALOGUE = "\n".join([
    "[doctor] hi , brian . what brings you in today ?",
    "[patient] my right knee has been hurting for about two weeks .",
    "[doctor] any swelling ?",
    "[patient] yes , it swells after i walk .",
    "[doctor] on exam there is mild effusion of the right knee .",
    "[doctor] i think this is a knee sprain . let's get an x-ray and start ibuprofen 400 mg .",
])

NOTE = {
    "subjective": [{"text": "Right knee pain for two weeks", "evidence": [2], "unsupported": False},
                   {"text": "Swelling after walking", "evidence": [3, 4], "unsupported": False}],
    "objective": [{"text": "Mild right knee effusion", "evidence": [5], "unsupported": False}],
    "assessment": [{"text": "Right knee sprain", "evidence": [6], "unsupported": False}],
    "plan": [{"text": "Right knee x-ray", "evidence": [6], "unsupported": False},
             {"text": "Start metformin 500 mg", "evidence": [], "unsupported": True}],
}


class ScriptedLLM:
    def chat(self, messages, *, format=None, options=None):
        return json.dumps(NOTE)


class ScriptedEntailer:
    name = "scripted"
    WEAK = {"Swelling after walking": 0.35}  # -> amber

    def judge(self, pairs):
        return [Judgement(entailment=self.WEAK.get(h, 0.95), contradiction=0.01) for _, h, _ in pairs]


class SilentVoice:
    sample_rate = 16000

    def speak(self, text, deterministic=False):
        import numpy as np

        return np.zeros(int(0.02 * self.sample_rate * max(len(text), 1) / 10), dtype=np.int16)


def build(db: str):
    return create_app(Services(store=NoteStore(db), llm=ScriptedLLM(), entailer=ScriptedEntailer(),
                               voices={"doctor": SilentVoice(), "patient": SilentVoice()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--db", default="data/e2e.db")
    a = ap.parse_args()
    uvicorn.run(build(a.db), host=a.host, port=a.port, log_level="warning")
