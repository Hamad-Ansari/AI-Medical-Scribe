"""Real NLI model on the hand-labelled gold set (slow: downloads ~700 MB once).

Guards against regressions in the verifier's ability to catch invented facts.
Run with:  pytest -m slow
"""
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

if importlib.util.find_spec("transformers") is None:
    pytest.skip("transformers not installed", allow_module_level=True)


def test_nli_verifier_on_gold_set():
    from verifier_eval import run

    m = run("nli")["metrics"]
    # measured with nli-deberta-v3-base: catch 89%, false alarm 0% (n=40)
    assert m["catch_rate"] >= 0.85, m
    assert m["false_alarm_rate"] <= 0.10, m
