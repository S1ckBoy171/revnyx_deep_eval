"""
Prompt quality evaluation tests.
Run with: deepeval test run test_prompts.py
or
          deepeval test run test_prompts.py -- --tb=short

Evaluates single-turn responses against metrics configured in eval_config.json.
All metrics (builtin and custom) are loaded dynamically — nothing is hardcoded.
"""

import json
import os
import time
import atexit
import pytest
from datetime import datetime, timezone
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval, HallucinationMetric

import llm_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config = llm_client.load_config()
system_prompt = llm_client.load_system_prompt() or ""

with open("eval_config.json") as f:
    eval_config = json.load(f)

with open("goldens.json") as f:
    goldens = json.load(f)

# ---------------------------------------------------------------------------
# Map eval_params strings to LLMTestCaseParams enum values
# ---------------------------------------------------------------------------

PARAM_MAP = {
    "INPUT": LLMTestCaseParams.INPUT,
    "ACTUAL_OUTPUT": LLMTestCaseParams.ACTUAL_OUTPUT,
    "EXPECTED_OUTPUT": LLMTestCaseParams.EXPECTED_OUTPUT,
    "CONTEXT": LLMTestCaseParams.CONTEXT,
}

# ---------------------------------------------------------------------------
# Build metrics list dynamically from eval_config.json
# ---------------------------------------------------------------------------

BUILTIN_REGISTRY = {
    "answer_relevancy": lambda cfg: AnswerRelevancyMetric(threshold=cfg.get("threshold", 0.7)),
    "hallucination": lambda cfg: HallucinationMetric(threshold=cfg.get("threshold", 0.7)),
}

metrics_list = []

# 1. Builtin metrics
builtin_metrics_cfg = eval_config.get("builtin_metrics", {})
for key, cfg in builtin_metrics_cfg.items():
    if cfg.get("enabled", False) and key in BUILTIN_REGISTRY:
        metrics_list.append(BUILTIN_REGISTRY[key](cfg))

# 2. Custom GEval metrics (apply_to "all" or "single_turn")
custom_metrics_cfg = eval_config.get("custom_metrics", [])
for cm in custom_metrics_cfg:
    apply_to = cm.get("apply_to", "all")
    if apply_to in ("all", "single_turn"):
        eval_params = [PARAM_MAP[p] for p in cm.get("eval_params", ["INPUT", "ACTUAL_OUTPUT"]) if p in PARAM_MAP]
        metrics_list.append(
            GEval(
                name=cm["name"],
                criteria=cm["criteria"],
                evaluation_params=eval_params,
                threshold=cm.get("threshold", 0.7),
            )
        )

# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

RESULTS_FILE = "results.json"
_test_results = []
_start_time = time.time()


def _save_result(input_text, metric_name, score, passed, reason):
    _test_results.append({
        "input": input_text,
        "metric": metric_name,
        "score": score,
        "passed": passed,
        "reason": reason,
    })


def _flush_results():
    if not _test_results:
        return
    duration = round(time.time() - _start_time, 1)
    run = {
        "type": "evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompt": system_prompt,
        "model": config["model"],
        "duration_seconds": duration,
        "tests": list(_test_results),
    }
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            data = json.load(f)
    else:
        data = []
    data.append(run)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


atexit.register(_flush_results)

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("golden", goldens, ids=[g["input"][:40] for g in goldens])
def test_prompt_quality(golden):
    """Single test per golden that evaluates against all configured metrics with one LLM call."""

    if not metrics_list:
        pytest.skip("No metrics configured in eval_config.json — skipping evaluation.")

    actual_output = llm_client.call(golden["input"])

    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden.get("expected_output"),
        context=golden.get("context"),
    )

    failures = []
    for metric in metrics_list:
        metric.measure(test_case)
        passed = metric.score >= metric.threshold
        metric_name = getattr(metric, "name", metric.__class__.__name__)
        _save_result(golden["input"], metric_name, metric.score, passed, metric.reason)
        if not passed:
            failures.append(f"{metric_name}: {metric.score:.2f}")

    if failures:
        pytest.fail(f"Failed metrics: {', '.join(failures)}")
