"""
Prompt quality evaluation tests.
Run with: deepeval test run test_prompts.py
or
          deepeval test run test_prompts.py -- --tb=short

Evaluates single-turn responses against multiple metrics:
- AnswerRelevancy
- Hallucination (stricter threshold)
- LanguageCompliance (Devanagari, female verb forms, no romanized Hindi)
- NumberFormatCompliance (English words only, no digits/symbols)
- Correctness (GEval against expected_output)
"""

import json
import pytest
from datetime import datetime, timezone
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval, HallucinationMetric

import llm_client

config = llm_client.load_config()
system_prompt = llm_client.load_system_prompt() or ""

with open("goldens.json") as f:
    goldens = json.load(f)

import os
import time
import atexit

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


# Domain-specific metrics
language_compliance_metric = GEval(
    name="LanguageCompliance",
    criteria=(
        "The response uses proper Hinglish: Hindi words in Devanagari script, English words in English script. "
        "Never romanized Hindi (e.g. 'theek hai' is WRONG, 'ठीक है' is correct). "
        "Uses female verb forms for self-reference (समझ गई not समझ गया, बोल रही हूँ not बोल रहा हूँ). "
        "Acronyms are in Devanagari (पैन not PAN, ओटीपी not OTP, आईएफएससी not IFSC, केवाईसी not KYC). "
        "Never uses 'sir', 'ma'am', 'bhaiya', 'didi'. Addresses creator as 'आप'."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.85,
)

number_format_metric = GEval(
    name="NumberFormatCompliance",
    criteria=(
        "All numbers, amounts, durations, and counts are expressed in English words only "
        "(e.g. 'four to five working days', 'twenty four hours', 'five hundred rupees'). "
        "Never uses digits (4-5, 24, 500) or symbols (₹, Rs, INR). "
        "Only exception: 'दो–तीन minute' is permitted for 2-3 minutes. "
        "Money uses 'rupees' (English) or 'रुपये' (Hinglish) — never ₹ symbol."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.85,
)

correctness_metric = GEval(
    name="Correctness",
    criteria=(
        "The response accurately addresses the user's issue with correct information matching the expected output. "
        "It provides the right resolution steps, uses appropriate empathy, and ends with the standard follow-up question. "
        "The information shared must be factually consistent with the expected output — no invented features or wrong procedures."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.8,
)

forbidden_phrases_metric = GEval(
    name="ForbiddenPhraseAbsence",
    criteria=(
        "The response must NOT contain: "
        "1) 'I can't continue this call' or any variant of refusing to continue. "
        "2) Threatening language about banning or blocking earnings as punishment. "
        "3) The creator's name after the opening (name is for identity confirmation only). "
        "4) A numbered menu of issue categories. "
        "5) Masculine self-referential verb forms (समझ गया, कर दिया मैंने, बोल दिया, सुन नहीं पाया). "
        "If none of these are present, the score should be 1.0."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.9,
)


@pytest.mark.parametrize("golden", goldens, ids=[g["input"][:40] for g in goldens])
def test_prompt_quality(golden):
    """Single test per golden that evaluates against all metrics with one LLM call."""
    actual_output = llm_client.call(golden["input"])

    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden["expected_output"],
        context=golden.get("context"),
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.8),
        HallucinationMetric(threshold=0.7),
        correctness_metric,
        language_compliance_metric,
        number_format_metric,
        forbidden_phrases_metric,
    ]

    failures = []
    for metric in metrics:
        metric.measure(test_case)
        passed = metric.score >= metric.threshold
        _save_result(golden["input"], metric.__class__.__name__ if hasattr(metric, '__class__') else metric.name,
                     metric.score, passed, metric.reason)
        if not passed:
            failures.append(f"{metric.name}: {metric.score:.2f}")

    if failures:
        pytest.fail(f"Failed metrics: {', '.join(failures)}")
