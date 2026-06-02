"""
Prompt quality evaluation tests.
Run with: deepeval test run test_prompts.py
or
          deepeval test run test_prompts.py -- --tb=short
or
          deepeval test run test_prompts.py -- --tb=line
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


@pytest.mark.parametrize("golden", goldens, ids=[g["input"][:40] for g in goldens])
def test_answer_relevancy(golden):
    actual_output = llm_client.call(golden["input"])
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden["expected_output"],
        context=golden.get("context"),
    )
    metric = AnswerRelevancyMetric(threshold=0.7)
    metric.measure(test_case)
    _save_result(golden["input"], "AnswerRelevancy", metric.score, metric.score >= 0.7, metric.reason)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("golden", goldens, ids=[g["input"][:40] for g in goldens])
def test_hallucination(golden):
    actual_output = llm_client.call(golden["input"])
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden["expected_output"],
        context=golden.get("context"),
    )
    metric = HallucinationMetric(threshold=0.5)
    metric.measure(test_case)
    _save_result(golden["input"], "Hallucination", metric.score, metric.score >= 0.5, metric.reason)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("golden", goldens, ids=[g["input"][:40] for g in goldens])
def test_custom_geval(golden):
    actual_output = llm_client.call(golden["input"])
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden["expected_output"],
        context=golden.get("context"),
    )
    metric = GEval(
        name="Helpfulness",
        criteria="The response is helpful, accurate, and directly addresses the user's question without unnecessary information.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.7,
    )
    metric.measure(test_case)
    _save_result(golden["input"], "Helpfulness (GEval)", metric.score, metric.score >= 0.7, metric.reason)
    assert_test(test_case, [metric])
