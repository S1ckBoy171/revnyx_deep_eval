"""
Tool call evaluation tests.
Run with: deepeval test run test_tools.py
"""

import json
import os
import time
import atexit
from datetime import datetime, timezone
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric

RESULTS_FILE = "results.json"
TOOL_GOLDENS_FILE = "tool_goldens.json"
_test_results = []
_start_time = time.time()

with open(TOOL_GOLDENS_FILE) as f:
    tool_goldens = json.load(f)


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
    with open("system_prompt.txt") as f:
        system_prompt = f.read().strip()
    run = {
        "type": "tool_evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompt": system_prompt,
        "model": json.load(open("config.json"))["model"],
        "duration_seconds": duration,
        "tests": [],
        "tool_tests": list(_test_results),
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


@pytest.mark.parametrize("tg", tool_goldens, ids=[g["input"][:40] for g in tool_goldens])
def test_tool_correctness(tg):
    test_case = LLMTestCase(
        input=tg["input"],
        actual_output=tg["actual_output"],
        tools_called=[ToolCall(name=t["name"], input=t["input"]) for t in tg["tools_called"]],
        expected_tools=[ToolCall(name=t["name"], input=t["input"]) for t in tg["expected_tools"]],
    )
    metric = ToolCorrectnessMetric(threshold=0.7)
    metric.measure(test_case)
    _save_result(tg["input"], "ToolCorrectness", metric.score, metric.score >= 0.7, metric.reason)
    assert_test(test_case, [metric])
