"""
Tool call evaluation tests.
Run with: deepeval test run test_tools.py

Tests that the agent calls the right tools with correct arguments in the right scenarios.
Uses live LLM calls with tool definitions to verify actual tool invocation behavior.
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

import llm_client

RESULTS_FILE = "results.json"
TOOL_GOLDENS_FILE = "tool_goldens.json"
_test_results = []
_start_time = time.time()

config = llm_client.load_config()

with open(TOOL_GOLDENS_FILE) as f:
    tool_goldens = json.load(f)


def _save_result(input_text, metric_name, score, passed, reason, tools_called=None, expected_tools=None):
    _test_results.append({
        "input": input_text,
        "metric": metric_name,
        "score": score,
        "passed": passed,
        "reason": reason,
        "tools_called": tools_called or [],
        "expected_tools": expected_tools or [],
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
        "model": config["model"],
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
def test_tool_correctness_live(tg):
    """Test tool invocation by actually calling the LLM with tools enabled."""
    result = llm_client.call(tg["input"], use_tools=True)

    # Parse the response — if tools were called, result is JSON
    actual_tools = []
    actual_output = result
    try:
        parsed = json.loads(result)
        if "tools_called" in parsed:
            actual_tools = parsed["tools_called"]
            actual_output = parsed.get("text", "")
    except (json.JSONDecodeError, TypeError):
        pass

    tools_called = [ToolCall(name=t["name"], input_parameters=t.get("arguments", t.get("input", {}))) for t in actual_tools]
    expected_tools = [ToolCall(name=t["name"], input_parameters=t.get("input", {})) for t in tg["expected_tools"]]

    test_case = LLMTestCase(
        input=tg["input"],
        actual_output=actual_output or tg.get("actual_output", ""),
        tools_called=tools_called,
        expected_tools=expected_tools,
    )

    metric = ToolCorrectnessMetric(threshold=0.7)
    metric.measure(test_case)

    _save_result(
        tg["input"], "ToolCorrectness", metric.score, metric.score >= 0.7, metric.reason,
        tools_called=[{"name": t.name, "input": t.input_parameters} for t in tools_called],
        expected_tools=[{"name": t.name, "input": t.input_parameters} for t in expected_tools],
    )

    assert_test(test_case, [metric])


@pytest.mark.parametrize("tg", tool_goldens, ids=[g["input"][:40] for g in tool_goldens])
def test_tool_correctness_static(tg):
    """Static assertion test using pre-defined expected vs actual tools from goldens."""
    tools_called = [ToolCall(name=t["name"], input_parameters=t.get("input", {})) for t in tg["tools_called"]]
    expected_tools = [ToolCall(name=t["name"], input_parameters=t.get("input", {})) for t in tg["expected_tools"]]

    test_case = LLMTestCase(
        input=tg["input"],
        actual_output=tg["actual_output"],
        tools_called=tools_called,
        expected_tools=expected_tools,
    )

    metric = ToolCorrectnessMetric(threshold=0.7)
    metric.measure(test_case)

    _save_result(
        tg["input"], "ToolCorrectness (Static)", metric.score, metric.score >= 0.7, metric.reason,
        tools_called=[{"name": t["name"], "input": t.get("input", {})} for t in tg["tools_called"]],
        expected_tools=[{"name": t["name"], "input": t.get("input", {})} for t in tg["expected_tools"]],
    )

    assert_test(test_case, [metric])
