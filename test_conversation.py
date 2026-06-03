"""
Conversation flow evaluation tests.
Run with: deepeval test run test_conversation.py
or from the dashboard: Test Conversation button

Tests multi-turn conversations for:
- Flow correctness (agent follows intro -> qualify -> pitch -> CTA)
- Language compliance (Hinglish, female verb forms)
- Edge case handling (objections, interruptions, off-topic)
"""

import json
import os
import sys
import time
import atexit
from datetime import datetime, timezone
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

import llm_client

sys.stdout.reconfigure(line_buffering=True)

RESULTS_FILE = "results.json"
CONV_GOLDENS_FILE = "conversation_goldens.json"

config = llm_client.load_config()
system_prompt = llm_client.load_system_prompt() or ""

with open(CONV_GOLDENS_FILE) as f:
    conversations = json.load(f)

_test_results = []
_start_time = time.time()


def _save_result(scenario, turn_idx, input_text, metric_name, score, passed, reason, output_text=""):
    _test_results.append({
        "scenario": scenario,
        "turn": turn_idx,
        "input": input_text,
        "output": output_text,
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
        "type": "conversation_evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompt": system_prompt,
        "model": config["model"],
        "duration_seconds": duration,
        "tests": list(_test_results),
        "tool_tests": [],
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


def run_conversation(conv):
    """Run a full conversation, returning list of (user_msg, agent_response) pairs."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    turns = []
    user_turns = [t for t in conv["turns"] if t["role"] == "user"]

    for i, turn in enumerate(user_turns):
        messages.append({"role": "user", "content": turn["content"]})
        print(f"  [Turn {i+1}/{len(user_turns)}] User: {turn['content'][:50]}...")
        sys.stdout.flush()

        client = llm_client.get_client()
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config.get("max_tokens", 1024),
        )
        agent_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": agent_response})
        turns.append({"user": turn["content"], "agent": agent_response})
        print(f"    Agent: {agent_response[:60]}...")
        sys.stdout.flush()

    return turns


# Metrics
flow_metric = GEval(
    name="FlowCorrectness",
    criteria="The agent follows a logical call flow: greeting/intro, then qualifying the user (income, needs), then pitching the right plan, then driving a CTA. It does not skip steps, repeat itself unnecessarily, or lose track of where in the conversation it is.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

language_metric = GEval(
    name="LanguageCompliance",
    criteria="The agent speaks in Hinglish (natural Hindi-English mix). Hindi words are in Devanagari, English in English script. The agent uses female verb forms (e.g. 'मैं बता रही हूँ', 'समझ गई') and never masculine forms (e.g. 'समझ गया', 'बोल दिया'). Tone is warm and professional.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

edge_case_metric = GEval(
    name="EdgeCaseHandling",
    criteria="When the user objects, interrupts, goes off-topic, or expresses disinterest, the agent handles it gracefully — acknowledges the concern, doesn't get pushy or robotic, and either redirects naturally or closes politely. The agent never ignores what the user said.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

METRICS_MAP = {
    "flow_correctness": flow_metric,
    "language": language_metric,
    "edge_case": edge_case_metric,
}


@pytest.mark.parametrize("conv", conversations, ids=[c["scenario"][:40] for c in conversations])
def test_conversation_flow(conv):
    scenario = conv["scenario"]
    criteria = conv.get("eval_criteria", ["flow_correctness", "language", "edge_case"])
    print(f"\n{'='*50}")
    print(f"Scenario: {scenario}")
    print(f"{'='*50}")
    sys.stdout.flush()

    turns = run_conversation(conv)

    # Build full conversation context for evaluation
    full_input = "\n".join([f"User: {t['user']}" for t in turns])
    full_output = "\n".join([f"Agent: {t['agent']}" for t in turns])

    failures = []

    for criterion in criteria:
        metric = METRICS_MAP.get(criterion)
        if not metric:
            continue

        test_case = LLMTestCase(
            input=full_input,
            actual_output=full_output,
        )
        metric.measure(test_case)
        passed = metric.score >= 0.7
        _save_result(scenario, len(turns), full_input, metric.name, metric.score, passed, metric.reason, full_output)
        print(f"  [{metric.name}] Score: {metric.score:.2f} {'PASS' if passed else 'FAIL'}")
        sys.stdout.flush()
        if not passed:
            failures.append(f"{metric.name}: {metric.score:.2f}")

    # Also evaluate each individual turn for language compliance
    for i, turn in enumerate(turns):
        test_case = LLMTestCase(
            input=turn["user"],
            actual_output=turn["agent"],
        )
        language_metric.measure(test_case)
        passed = language_metric.score >= 0.7
        _save_result(scenario, i + 1, turn["user"], "LanguageCompliance (Turn " + str(i+1) + ")", language_metric.score, passed, language_metric.reason, turn["agent"])
        if not passed:
            failures.append(f"LanguageCompliance Turn {i+1}: {language_metric.score:.2f}")
            print(f"  [Turn {i+1} Language] FAIL ({language_metric.score:.2f}): {language_metric.reason[:80]}")
            sys.stdout.flush()

    if failures:
        pytest.fail(f"Failed metrics: {', '.join(failures)}")
