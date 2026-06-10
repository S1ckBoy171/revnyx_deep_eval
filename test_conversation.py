"""
Conversation flow evaluation tests.
Run with: deepeval test run test_conversation.py
or from the dashboard: Test Conversation button

Tests multi-turn conversations using metrics defined in eval_config.json:
- Flow correctness (cohort-specific from config, or generic fallback)
- Conversation-level metrics (from eval_config conversation_metrics / custom_metrics)
- Per-turn metrics (if a per-turn metric like LanguageCompliance is configured)
"""

import json
import os
import sys
import time
import atexit
from datetime import datetime, timezone
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval

import llm_client

sys.stdout.reconfigure(line_buffering=True)

RESULTS_FILE = "results.json"
CONV_GOLDENS_FILE = "conversation_goldens.json"
EVAL_CONFIG_FILE = "eval_config.json"

# Load eval config at module level
with open(EVAL_CONFIG_FILE) as f:
    eval_config = json.load(f)

config = llm_client.load_config()
system_prompt = llm_client.load_system_prompt() or ""

with open(CONV_GOLDENS_FILE) as f:
    conversations = json.load(f)

_test_results = []
_start_time = time.time()

GENERIC_FLOW_CRITERIA = (
    "The agent follows a logical conversation flow, responds appropriately to user messages, "
    "and handles the conversation professionally."
)

# Map of eval_params strings to SingleTurnParams
PARAM_MAP = {
    "INPUT": SingleTurnParams.INPUT,
    "ACTUAL_OUTPUT": SingleTurnParams.ACTUAL_OUTPUT,
    "EXPECTED_OUTPUT": SingleTurnParams.EXPECTED_OUTPUT,
    "CONTEXT": SingleTurnParams.CONTEXT,
    "RETRIEVAL_CONTEXT": SingleTurnParams.RETRIEVAL_CONTEXT,
}


def _build_eval_params(param_list):
    """Convert a list of param name strings to SingleTurnParams."""
    if not param_list:
        return [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
    return [PARAM_MAP[p] for p in param_list if p in PARAM_MAP]


def _get_cohort_flow_criteria(cohort_name):
    """Look up cohort flow_criteria from eval_config. Returns criteria string or None."""
    for cohort in eval_config.get("cohorts", []):
        if cohort.get("name", "").lower() == cohort_name.lower():
            return cohort.get("flow_criteria")
    return None


def _build_flow_metric(cohort_name):
    """Build a GEval metric for flow correctness based on cohort config."""
    criteria = _get_cohort_flow_criteria(cohort_name) if cohort_name else None
    if not criteria:
        criteria = GENERIC_FLOW_CRITERIA
    return GEval(
        name="FlowCorrectness",
        criteria=criteria,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.75,
    )


def _build_conversation_metrics():
    """Build all conversation-level metrics from eval_config."""
    metrics = {}
    # From conversation_metrics
    for m in eval_config.get("conversation_metrics", []):
        name = m.get("name", "")
        metric = GEval(
            name=name,
            criteria=m.get("criteria", ""),
            evaluation_params=_build_eval_params(m.get("eval_params")),
            threshold=m.get("threshold", 0.75),
        )
        metrics[name.lower()] = metric

    # From custom_metrics where apply_to is "all" or "conversation"
    for m in eval_config.get("custom_metrics", []):
        apply_to = m.get("apply_to", "").lower()
        if apply_to in ("all", "conversation"):
            name = m.get("name", "")
            metric = GEval(
                name=name,
                criteria=m.get("criteria", ""),
                evaluation_params=_build_eval_params(m.get("eval_params")),
                threshold=m.get("threshold", 0.75),
            )
            metrics[name.lower()] = metric

    return metrics


# Build the conversation metrics map once at module level
CONVERSATION_METRICS = _build_conversation_metrics()


def _find_per_turn_metric():
    """Find a per-turn metric (e.g. LanguageCompliance) from config. Returns metric or None."""
    # Look for any metric with apply_to == "turn" or name containing "language" in conversation_metrics
    for m in eval_config.get("conversation_metrics", []):
        apply_to = m.get("apply_to", "").lower()
        if apply_to == "turn":
            return GEval(
                name=m.get("name", "PerTurnMetric"),
                criteria=m.get("criteria", ""),
                evaluation_params=_build_eval_params(m.get("eval_params")),
                threshold=m.get("threshold", 0.8),
            )

    # Also check custom_metrics with apply_to == "turn"
    for m in eval_config.get("custom_metrics", []):
        apply_to = m.get("apply_to", "").lower()
        if apply_to == "turn":
            return GEval(
                name=m.get("name", "PerTurnMetric"),
                criteria=m.get("criteria", ""),
                evaluation_params=_build_eval_params(m.get("eval_params")),
                threshold=m.get("threshold", 0.8),
            )

    # Fallback: check if there's a metric named like "LanguageCompliance" at conversation level
    # that we should also apply per-turn
    for m in eval_config.get("conversation_metrics", []):
        name = m.get("name", "").lower()
        if "language" in name or "compliance" in name:
            return GEval(
                name=m.get("name", "LanguageCompliance"),
                criteria=m.get("criteria", ""),
                evaluation_params=_build_eval_params(m.get("eval_params")),
                threshold=m.get("threshold", 0.8),
            )

    return None


PER_TURN_METRIC = _find_per_turn_metric()


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
    """Run a full conversation with template vars and tool support."""
    template_vars = conv.get("template_vars", {})
    client = llm_client.get_client()

    messages = []
    prompt = llm_client.inject_template_vars(system_prompt, template_vars)
    if prompt:
        messages.append({"role": "system", "content": prompt})

    turns = []
    tools_invoked = []
    user_turns = [t for t in conv["turns"] if t["role"] == "user"]
    tools = llm_client.get_tools()

    for i, turn in enumerate(user_turns):
        messages.append({"role": "user", "content": turn["content"]})
        print(f"  [Turn {i+1}/{len(user_turns)}] User: {turn['content'][:50]}...")
        sys.stdout.flush()

        kwargs = {
            "model": config["model"],
            "messages": messages,
            "temperature": config["temperature"],
            "max_completion_tokens": config.get("max_tokens", 2048),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        agent_text = message.content or ""
        turn_tools = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_info = {
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                }
                turn_tools.append(tool_info)
                tools_invoked.append(tool_info)

        messages.append({"role": "assistant", "content": agent_text})
        turns.append({"user": turn["content"], "agent": agent_text, "tools": turn_tools})
        tool_str = f" [Tools: {', '.join(t['name'] for t in turn_tools)}]" if turn_tools else ""
        print(f"    Agent: {agent_text[:60]}...{tool_str}")
        sys.stdout.flush()

    return turns, tools_invoked


def _resolve_metrics_for_golden(conv):
    """Resolve which metrics to run for a given golden, based on eval_criteria."""
    cohort = conv.get("cohort", "")
    criteria_list = conv.get("eval_criteria", ["flow_correctness"])
    metrics = []

    for criterion in criteria_list:
        criterion_lower = criterion.lower()
        if criterion_lower == "flow_correctness":
            metrics.append(_build_flow_metric(cohort))
        elif criterion_lower in CONVERSATION_METRICS:
            metrics.append(CONVERSATION_METRICS[criterion_lower])
        else:
            # Try to find in conversation_metrics or custom_metrics by case-insensitive name
            found = False
            for m in eval_config.get("conversation_metrics", []):
                if m.get("name", "").lower() == criterion_lower:
                    metric = GEval(
                        name=m["name"],
                        criteria=m.get("criteria", ""),
                        evaluation_params=_build_eval_params(m.get("eval_params")),
                        threshold=m.get("threshold", 0.75),
                    )
                    metrics.append(metric)
                    found = True
                    break
            if not found:
                for m in eval_config.get("custom_metrics", []):
                    if m.get("name", "").lower() == criterion_lower:
                        metric = GEval(
                            name=m["name"],
                            criteria=m.get("criteria", ""),
                            evaluation_params=_build_eval_params(m.get("eval_params")),
                            threshold=m.get("threshold", 0.75),
                        )
                        metrics.append(metric)
                        break

    # If no metrics resolved at all, add flow_correctness with generic fallback
    if not metrics:
        metrics.append(_build_flow_metric(cohort))

    return metrics


if not conversations:
    pytest.skip("No conversation goldens configured", allow_module_level=True)


@pytest.mark.parametrize("conv", conversations, ids=[c["scenario"][:40] for c in conversations])
def test_conversation_flow(conv):
    scenario = conv["scenario"]
    cohort = conv.get("cohort", "")
    print(f"\n{'='*50}")
    print(f"Scenario: {scenario} (cohort: {cohort})")
    print(f"{'='*50}")
    sys.stdout.flush()

    # Resolve metrics for this golden
    metrics = _resolve_metrics_for_golden(conv)
    if not metrics and PER_TURN_METRIC is None:
        pytest.skip(f"No metrics available for scenario: {scenario}")

    turns, tools_invoked = run_conversation(conv)

    full_input = "\n".join([f"User: {t['user']}" for t in turns])
    full_output = "\n".join([f"Agent: {t['agent']}" for t in turns])

    if tools_invoked:
        full_output += "\n\nTools Called: " + json.dumps(tools_invoked)

    failures = []

    # Evaluate conversation-level metrics
    for metric in metrics:
        test_case = LLMTestCase(
            input=full_input,
            actual_output=full_output,
        )
        metric.measure(test_case)
        passed = metric.score >= metric.threshold
        _save_result(scenario, len(turns), full_input, metric.name, metric.score, passed, metric.reason, full_output)
        print(f"  [{metric.name}] Score: {metric.score:.2f} {'PASS' if passed else 'FAIL'}")
        sys.stdout.flush()
        if not passed:
            failures.append(f"{metric.name}: {metric.score:.2f}")

    # Per-turn metric evaluation (only if configured)
    if PER_TURN_METRIC is not None:
        for i, turn in enumerate(turns):
            if not turn["agent"]:
                continue
            test_case = LLMTestCase(
                input=turn["user"],
                actual_output=turn["agent"],
            )
            PER_TURN_METRIC.measure(test_case)
            passed = PER_TURN_METRIC.score >= PER_TURN_METRIC.threshold
            _save_result(scenario, i + 1, turn["user"], f"{PER_TURN_METRIC.name} (Turn {i+1})",
                         PER_TURN_METRIC.score, passed, PER_TURN_METRIC.reason, turn["agent"])
            if not passed:
                failures.append(f"{PER_TURN_METRIC.name} Turn {i+1}: {PER_TURN_METRIC.score:.2f}")
                print(f"  [Turn {i+1} {PER_TURN_METRIC.name}] FAIL ({PER_TURN_METRIC.score:.2f}): {PER_TURN_METRIC.reason[:80]}")
                sys.stdout.flush()

    if failures:
        pytest.fail(f"Failed metrics: {', '.join(failures)}")
