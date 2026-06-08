"""
Conversation flow evaluation tests.
Run with: deepeval test run test_conversation.py
or from the dashboard: Test Conversation button

Tests multi-turn conversations for:
- Flow correctness (cohort-specific: opener, qualify, resolution, wrap-up)
- Language compliance (Hinglish, Devanagari, female verb forms)
- Edge case handling (objections, interruptions, off-topic, gender gate)
- Tool invocation correctness (when tools should be called)
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
            "max_tokens": config.get("max_tokens", 2048),
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


# Cohort-specific flow metrics
def get_flow_metric(cohort):
    """Return a flow metric tailored to the cohort's expected conversation structure."""
    cohort_criteria = {
        "inactive": (
            "The agent follows the inactive cohort flow: after identity confirmation, delivers the inactive opener "
            "(mentions noticing creator hasn't been online, asks why). Then listens to response, identifies the issue "
            "category conversationally (never a numbered menu), provides the matching resolution script, and ends with "
            "'क्या इससे आपकी problem solve हो गई, या कुछ और पूछना है?' in the same turn. Wraps up warmly if resolved."
        ),
        "performance_drop": (
            "The agent follows the performance_drop cohort flow: after identity confirmation, delivers the performance_drop "
            "opener (mentions active time has decreased, asks if there's a problem). Then identifies the issue and provides resolution."
        ),
        "d1_d2": (
            "The agent follows the d1_d2 cohort flow: welcomes the new creator, asks about their experience on the app, "
            "and guides them on how to get started (going online, getting calls, earning)."
        ),
        "nudge": (
            "The agent follows the nudge cohort flow: delivers nudge opener about calls coming in and earning opportunity. "
            "If creator says yes → encourage. If later → offer reminder/schedule. If not today → accept gracefully."
        ),
        "campaign": (
            "The agent follows the campaign cohort flow: delivers campaign opener with details from cohortDetail if available. "
            "Answers questions about the campaign. If no cohortDetail, offers to have team send details and schedule callback."
        ),
        "warning": (
            "The agent follows the warning cohort flow: opens gently about earnings/account safety, delivers platform guidelines "
            "framed as protection (not accusation). If creator acknowledges → wrap up. If denies → reframe as general reminder. "
            "If resistant → use escalation angles one by one. Never threatens. Never accuses."
        ),
    }
    criteria = cohort_criteria.get(cohort, cohort_criteria["inactive"])
    return GEval(
        name="FlowCorrectness",
        criteria=criteria,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.75,
    )


language_metric = GEval(
    name="LanguageCompliance",
    criteria=(
        "The agent speaks in Hinglish (natural Hindi-English mix). Hindi words are in Devanagari, English in English script. "
        "The agent uses female verb forms (e.g. 'मैं बता रही हूँ', 'समझ गई') and never masculine forms (e.g. 'समझ गया', 'बोल दिया'). "
        "Acronyms are in Devanagari (पैन, ओटीपी, आईएफएससी, केवाईसी). "
        "Numbers in English words (not digits). Tone is warm and professional. "
        "Never uses 'sir', 'ma'am', 'bhaiya', 'didi'."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.8,
)

edge_case_metric = GEval(
    name="EdgeCaseHandling",
    criteria=(
        "When the user objects, interrupts, goes off-topic, expresses disinterest, identifies as male, or says wrong number: "
        "the agent handles it per protocol. Male caller → end call gracefully with proper script. Wrong number → apologize and end. "
        "Don't call me (angry about the call itself) → offer DNC list. Busy → ask for time, schedule callback. "
        "The agent never ignores what the user said, never gets pushy, never continues the main flow when it shouldn't."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.75,
)

gender_gate_metric = GEval(
    name="GenderGateCompliance",
    criteria=(
        "If the caller identifies as male (uses masculine verb forms like 'बोल रहा हूँ', or says husband/bhai/beta), "
        "the agent must immediately say the male-caller script and end the call. "
        "If gender metadata is 'male', the call should not proceed at all. "
        "The agent must NOT continue the regular flow with a male-identified caller."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.9,
)

METRICS_MAP = {
    "flow_correctness": None,  # dynamically created per cohort
    "language": language_metric,
    "edge_case": edge_case_metric,
    "gender_gate": gender_gate_metric,
}


@pytest.mark.parametrize("conv", conversations, ids=[c["scenario"][:40] for c in conversations])
def test_conversation_flow(conv):
    scenario = conv["scenario"]
    cohort = conv.get("cohort", "inactive")
    criteria = conv.get("eval_criteria", ["flow_correctness", "language", "edge_case"])
    print(f"\n{'='*50}")
    print(f"Scenario: {scenario} (cohort: {cohort})")
    print(f"{'='*50}")
    sys.stdout.flush()

    turns, tools_invoked = run_conversation(conv)

    full_input = "\n".join([f"User: {t['user']}" for t in turns])
    full_output = "\n".join([f"Agent: {t['agent']}" for t in turns])

    if tools_invoked:
        full_output += "\n\nTools Called: " + json.dumps(tools_invoked)

    failures = []

    for criterion in criteria:
        if criterion == "flow_correctness":
            metric = get_flow_metric(cohort)
        else:
            metric = METRICS_MAP.get(criterion)

        if not metric:
            continue

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

    # Per-turn language compliance
    for i, turn in enumerate(turns):
        if not turn["agent"]:
            continue
        test_case = LLMTestCase(
            input=turn["user"],
            actual_output=turn["agent"],
        )
        language_metric.measure(test_case)
        passed = language_metric.score >= 0.8
        _save_result(scenario, i + 1, turn["user"], f"LanguageCompliance (Turn {i+1})",
                     language_metric.score, passed, language_metric.reason, turn["agent"])
        if not passed:
            failures.append(f"LanguageCompliance Turn {i+1}: {language_metric.score:.2f}")
            print(f"  [Turn {i+1} Language] FAIL ({language_metric.score:.2f}): {language_metric.reason[:80]}")
            sys.stdout.flush()

    if failures:
        pytest.fail(f"Failed metrics: {', '.join(failures)}")
