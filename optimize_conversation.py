"""
Conversation prompt optimization script.
Run with: python3 optimize_conversation.py

Uses DeepEval's PromptOptimizer with:
- Metrics loaded dynamically from eval_config.json (conversation_metrics + custom)
- Template variable injection for realistic testing
- Algorithm and iterations from optimizer_config.json
- Multi-turn conversation playback with tool support
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.algorithms.gepa.gepa import GEPA
from deepeval.optimizer.algorithms.miprov2.miprov2 import MIPROV2
from deepeval.optimizer.algorithms.copro.copro import COPRO
from deepeval.optimizer.algorithms.simba.simba import SIMBA
from deepeval.prompt import Prompt
from deepeval.dataset import Golden
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

import llm_client

sys.stdout.reconfigure(line_buffering=True)

CONV_GOLDENS_FILE = "conversation_goldens.json"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

with open("system_prompt.txt") as f:
    initial_prompt_text = f.read().strip()

with open("optimizer_config.json") as f:
    opt_config = json.load(f)

with open("eval_config.json") as f:
    eval_config = json.load(f)

config = llm_client.load_config()

if not os.path.exists(CONV_GOLDENS_FILE):
    print("ERROR: No conversation_goldens.json found.")
    print("Add scenarios from the dashboard (Generate or From Transcript) and try again.")
    sys.exit(1)

with open(CONV_GOLDENS_FILE) as f:
    raw_conv = json.load(f)

goldens = []
for conv in raw_conv:
    user_turns = [t["content"] for t in conv.get("turns", []) if t.get("role") == "user"]
    if user_turns:
        goldens.append(
            Golden(
                input=json.dumps({
                    "scenario": conv.get("scenario", ""),
                    "cohort": conv.get("cohort", ""),
                    "template_vars": conv.get("template_vars", {}),
                    "turns": user_turns,
                }),
                expected_output=conv.get("expected_output", "Agent responds appropriately to the conversation."),
            )
        )

if len(goldens) < 2:
    print(f"ERROR: GEPA requires at least 2 conversation scenarios, but only {len(goldens)} found.")
    print("Add more scenarios from the dashboard's Conversation Scenarios section.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Map eval_params strings to SingleTurnParams enum values
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Build metrics list dynamically from eval_config.json
# ---------------------------------------------------------------------------

threshold = opt_config.get("conversation_threshold", opt_config.get("threshold", 0.8))

metrics = []

# 1. Conversation metrics from eval_config
for m in eval_config.get("conversation_metrics", []):
    metrics.append(
        GEval(
            name=m.get("name", "ConversationMetric"),
            criteria=m.get("criteria", ""),
            evaluation_params=_build_eval_params(m.get("eval_params")),
            threshold=m.get("threshold", threshold),
        )
    )

# 2. Custom metrics where apply_to is "all" or "conversation"
for cm in eval_config.get("custom_metrics", []):
    apply_to = cm.get("apply_to", "all").lower()
    if apply_to in ("all", "conversation"):
        metrics.append(
            GEval(
                name=cm["name"],
                criteria=cm["criteria"],
                evaluation_params=_build_eval_params(cm.get("eval_params")),
                threshold=cm.get("threshold", threshold),
            )
        )

# Fallback: if no conversation metrics configured, use a generic flow metric
if not metrics:
    metrics = [
        GEval(
            name="FlowCorrectness",
            criteria="Agent follows a logical conversation flow, responds appropriately to user messages, and handles the conversation professionally.",
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            threshold=0.75,
        )
    ]

# ---------------------------------------------------------------------------
# Algorithm selection from optimizer_config.json
# ---------------------------------------------------------------------------

algo_name = opt_config.get("algorithm", "GEPA")
iterations = opt_config.get("iterations", 10)

if algo_name == "GEPA":
    algorithm = GEPA(iterations=iterations)
elif algo_name == "MIPROv2":
    algorithm = MIPROV2(num_trials=iterations * 6)
elif algo_name == "COPRO":
    algorithm = COPRO(depth=iterations)
elif algo_name == "SIMBA":
    algorithm = SIMBA(iterations=iterations)
else:
    algorithm = GEPA(iterations=iterations)

optimizer_model = config.get("optimizer_model", "gpt-4o")

_call_count = 0


def model_callback(prompt: Prompt, golden: Golden) -> str:
    """Play out a full multi-turn conversation with template vars and return the agent transcript."""
    global _call_count
    _call_count += 1

    system_text = prompt.text_template or ""
    conv_data = json.loads(golden.input)
    scenario = conv_data.get("scenario", "")
    cohort = conv_data.get("cohort", "")
    template_vars = conv_data.get("template_vars", {})
    user_turns = conv_data["turns"]

    print(f"[Call {_call_count}] Scenario: {scenario[:30]} (cohort: {cohort}, {len(user_turns)} turns)")
    sys.stdout.flush()

    # Inject template variables from the golden into the system prompt
    injected_prompt = llm_client.inject_template_vars(system_text, template_vars)

    messages = []
    if injected_prompt:
        messages.append({"role": "system", "content": injected_prompt})

    agent_responses = []
    client = llm_client.get_client()
    tools = llm_client.get_tools()

    for turn in user_turns:
        messages.append({"role": "user", "content": turn})

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
        agent_reply = message.content or ""

        tool_info = ""
        if message.tool_calls:
            tool_names = [tc.function.name for tc in message.tool_calls]
            tool_info = f" [Tools: {', '.join(tool_names)}]"

        messages.append({"role": "assistant", "content": agent_reply})
        agent_responses.append(agent_reply + tool_info)

    result = "\n".join([f"Agent: {r}" for r in agent_responses])
    print(f"  -> {len(agent_responses)} turns, {len(result)} chars")
    sys.stdout.flush()
    return result


if __name__ == "__main__":
    start_time = time.time()

    metric_names = ", ".join(m.name for m in metrics)
    print(f"=== Conversation Prompt Optimization ===")
    print(f"Target Model: {config['model']}")
    print(f"Optimizer Model: {optimizer_model}")
    print(f"Algorithm: {algo_name}")
    print(f"Iterations: {iterations}")
    print(f"Metrics: {metric_names} (threshold: {threshold})")
    print(f"Scenarios: {len(goldens)}")
    print(f"Prompt: {initial_prompt_text[:60]}...")
    print(f"-" * 40)
    sys.stdout.flush()

    prompt = Prompt(
        alias="system_prompt",
        text_template=initial_prompt_text,
    )

    optimizer = PromptOptimizer(
        model_callback=model_callback,
        metrics=metrics,
        optimizer_model=optimizer_model,
        algorithm=algorithm,
    )

    optimized_prompt = optimizer.optimize(prompt=prompt, goldens=goldens)
    best_prompt_text = optimized_prompt.text_template
    duration = round(time.time() - start_time, 1)
    print(f"-" * 40)
    print(f"Done in {duration}s")

    with open("optimized_prompts.txt", "a") as f:
        f.write(f"=== Run: {datetime.now(timezone.utc).isoformat()} ===\n")
        f.write(f"Type: Conversation Optimization\n")
        f.write(f"Algorithm: {algo_name}\n")
        f.write(f"Metrics: {metric_names}\n")
        f.write(f"Optimizer Model: {optimizer_model}\n")
        f.write(f"Target Model: {config['model']}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Duration: {duration}s\n")
        f.write(f"Prompt:\n{best_prompt_text}\n")
        f.write(f"{'=' * 60}\n\n")

    results_file = "results.json"
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
    else:
        data = []
    data.append({
        "type": "conversation_optimization",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompt": initial_prompt_text,
        "optimized_prompt": best_prompt_text,
        "model": config["model"],
        "optimizer_model": optimizer_model,
        "algorithm": algo_name,
        "metrics": metric_names,
        "iterations": iterations,
        "duration_seconds": duration,
        "tests": [],
    })
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nOptimized prompt saved to optimized_prompts.txt")
    print(f"\nNew prompt:\n{best_prompt_text}")
