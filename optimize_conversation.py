"""
Conversation prompt optimization script.
Run with: python3 optimize_conversation.py

Uses DeepEval's PromptOptimizer to improve your system prompt
based on conversation scenarios (conversation_goldens.json) and all 3 metrics:
FlowCorrectness, LanguageCompliance, EdgeCaseHandling.

GEPA maintains a Pareto frontier across all metrics simultaneously.
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
from deepeval.test_case import LLMTestCaseParams

import llm_client

sys.stdout.reconfigure(line_buffering=True)

CONV_GOLDENS_FILE = "conversation_goldens.json"

with open("system_prompt.txt") as f:
    initial_prompt_text = f.read().strip()

with open("optimizer_config.json") as f:
    opt_config = json.load(f)

config = llm_client.load_config()

# Load conversation goldens and flatten into Goldens for the optimizer.
# Each conversation becomes one Golden where input = JSON-encoded turns.
# The model_callback plays out the full multi-turn conversation.
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
                input=json.dumps({"scenario": conv.get("scenario", ""), "turns": user_turns}),
                expected_output="Agent follows greeting->qualify->pitch->CTA flow in Hinglish with female verb forms, handles objections gracefully.",
            )
        )

if len(goldens) < 2:
    print(f"ERROR: GEPA requires at least 2 conversation scenarios, but only {len(goldens)} found.")
    print("Add more scenarios from the dashboard's Conversation Scenarios section.")
    sys.exit(1)

# All 3 conversation metrics (same as test_conversation.py)
threshold = opt_config.get("threshold", 0.7)

flow_metric = GEval(
    name="FlowCorrectness",
    criteria="The agent follows a logical call flow: greeting/intro, then qualifying the user (income, needs), then pitching the right plan, then driving a CTA. It does not skip steps, repeat itself unnecessarily, or lose track of where in the conversation it is.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=threshold,
)

language_metric = GEval(
    name="LanguageCompliance",
    criteria="The agent speaks in Hinglish (natural Hindi-English mix). Hindi words are in Devanagari, English in English script. The agent uses female verb forms (e.g. 'मैं बता रही हूँ', 'समझ गई') and never masculine forms (e.g. 'समझ गया', 'बोल दिया'). Tone is warm and professional.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=threshold,
)

edge_case_metric = GEval(
    name="EdgeCaseHandling",
    criteria="When the user objects, interrupts, goes off-topic, or expresses disinterest, the agent handles it gracefully — acknowledges the concern, doesn't get pushy or robotic, and either redirects naturally or closes politely. The agent never ignores what the user said.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=threshold,
)

metrics = [flow_metric, language_metric, edge_case_metric]

# Select algorithm based on config
algo_name = opt_config.get("algorithm", "GEPA")
iterations = opt_config.get("iterations", 5)

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

_call_count = 0


def model_callback(prompt: Prompt, golden: Golden) -> str:
    """Play out a full multi-turn conversation and return the complete agent transcript."""
    global _call_count
    _call_count += 1

    system_text = prompt.text_template or ""
    conv_data = json.loads(golden.input)
    scenario = conv_data.get("scenario", "")
    user_turns = conv_data["turns"]

    print(f"[Call {_call_count}] Scenario: {scenario[:40]}... ({len(user_turns)} turns)")
    sys.stdout.flush()

    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})

    agent_responses = []
    client = llm_client.get_client()
    for turn in user_turns:
        messages.append({"role": "user", "content": turn})
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config.get("max_tokens", 1024),
        )
        agent_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": agent_reply})
        agent_responses.append(agent_reply)

    result = "\n".join([f"Agent: {r}" for r in agent_responses])
    print(f"  -> {len(agent_responses)} turns, {len(result)} chars")
    sys.stdout.flush()
    return result


if __name__ == "__main__":
    start_time = time.time()

    metric_names = ", ".join(m.name for m in metrics)
    print(f"=== Conversation Prompt Optimization ===")
    print(f"Model: {config['model']}")
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
        optimizer_model=config["model"],
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
        f.write(f"Model: {config['model']}\n")
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
        "type": "optimization",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompt": initial_prompt_text,
        "optimized_prompt": best_prompt_text,
        "model": config["model"],
        "algorithm": algo_name,
        "metrics": metric_names,
        "duration_seconds": duration,
        "tests": [],
    })
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nOptimized prompt saved to optimized_prompts.txt")
    print(f"\nNew prompt:\n{best_prompt_text}")
