"""
Prompt optimization script (single-turn).
Run with: python3 optimize_prompt.py

Uses DeepEval's PromptOptimizer to improve your system prompt
based on single-turn goldens (goldens.json) and selected metric.
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
from deepeval.metrics import AnswerRelevancyMetric, GEval, HallucinationMetric
from deepeval.test_case import LLMTestCaseParams

import llm_client

sys.stdout.reconfigure(line_buffering=True)

with open("goldens.json") as f:
    raw_goldens = json.load(f)

with open("system_prompt.txt") as f:
    initial_prompt_text = f.read().strip()

with open("optimizer_config.json") as f:
    opt_config = json.load(f)

config = llm_client.load_config()

goldens = [
    Golden(
        input=g["input"],
        expected_output=g["expected_output"],
        context=g.get("context"),
    )
    for g in raw_goldens
]

if len(goldens) < 2:
    print(f"ERROR: GEPA requires at least 2 goldens, but only {len(goldens)} found.")
    print("Add more goldens from the dashboard.")
    sys.exit(1)

# Select metric based on config
metric_name = opt_config.get("metric", "AnswerRelevancy")
threshold = opt_config.get("threshold", 0.7)

if metric_name == "AnswerRelevancy":
    metric = AnswerRelevancyMetric(threshold=threshold)
elif metric_name == "Hallucination":
    metric = HallucinationMetric(threshold=threshold)
elif metric_name == "Helpfulness":
    metric = GEval(
        name="Helpfulness",
        criteria="The response is helpful, accurate, and directly addresses the user's question.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=threshold,
    )
else:
    metric = AnswerRelevancyMetric(threshold=threshold)

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
    global _call_count
    _call_count += 1
    print(f"[Call {_call_count}] {golden.input[:60]}...")
    sys.stdout.flush()
    system_text = prompt.text_template or ""
    result = llm_client.call(golden.input, system_prompt=system_text if system_text else None)
    print(f"  -> {len(result)} chars")
    sys.stdout.flush()
    return result


if __name__ == "__main__":
    start_time = time.time()

    print(f"=== Prompt Optimization (Single-Turn) ===")
    print(f"Model: {config['model']}")
    print(f"Algorithm: {algo_name}")
    print(f"Iterations: {iterations}")
    print(f"Metric: {metric_name} (threshold: {threshold})")
    print(f"Goldens: {len(goldens)}")
    print(f"Prompt: {initial_prompt_text[:60]}...")
    print(f"-" * 40)
    sys.stdout.flush()

    prompt = Prompt(
        alias="system_prompt",
        text_template=initial_prompt_text,
    )

    optimizer = PromptOptimizer(
        model_callback=model_callback,
        metrics=[metric],
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
        f.write(f"Algorithm: {algo_name}\n")
        f.write(f"Metric: {metric_name}\n")
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
        "metric": metric_name,
        "duration_seconds": duration,
        "tests": [],
    })
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nOptimized prompt saved to optimized_prompts.txt")
    print(f"\nNew prompt:\n{best_prompt_text}")
