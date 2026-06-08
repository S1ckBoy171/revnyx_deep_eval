"""
Prompt optimization script (single-turn).
Run with: python3 optimize_prompt.py

Uses DeepEval's PromptOptimizer with:
- Multi-metric optimization (AnswerRelevancy + LanguageCompliance + Correctness)
- Stronger optimizer model (gpt-4o) for better prompt mutations
- Baseline measurement before optimization
- 10 iterations for deeper search
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
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

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

# Multi-metric setup
threshold = opt_config.get("threshold", 0.85)

metrics = []
metric_names_config = opt_config.get("metrics", ["AnswerRelevancy"])

for metric_name in metric_names_config:
    if metric_name == "AnswerRelevancy":
        metrics.append(AnswerRelevancyMetric(threshold=threshold))
    elif metric_name == "Hallucination":
        metrics.append(HallucinationMetric(threshold=0.7))
    elif metric_name == "LanguageCompliance":
        metrics.append(GEval(
            name="LanguageCompliance",
            criteria=(
                "The response uses proper Hinglish: Hindi words in Devanagari script, English words in English script. "
                "Never romanized Hindi. Uses female verb forms for self-reference. "
                "Acronyms in Devanagari (पैन, ओटीपी, आईएफएससी, केवाईसी). Warm professional tone."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=threshold,
        ))
    elif metric_name == "Correctness":
        metrics.append(GEval(
            name="Correctness",
            criteria=(
                "The response accurately addresses the user's issue with correct information. "
                "Provides the right resolution steps and ends with the standard follow-up question. "
                "Information is factually consistent with expected output."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=threshold,
        ))
    elif metric_name == "Helpfulness":
        metrics.append(GEval(
            name="Helpfulness",
            criteria="The response is helpful, accurate, and directly addresses the user's question.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=threshold,
        ))

if not metrics:
    metrics = [AnswerRelevancyMetric(threshold=threshold)]

# Select algorithm
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

# Use stronger model for optimization judgments
optimizer_model = config.get("optimizer_model", "gpt-4o")

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


def measure_baseline():
    """Measure current prompt performance before optimization."""
    print("\n--- Baseline Measurement ---")
    scores = {m.name: [] for m in metrics}
    sample = goldens[:min(5, len(goldens))]

    for i, golden in enumerate(sample):
        output = llm_client.call(golden.input)
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=output,
            expected_output=golden.expected_output,
            context=golden.context,
        )
        for metric in metrics:
            metric.measure(test_case)
            scores[metric.name].append(metric.score)
        print(f"  [{i+1}/{len(sample)}] {golden.input[:40]}...")

    print("\nBaseline scores:")
    baseline = {}
    for name, s in scores.items():
        avg = sum(s) / len(s) if s else 0
        baseline[name] = avg
        print(f"  {name}: {avg:.3f}")
    print("---\n")
    return baseline


if __name__ == "__main__":
    start_time = time.time()

    metric_names_str = ", ".join(m.name for m in metrics)
    print(f"=== Prompt Optimization (Single-Turn) ===")
    print(f"Target Model: {config['model']}")
    print(f"Optimizer Model: {optimizer_model}")
    print(f"Algorithm: {algo_name}")
    print(f"Iterations: {iterations}")
    print(f"Metrics: {metric_names_str} (threshold: {threshold})")
    print(f"Goldens: {len(goldens)}")
    print(f"Prompt: {initial_prompt_text[:60]}...")
    print(f"-" * 40)
    sys.stdout.flush()

    baseline = measure_baseline()

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
        f.write(f"Algorithm: {algo_name}\n")
        f.write(f"Metrics: {metric_names_str}\n")
        f.write(f"Optimizer Model: {optimizer_model}\n")
        f.write(f"Target Model: {config['model']}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Duration: {duration}s\n")
        f.write(f"Baseline: {json.dumps(baseline)}\n")
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
        "optimizer_model": optimizer_model,
        "algorithm": algo_name,
        "metrics": metric_names_str,
        "iterations": iterations,
        "baseline_scores": baseline,
        "duration_seconds": duration,
        "tests": [],
    })
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nOptimized prompt saved to optimized_prompts.txt")
    print(f"\nNew prompt:\n{best_prompt_text}")
