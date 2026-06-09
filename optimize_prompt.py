"""
Prompt optimization script (single-turn).
Run with: python3 optimize_prompt.py

Uses DeepEval's PromptOptimizer with:
- Metrics loaded dynamically from eval_config.json (builtin + custom)
- Algorithm and iterations from optimizer_config.json
- Baseline measurement before optimization
- Results persistence
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
from deepeval.test_case import LLMTestCase, SingleTurnParams

import llm_client

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

with open("goldens.json") as f:
    raw_goldens = json.load(f)

with open("system_prompt.txt") as f:
    initial_prompt_text = f.read().strip()

with open("optimizer_config.json") as f:
    opt_config = json.load(f)

with open("eval_config.json") as f:
    eval_config = json.load(f)

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

# ---------------------------------------------------------------------------
# Build metrics list dynamically from eval_config.json
# ---------------------------------------------------------------------------

BUILTIN_REGISTRY = {
    "answer_relevancy": lambda cfg: AnswerRelevancyMetric(threshold=cfg.get("threshold", 0.7)),
    "hallucination": lambda cfg: HallucinationMetric(threshold=cfg.get("threshold", 0.7)),
}

metrics = []

# 1. Builtin metrics
builtin_metrics_cfg = eval_config.get("builtin_metrics", {})
for key, cfg in builtin_metrics_cfg.items():
    if cfg.get("enabled", False) and key in BUILTIN_REGISTRY:
        metrics.append(BUILTIN_REGISTRY[key](cfg))

# 2. Custom GEval metrics (apply_to "all" or "single_turn")
custom_metrics_cfg = eval_config.get("custom_metrics", [])
for cm in custom_metrics_cfg:
    apply_to = cm.get("apply_to", "all")
    if apply_to in ("all", "single_turn"):
        eval_params = [PARAM_MAP[p] for p in cm.get("eval_params", ["INPUT", "ACTUAL_OUTPUT"]) if p in PARAM_MAP]
        metrics.append(
            GEval(
                name=cm["name"],
                criteria=cm["criteria"],
                evaluation_params=eval_params,
                threshold=cm.get("threshold", 0.7),
            )
        )

# Fallback: if no metrics configured, use AnswerRelevancyMetric so optimization can still run
if not metrics:
    metrics = [AnswerRelevancyMetric(threshold=0.8)]

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

# Use optimizer model from config
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
    scores = {m.name if hasattr(m, 'name') else m.__class__.__name__: [] for m in metrics}
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
            metric_name = metric.name if hasattr(metric, 'name') else metric.__class__.__name__
            scores[metric_name].append(metric.score)
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

    metric_names_str = ", ".join(m.name if hasattr(m, 'name') else m.__class__.__name__ for m in metrics)
    print(f"=== Prompt Optimization (Single-Turn) ===")
    print(f"Target Model: {config['model']}")
    print(f"Optimizer Model: {optimizer_model}")
    print(f"Algorithm: {algo_name}")
    print(f"Iterations: {iterations}")
    print(f"Metrics: {metric_names_str}")
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
