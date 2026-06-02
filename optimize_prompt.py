"""
Prompt optimization script.
Run with: python3 optimize_prompt.py

Uses DeepEval's PromptOptimizer to improve your system prompt
based on the goldens dataset and a chosen metric.
"""

import json
from datetime import datetime, timezone

from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.algorithms.gepa.gepa import GEPA
from deepeval.prompt import Prompt
from deepeval.dataset import Golden
from deepeval.metrics import AnswerRelevancyMetric

import llm_client

with open("goldens.json") as f:
    raw_goldens = json.load(f)

with open("system_prompt.txt") as f:
    initial_prompt_text = f.read().strip()

config = llm_client.load_config()

goldens = [
    Golden(
        input=g["input"],
        expected_output=g["expected_output"],
        context=g.get("context"),
    )
    for g in raw_goldens
]

metric = AnswerRelevancyMetric(threshold=0.7)


def model_callback(prompt: Prompt, golden: Golden) -> str:
    """Call the LLM with the prompt template and golden input."""
    system_text = prompt.text_template or ""
    return llm_client.call(golden.input, system_prompt=system_text if system_text else None)


if __name__ == "__main__":
    prompt = Prompt(
        alias="system_prompt",
        text_template=initial_prompt_text,
    )

    optimizer = PromptOptimizer(
        model_callback=model_callback,
        metrics=[metric],
        optimizer_model=config["model"],
        algorithm=GEPA(),
    )

    optimized_prompt = optimizer.optimize(prompt=prompt, goldens=goldens)
    best_prompt_text = optimized_prompt.text_template

    with open("optimized_prompts.txt", "a") as f:
        f.write(f"=== Run: {datetime.now(timezone.utc).isoformat()} ===\n")
        f.write(f"Algorithm: GEPA\n")
        f.write(f"Model: {config['model']}\n")
        f.write(f"Prompt:\n{best_prompt_text}\n")
        f.write(f"{'=' * 60}\n\n")

    print(f"\nOptimization complete!")
    print(f"Optimized prompt saved to optimized_prompts.txt")
    print(f"\nNew prompt:\n{best_prompt_text}")
