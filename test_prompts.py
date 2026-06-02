"""
Prompt quality evaluation tests.
Run with: deepeval test run test_prompts.py
or
          deepeval test run test_prompts.py -- --tb=short
or            
          deepeval test run test_prxompts.py -- --tb=line
"""

import json
import pytest
from deepeval import assert_test, log_hyperparameters
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval, HallucinationMetric

import llm_client

config = llm_client.load_config()
system_prompt = llm_client.load_system_prompt() or ""

log_hyperparameters(
    model=config["model"],
    prompt_template=system_prompt,
    hyperparameters={
        "temperature": config["temperature"],
        "max_tokens": config.get("max_tokens", 1024),
    },
)

with open("goldens.json") as f:
    goldens = json.load(f)


@pytest.mark.parametrize("golden", goldens, ids=[g["input"][:40] for g in goldens])
def test_answer_relevancy(golden):
    actual_output = llm_client.call(golden["input"])
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden["expected_output"],
        context=golden.get("context"),
    )
    metric = AnswerRelevancyMetric(threshold=0.7)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("golden", goldens, ids=[g["input"][:40] for g in goldens])
def test_hallucination(golden):
    actual_output = llm_client.call(golden["input"])
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden["expected_output"],
        context=golden.get("context"),
    )
    metric = HallucinationMetric(threshold=0.5)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("golden", goldens, ids=[g["input"][:40] for g in goldens])
def test_custom_geval(golden):
    actual_output = llm_client.call(golden["input"])
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden["expected_output"],
        context=golden.get("context"),
    )
    metric = GEval(
        name="Helpfulness",
        criteria="The response is helpful, accurate, and directly addresses the user's question without unnecessary information.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.7,
    )
    assert_test(test_case, [metric])
