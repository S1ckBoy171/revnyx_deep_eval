"""
Tool call evaluation tests.
Run with: deepeval test run test_tools.py
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric


def test_web_search_tool_called():
    test_case = LLMTestCase(
        input="Find me the latest news on AI",
        actual_output="Here are the latest AI news articles...",
        tools_called=[
            ToolCall(name="WebSearch", input={"query": "latest AI news 2024"})
        ],
        expected_tools=[
            ToolCall(name="WebSearch", input={"query": "latest AI news"})
        ],
    )
    metric = ToolCorrectnessMetric(threshold=0.7)
    assert_test(test_case, [metric])


def test_database_lookup_tool_called():
    test_case = LLMTestCase(
        input="What is the order status for order #12345?",
        actual_output="Order #12345 is currently being shipped.",
        tools_called=[
            ToolCall(name="DatabaseLookup", input={"order_id": "12345"})
        ],
        expected_tools=[
            ToolCall(name="DatabaseLookup", input={"order_id": "12345"})
        ],
    )
    metric = ToolCorrectnessMetric(threshold=0.7)
    assert_test(test_case, [metric])
