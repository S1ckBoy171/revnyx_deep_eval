"""
LLM client wrapper for the Revnyx DeepEval project.
- Config (model, temperature, etc.) loaded from config.json
- System prompt loaded from system_prompt.txt (optional — can be empty)
- Supports tool definitions for function calling
- Supports template variable injection into system prompts
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI

load_dotenv()

CONFIG_PATH = "config.json"
SYSTEM_PROMPT_PATH = "system_prompt.txt"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_system_prompt():
    with open(SYSTEM_PROMPT_PATH) as f:
        content = f.read().strip()
    return content if content else None


def inject_template_vars(prompt: str, template_vars: dict) -> str:
    """Replace @{{varName}} placeholders with actual values."""
    if not template_vars:
        return prompt
    for key, value in template_vars.items():
        prompt = prompt.replace(f"@{{{{{key}}}}}", value)
    return prompt


def get_client():
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_tools():
    """Return tool definitions from config, or empty list if none defined."""
    config = load_config()
    return config.get("tools", [])


def call(user_input: str, system_prompt: str = None, template_vars: dict = None, use_tools: bool = False) -> str:
    """Call the LLM with the given input and return the response text."""
    config = load_config()
    client = get_client()

    messages = []
    sys_prompt = system_prompt if system_prompt is not None else load_system_prompt()
    if sys_prompt:
        sys_prompt = inject_template_vars(sys_prompt, template_vars)
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_input})

    kwargs = {
        "model": config["model"],
        "messages": messages,
        "temperature": config["temperature"],
        "max_completion_tokens": config.get("max_tokens", 2048),
    }

    if use_tools:
        tools = get_tools()
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message

    if message.tool_calls:
        tool_results = []
        for tc in message.tool_calls:
            tool_results.append({
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            })
        return json.dumps({"text": message.content or "", "tools_called": tool_results})

    return message.content


def call_conversation(messages_history: list, system_prompt: str = None, template_vars: dict = None, use_tools: bool = False) -> dict:
    """Call the LLM with full conversation history. Returns dict with text and tools_called."""
    config = load_config()
    client = get_client()

    messages = []
    sys_prompt = system_prompt if system_prompt is not None else load_system_prompt()
    if sys_prompt:
        sys_prompt = inject_template_vars(sys_prompt, template_vars)
        messages.append({"role": "system", "content": sys_prompt})
    messages.extend(messages_history)

    kwargs = {
        "model": config["model"],
        "messages": messages,
        "temperature": config["temperature"],
        "max_completion_tokens": config.get("max_tokens", 2048),
    }

    if use_tools:
        tools = get_tools()
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message

    tools_called = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tools_called.append({
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments),
            })

    return {
        "text": message.content or "",
        "tools_called": tools_called,
    }
