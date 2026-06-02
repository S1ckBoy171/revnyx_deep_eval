"""
LLM client wrapper for the Revnyx DeepEval project.
- Config (model, temperature, etc.) loaded from config.json
- System prompt loaded from system_prompt.txt (optional — can be empty)
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

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


def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call(user_input: str, system_prompt: str = None) -> str:
    """Call the LLM with the given input and return the response text."""
    config = load_config()
    client = get_client()

    messages = []
    sys_prompt = system_prompt if system_prompt is not None else load_system_prompt()
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        temperature=config["temperature"],
        max_tokens=config.get("max_tokens", 1024),
    )
    return response.choices[0].message.content
