# Revnyx DeepEval

LLM prompt evaluation and optimization using [DeepEval](https://deepeval.com).

## Setup

```bash
pip install deepeval
```

Add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=your-key-here
```

## Configuration

| File | Purpose |
|------|---------|
| `config.json` | LLM settings (model, temperature, max_tokens) |
| `system_prompt.txt` | Your system prompt (leave empty to skip) |
| `goldens.json` | Test dataset — inputs + expected outputs |

## Usage

### Evaluate your prompt

```bash
deepeval test run test_prompts.py -- --tb=short
```

### Optimize your prompt

```bash
python3 optimize_prompt.py
```

Result saved to `optimized_prompts.txt`.

### Evaluate tool calls (optional)

```bash
deepeval test run test_tools.py -- --tb=short
```

## File Structure

```
├── config.json             ← LLM settings
├── system_prompt.txt       ← Your system prompt
├── goldens.json            ← Test dataset
├── llm_client.py           ← LLM wrapper (reads config + .env)
├── test_prompts.py         ← Prompt quality tests
├── test_tools.py           ← Tool correctness tests
├── optimize_prompt.py      ← Prompt optimizer script
└── optimized_prompts.txt   ← Output: optimized prompts (appended each run)
```
