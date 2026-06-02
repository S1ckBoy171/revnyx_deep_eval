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

## Dashboard (Recommended)

```bash
python3 dashboard.py
```

Open **http://localhost:8050**

### Features

- **System Prompt** — paste and edit your prompt (double-click for fullscreen editor)
- **Test Prompt** — evaluates prompt quality (AnswerRelevancy, Hallucination, Helpfulness)
- **Test Tools** — evaluates tool call correctness
- **Optimize** — runs prompt optimization with configurable algorithm and iterations
- **Optimizer Settings** — click the gear icon to configure:
  - Algorithm: GEPA, MIPROv2, COPRO, SIMBA
  - Iterations count
  - Metric: AnswerRelevancy, Hallucination, Helpfulness
  - Score threshold
- **Goldens Management** — add/edit/remove test cases, or generate them with AI
- **Tool Test Cases** — define expected tool call behavior
- **Live CLI Output** — see real-time logs as tests and optimization run
- **Run History** — expandable rows with scores, pass/fail, durations, and improvement suggestions
- **Fullscreen Editor** — expand any input field for comfortable editing

### Workflow

1. Paste your system prompt
2. Add goldens (manually or via AI generation)
3. Click **Test Prompt** to see current scores
4. Configure optimizer settings (gear icon)
5. Click **Optimize** to get an improved prompt
6. Copy the optimized prompt back into the prompt field
7. Re-test to verify improvement

## CLI Usage

### Evaluate prompt quality

```bash
deepeval test run test_prompts.py -- --tb=short
```

### Optimize your prompt

```bash
python3 optimize_prompt.py
```

Settings come from `optimizer_config.json`:

```json
{
  "algorithm": "GEPA",
  "iterations": 5,
  "metric": "AnswerRelevancy",
  "threshold": 0.7
}
```

Available algorithms:
- **GEPA** — gradient-based evaluation prompt algorithm (default, balanced)
- **MIPROv2** — multi-iteration, more thorough, slower
- **COPRO** — coordinate prompt optimization, breadth-first
- **SIMBA** — simulation-based, good for complex prompts

### Evaluate tool correctness

```bash
deepeval test run test_tools.py -- --tb=short
```

Add tool test cases in `tool_goldens.json` or via the dashboard.

## Configuration

| File | Purpose |
|------|---------|
| `config.json` | LLM settings (model, temperature, max_tokens) |
| `optimizer_config.json` | Optimizer settings (algorithm, iterations, metric, threshold) |
| `system_prompt.txt` | Your system prompt |
| `goldens.json` | Prompt test dataset |
| `tool_goldens.json` | Tool test cases |
| `.env` | API key (not committed) |

## File Structure

```
├── dashboard.py            ← Web dashboard (http://localhost:8050)
├── config.json             ← LLM settings
├── optimizer_config.json   ← Optimizer algorithm/metric settings
├── system_prompt.txt       ← Your system prompt
├── goldens.json            ← Prompt test cases
├── tool_goldens.json       ← Tool test cases
├── .env                    ← API key
├── llm_client.py           ← LLM wrapper
├── test_prompts.py         ← Prompt quality tests
├── test_tools.py           ← Tool correctness tests
├── optimize_prompt.py      ← Prompt optimizer
├── results.json            ← Run history (read by dashboard)
├── optimized_prompts.txt   ← Saved optimized prompts
└── logo.png                ← Revnyx logo
```
