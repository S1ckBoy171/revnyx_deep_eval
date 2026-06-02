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
- **Test Conversation** — multi-turn conversation flow testing for voice/chat agents
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
- **View Results** — full-screen modal with structured pass/fail tables and improvement suggestions
- **View Diff** — side-by-side comparison of original vs optimized prompt
- **Fullscreen Editor** — expand any input field for comfortable editing

### Workflow

1. Paste your system prompt
2. Add goldens (manually or via AI generation)
3. Click **Test Prompt** to see current scores
4. Click **Test Conversation** to test multi-turn flows
5. Configure optimizer settings (gear icon)
6. Click **Optimize** to get an improved prompt
7. View the diff to see what changed
8. Copy the optimized prompt back into the prompt field
9. Re-test to verify improvement

## Conversation Testing

For voice call agents or chat agents where **flow matters**, use conversation tests.

### How it works

1. Define conversation scenarios in `conversation_goldens.json`
2. Each scenario has a list of user turns (the agent responses are generated live)
3. The agent is evaluated on:
   - **FlowCorrectness** — follows the right sequence, doesn't skip or repeat steps
   - **LanguageCompliance** — maintains correct language, tone, verb forms
   - **EdgeCaseHandling** — handles objections, interruptions, off-topic gracefully

### Defining scenarios

Edit `conversation_goldens.json`:

```json
[
  {
    "scenario": "Happy path - full call flow",
    "eval_criteria": ["flow_correctness", "language", "edge_case"],
    "turns": [
      {"role": "user", "content": "Hello?"},
      {"role": "user", "content": "Haan boliye"},
      {"role": "user", "content": "Meri salary 12 lakh hai"},
      {"role": "user", "content": "Ok kitna cost hoga?"},
      {"role": "user", "content": "Theek hai let me think"}
    ]
  },
  {
    "scenario": "User objects mid-pitch",
    "eval_criteria": ["flow_correctness", "language", "edge_case"],
    "turns": [
      {"role": "user", "content": "Hello?"},
      {"role": "user", "content": "Mujhe interest nahi hai"},
      {"role": "user", "content": "Nahi nahi bye"}
    ]
  }
]
```

Each turn is a user message. The agent responds to each turn sequentially, building up the full conversation context.

### CLI usage

```bash
deepeval test run test_conversation.py -- --tb=short
```

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
| `conversation_goldens.json` | Multi-turn conversation scenarios |
| `.env` | API key (not committed) |

## File Structure

```
├── dashboard.py              ← Web dashboard (http://localhost:8050)
├── config.json               ← LLM settings
├── optimizer_config.json     ← Optimizer algorithm/metric settings
├── system_prompt.txt         ← Your system prompt
├── goldens.json              ← Prompt test cases
├── tool_goldens.json         ← Tool test cases
├── conversation_goldens.json ← Multi-turn conversation scenarios
├── .env                      ← API key
├── llm_client.py             ← LLM wrapper
├── test_prompts.py           ← Prompt quality tests
├── test_tools.py             ← Tool correctness tests
├── test_conversation.py      ← Conversation flow tests
├── optimize_prompt.py        ← Prompt optimizer
├── results.json              ← Run history (read by dashboard)
├── optimized_prompts.txt     ← Saved optimized prompts
└── logo.png                  ← Revnyx logo
```
