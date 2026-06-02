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

## Dashboard (Recommended)

The easiest way to use everything:

```bash
python3 dashboard.py
```

Open **http://localhost:8050** in your browser.

From the dashboard you can:
- Paste a system prompt directly
- Click **Test Prompt** — runs prompt quality evaluation
- Click **Test Tools** — runs tool correctness evaluation
- Click **Optimize Prompt** — runs the optimizer and shows the improved prompt

All results appear in the run history below with expandable details:
- Prompt used and optimized prompt (if applicable)
- Test scores table (metric, score, pass/fail)
- Tool test results in a separate tab
- Areas of improvement highlighting what failed and why

## CLI Usage

### Evaluate prompt quality

```bash
deepeval test run test_prompts.py -- --tb=short
```

Runs 3 metrics against every golden:
- **AnswerRelevancy** — is the response relevant to the question?
- **Hallucination** — is the LLM making things up beyond the context?
- **Helpfulness (GEval)** — custom criteria for overall helpfulness

### Optimize your prompt

```bash
python3 optimize_prompt.py
```

Takes the prompt from `system_prompt.txt`, runs it through the GEPA algorithm against all goldens, and saves the improved prompt to `optimized_prompts.txt`.

### Evaluate tool correctness

```bash
deepeval test run test_tools.py -- --tb=short
```

Checks whether your agent calls the right tools with the right arguments.

To add your own tool test cases, edit `test_tools.py` and add new test functions:

```python
def test_my_tool():
    test_case = LLMTestCase(
        input="User's question that should trigger a tool",
        actual_output="What your agent responded",
        tools_called=[
            ToolCall(name="ToolName", input={"arg": "actual_value"})
        ],
        expected_tools=[
            ToolCall(name="ToolName", input={"arg": "expected_value"})
        ],
    )
    metric = ToolCorrectnessMetric(threshold=0.7)
    metric.measure(test_case)
    _save_result("User's question", "ToolCorrectness", metric.score, metric.score >= 0.7, metric.reason)
    assert_test(test_case, [metric])
```

Each test case needs:
- `tools_called` — what your agent **actually** called
- `expected_tools` — what it **should** have called

Log tool calls from your agent's execution and paste them here to evaluate.

## File Structure

```
├── config.json             ← LLM settings
├── system_prompt.txt       ← Your system prompt
├── goldens.json            ← Test dataset (inputs + expected outputs)
├── .env                    ← API key (not committed)
├── llm_client.py           ← LLM wrapper (reads config + .env)
├── test_prompts.py         ← Prompt quality tests
├── test_tools.py           ← Tool correctness tests
├── optimize_prompt.py      ← Prompt optimizer script
├── dashboard.py            ← Local web dashboard (http://localhost:8050)
├── results.json            ← Stored run results (read by dashboard)
└── optimized_prompts.txt   ← Appended optimized prompts with timestamps
```
