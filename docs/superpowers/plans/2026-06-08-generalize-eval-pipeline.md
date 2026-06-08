# Generalize Evaluation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DeepEval pipeline fully generic — all metrics, cohorts, template variables, and tool definitions come from the dashboard UI (stored in JSON config), not hardcoded in Python.

**Architecture:** A new `eval_config.json` stores user-defined custom metrics (GEval criteria), cohort definitions, and template variables. The test files read this config at runtime. The dashboard provides full CRUD for these settings plus a "Test Config" button to validate the pipeline works end-to-end.

**Tech Stack:** Python, DeepEval, OpenAI SDK, vanilla JS dashboard (existing)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `eval_config.json` (CREATE) | User-defined metrics, cohorts, template vars — all editable from dashboard |
| `config.json` (MODIFY) | Keep only model settings + tools (tools also dashboard-editable) |
| `test_prompts.py` (REWRITE) | Reads metrics from eval_config.json, no hardcoded criteria |
| `test_conversation.py` (REWRITE) | Reads cohorts + metrics from eval_config.json, no hardcoded flows |
| `test_tools.py` (MODIFY) | Minor — already fairly generic |
| `optimize_prompt.py` (MODIFY) | Reads metrics from eval_config.json |
| `optimize_conversation.py` (MODIFY) | Reads metrics from eval_config.json |
| `llm_client.py` (KEEP) | Already generic |
| `dashboard.py` (MODIFY) | Add "Custom Metrics" section, "Cohorts & Variables" section, "Test Config" button, tool editor |

---

### Task 1: Create eval_config.json (generic config schema)

**Files:**
- Create: `eval_config.json`

- [ ] **Step 1: Write the eval_config.json with empty-but-structured schema**

```json
{
  "custom_metrics": [],
  "cohorts": [],
  "template_variables": {},
  "conversation_metrics": [],
  "builtin_metrics": {
    "answer_relevancy": {"enabled": true, "threshold": 0.8},
    "hallucination": {"enabled": true, "threshold": 0.7}
  }
}
```

`custom_metrics` array holds objects like:
```json
{
  "name": "LanguageCompliance",
  "criteria": "The response uses proper Hinglish...",
  "threshold": 0.85,
  "eval_params": ["INPUT", "ACTUAL_OUTPUT"],
  "apply_to": "all"
}
```

`cohorts` array holds:
```json
{
  "name": "inactive",
  "flow_criteria": "Agent follows inactive cohort flow...",
  "template_vars": {"participantName": "...", "cohort": "inactive", "gender": "female"}
}
```

- [ ] **Step 2: Commit**

---

### Task 2: Rewrite config.json to be model-only + tools

**Files:**
- Modify: `config.json`

- [ ] **Step 1: Slim config.json to only model settings and tools**

Remove `template_variables` (moved to eval_config.json). Keep:
- model, optimizer_model, temperature, max_tokens
- tools array (these stay because they're needed by the LLM client)

- [ ] **Step 2: Commit**

---

### Task 3: Rewrite test_prompts.py to be fully config-driven

**Files:**
- Modify: `test_prompts.py`

- [ ] **Step 1: Replace all hardcoded GEval metrics with config loader**

The file should:
1. Load `eval_config.json`
2. Build metrics list from `custom_metrics` where `apply_to` is "all" or "single_turn"
3. Add enabled builtin metrics (AnswerRelevancy, Hallucination)
4. Run all metrics against each golden with 1 LLM call

No hardcoded criteria strings. If eval_config has zero custom metrics, only builtins run.

- [ ] **Step 2: Commit**

---

### Task 4: Rewrite test_conversation.py to be config-driven

**Files:**
- Modify: `test_conversation.py`

- [ ] **Step 1: Replace hardcoded cohort logic with eval_config reader**

The file should:
1. Load `eval_config.json`
2. Get flow metric per conversation golden by matching `golden.cohort` to `cohorts[].name` → use that cohort's `flow_criteria`
3. If no match, use a generic "agent follows logical conversation flow" fallback
4. Get conversation-level metrics from `conversation_metrics` array
5. Template vars come from the golden itself (as stored by dashboard)

No `get_flow_metric()` with hardcoded criteria per cohort.

- [ ] **Step 2: Commit**

---

### Task 5: Update optimize_prompt.py and optimize_conversation.py

**Files:**
- Modify: `optimize_prompt.py`
- Modify: `optimize_conversation.py`

- [ ] **Step 1: Both optimizers load metrics from eval_config.json**

Same pattern as test files — build metrics list from config, not hardcoded.

- [ ] **Step 2: Commit**

---

### Task 6: Dashboard — Add "Custom Metrics" editor section

**Files:**
- Modify: `dashboard.py`

- [ ] **Step 1: Add a new card "Evaluation Metrics" between System Prompt and Goldens**

UI:
- List of user-defined custom metrics (name, criteria textarea, threshold slider, apply_to dropdown)
- "+ Add Metric" button
- Toggle switches for builtin metrics (AnswerRelevancy, Hallucination)
- "Save Metrics" button → POST /api/eval_config

- [ ] **Step 2: Add API endpoints**

- GET `/api/eval_config` — returns eval_config.json
- POST `/api/eval_config` — saves eval_config.json

- [ ] **Step 3: Commit**

---

### Task 7: Dashboard — Add "Cohorts & Template Variables" editor

**Files:**
- Modify: `dashboard.py`

- [ ] **Step 1: Add a collapsible section inside the Conversation Scenarios card**

UI:
- "Manage Cohorts" button that expands editor
- List of cohorts (name input, flow criteria textarea, template vars JSON input)
- "+ Add Cohort" button
- "Save Cohorts" button
- When user adds a conversation golden, cohort dropdown is populated from this config (not hardcoded)

- [ ] **Step 2: Update renderConvGoldens to read cohort list from eval_config**

The `<select>` for cohort builds `<option>` tags from loaded config, not a static list.

- [ ] **Step 3: Commit**

---

### Task 8: Dashboard — Add "Tools" editor section

**Files:**
- Modify: `dashboard.py`

- [ ] **Step 1: Add a collapsible "Tools" editor inside the Tool Test Cases card**

UI:
- "Manage Tools" button → shows list of defined tools
- Each tool: name, description, parameters JSON
- "+ Add Tool" / remove
- "Save Tools" → updates config.json tools array

- [ ] **Step 2: Add API endpoint**

- POST `/api/tools` — updates config.json's tools array

- [ ] **Step 3: Commit**

---

### Task 9: Dashboard — Add "Test Config" validation button

**Files:**
- Modify: `dashboard.py`

- [ ] **Step 1: Add a "Test Config" button in the settings area**

When clicked:
1. Saves current config
2. Runs a single golden through all metrics (or a dry-run with a dummy input)
3. Shows pass/fail with metric names and scores
4. Validates: API key works, metrics load, tools parse, template vars inject

- [ ] **Step 2: Add API endpoint**

- POST `/api/test_config` — runs validation, returns results

Backend logic:
1. Load eval_config.json, config.json
2. Create one test case with input "test" and call LLM
3. Build all metrics from config
4. Measure each metric
5. Return scores + any errors

- [ ] **Step 3: Commit**

---

### Task 10: Final cleanup and verify

- [ ] **Step 1: Remove any remaining hardcoded criteria strings from Python files**
- [ ] **Step 2: Run dashboard, test all sections load correctly**
- [ ] **Step 3: Verify "Test Config" works end to end**
- [ ] **Step 4: Final commit**

---
