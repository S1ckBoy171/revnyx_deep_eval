# Graph Report - .  (2026-06-03)

## Corpus Check
- Corpus is ~32,415 words - fits in a single context window. You may not need a graph.

## Summary
- 83 nodes · 90 edges · 17 communities (13 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Evaluation Framework Core|Evaluation Framework Core]]
- [[_COMMUNITY_Dashboard HTTP Server|Dashboard HTTP Server]]
- [[_COMMUNITY_LLM Client Module|LLM Client Module]]
- [[_COMMUNITY_Conversation Testing|Conversation Testing]]
- [[_COMMUNITY_Prompt Quality Testing|Prompt Quality Testing]]
- [[_COMMUNITY_Evaluation Concepts|Evaluation Concepts]]
- [[_COMMUNITY_Prompt Optimization|Prompt Optimization]]
- [[_COMMUNITY_Optimizer Configuration|Optimizer Configuration]]
- [[_COMMUNITY_Tool Correctness Testing|Tool Correctness Testing]]
- [[_COMMUNITY_LLM Configuration|LLM Configuration]]
- [[_COMMUNITY_Claude Permissions|Claude Permissions]]
- [[_COMMUNITY_Dashboard Handler Bridge|Dashboard Handler Bridge]]
- [[_COMMUNITY_LLM Client Bridge|LLM Client Bridge]]
- [[_COMMUNITY_Project Documentation|Project Documentation]]

## God Nodes (most connected - your core abstractions)
1. `Conversation Flow Tests` - 7 edges
2. `DashboardHandler` - 6 edges
3. `call()` - 6 edges
4. `Dashboard Server` - 5 edges
5. `Prompt Optimizer Script` - 5 edges
6. `Results History` - 5 edges
7. `llm_client.call` - 5 edges
8. `model_callback()` - 4 edges
9. `_save_result()` - 4 edges
10. `Prompt Quality Tests` - 4 edges

## Surprising Connections (you probably didn't know these)
- `Prompt Quality Tests` --implements--> `LLM-as-Judge Evaluation`  [INFERRED]
  test_prompts.py → DEEPEVAL_REFERENCE.txt
- `Dashboard Server` --references--> `Revnyx Logo`  [EXTRACTED]
  dashboard.py → logo.png
- `Prompt Optimizer Script` --implements--> `Prompt Optimization`  [EXTRACTED]
  optimize_prompt.py → DEEPEVAL_REFERENCE.txt
- `Prompt Quality Tests` --semantically_similar_to--> `Conversation Flow Tests`  [INFERRED] [semantically similar]
  test_prompts.py → test_conversation.py
- `Dashboard Server` --shares_data_with--> `Tool Goldens Dataset`  [EXTRACTED]
  dashboard.py → tool_goldens.json

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Evaluation Pipeline (Test -> LLM Client -> Results)** — revnyx_deep_eval_test_prompts, revnyx_deep_eval_test_tools, revnyx_deep_eval_test_conversation, revnyx_deep_eval_llm_client, revnyx_deep_eval_results [EXTRACTED 1.00]
- **Dashboard Orchestrates All Test Types** — revnyx_deep_eval_dashboard, revnyx_deep_eval_test_prompts, revnyx_deep_eval_test_tools, revnyx_deep_eval_test_conversation, revnyx_deep_eval_optimize_prompt [EXTRACTED 1.00]
- **Golden Test Datasets** — revnyx_deep_eval_conversation_goldens, revnyx_deep_eval_tool_goldens, concept_golden [INFERRED 0.95]

## Communities (17 total, 4 thin omitted)

### Community 0 - "Evaluation Framework Core"
Cohesion: 0.16
Nodes (18): Multi-Turn Conversation Flow Testing, GEPA Algorithm, Hinglish Voice Call Agent, Prompt Optimization, LLM Config, Conversation Goldens Dataset, Dashboard Server, llm_client.call (+10 more)

### Community 1 - "Dashboard HTTP Server"
Cohesion: 0.29
Nodes (3): BaseHTTPRequestHandler, DashboardHandler, Local dashboard server. Run with: python3 dashboard.py Then open: http://localho

### Community 2 - "LLM Client Module"
Cohesion: 0.36
Nodes (7): call(), get_client(), load_config(), load_system_prompt(), str, LLM client wrapper for the Revnyx DeepEval project. - Config (model, temperature, Call the LLM with the given input and return the response text.

### Community 3 - "Conversation Testing"
Cohesion: 0.38
Nodes (5): Conversation flow evaluation tests. Run with: deepeval test run test_conversatio, Run a full conversation, returning list of (user_msg, agent_response) pairs., run_conversation(), _save_result(), test_conversation_flow()

### Community 4 - "Prompt Quality Testing"
Cohesion: 0.43
Nodes (5): Prompt quality evaluation tests. Run with: deepeval test run test_prompts.py or, _save_result(), test_answer_relevancy(), test_custom_geval(), test_hallucination()

### Community 5 - "Evaluation Concepts"
Cohesion: 0.33
Nodes (6): Golden (Test Reference Data Point), LLM-as-Judge Evaluation, Tool Correctness Evaluation, DeepEval Reference Guide, Tool Correctness Tests, Tool Goldens Dataset

### Community 6 - "Prompt Optimization"
Cohesion: 0.33
Nodes (5): Golden, model_callback(), str, Prompt optimization script. Run with: python3 optimize_prompt.py  Uses DeepEval', Prompt

### Community 7 - "Optimizer Configuration"
Cohesion: 0.40
Nodes (4): algorithm, iterations, metric, threshold

### Community 8 - "Tool Correctness Testing"
Cohesion: 0.50
Nodes (3): Tool call evaluation tests. Run with: deepeval test run test_tools.py, _save_result(), test_tool_correctness()

### Community 9 - "LLM Configuration"
Cohesion: 0.50
Nodes (3): max_tokens, model, temperature

## Knowledge Gaps
- **20 isolated node(s):** `allow`, `model`, `temperature`, `max_tokens`, `str` (+15 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Results History` connect `Evaluation Framework Core` to `Evaluation Concepts`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **What connects `allow`, `model`, `temperature` to the rest of the system?**
  _29 weakly-connected nodes found - possible documentation gaps or missing edges._