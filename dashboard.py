"""
Local dashboard server.
Run with: python3 dashboard.py
Then open: http://localhost:8050
"""

import json
import os
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PORT = 8050
RESULTS_FILE = "results.json"
GOLDENS_FILE = "goldens.json"
TOOL_GOLDENS_FILE = "tool_goldens.json"

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Revnyx DeepEval Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e1e4e8; padding: 24px; }
h1 { font-size: 24px; margin-bottom: 8px; color: #fff; }
.subtitle { color: #8b949e; margin-bottom: 24px; font-size: 14px; }

/* Control Panel */
.control-panel { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 24px; margin-bottom: 20px; }
.control-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #fff; }
.prompt-input { width: 100%; min-height: 100px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; font-family: 'SF Mono', Menlo, monospace; font-size: 13px; color: #e1e4e8; resize: vertical; line-height: 1.5; }
.prompt-input:focus { outline: none; border-color: #58a6ff; }
.btn-row { display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
.btn { padding: 10px 20px; border-radius: 6px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-test { background: #238636; color: #fff; }
.btn-test:hover:not(:disabled) { background: #2ea043; }
.btn-tools { background: #1f6feb; color: #fff; }
.btn-tools:hover:not(:disabled) { background: #388bfd; }
.btn-optimize { background: #8957e5; color: #fff; }
.btn-optimize:hover:not(:disabled) { background: #a371f7; }
.status-bar { margin-top: 12px; padding: 10px 14px; border-radius: 6px; font-size: 13px; display: none; }
.status-bar.status-running { display: block; background: #1c2128; border: 1px solid #30363d; color: #8b949e; }
.status-bar.status-done { display: block; background: #1b4332; border: 1px solid #2d6a4f; color: #6ee7b7; }
.status-bar.status-error { display: block; background: #4c1d1d; border: 1px solid #7f1d1d; color: #fca5a5; }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #30363d; border-top-color: #58a6ff; border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 8px; vertical-align: middle; }
@keyframes spin { to { transform: rotate(360deg); } }
.disabled-notice { color: #f59e0b; font-size: 12px; margin-top: 8px; display: none; }

/* Goldens Panel */
.goldens-panel { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 24px; margin-bottom: 32px; }
.goldens-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
.goldens-title { font-size: 16px; font-weight: 600; color: #fff; }
.goldens-count { font-size: 13px; color: #8b949e; margin-left: 8px; }
.goldens-actions { display: flex; gap: 8px; }
.btn-sm { padding: 6px 14px; border-radius: 6px; border: none; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
.btn-add { background: #238636; color: #fff; }
.btn-add:hover { background: #2ea043; }
.btn-ai { background: #da7821; color: #fff; display: flex; align-items: center; gap: 6px; position: relative; }
.btn-ai:hover { background: #e8923b; }
.ai-icon { width: 16px; height: 16px; filter: invert(1); }
.btn-ai-wrap { display: flex; flex-direction: column; align-items: flex-end; gap: 2px; }
.btn-ai-desc { font-size: 11px; color: #8b949e; }
.golden-item { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 16px; margin-bottom: 12px; position: relative; }
.golden-item.ai-generated { border-color: #da7821; }
.golden-remove { position: absolute; top: 10px; right: 10px; background: #4c1d1d; border: none; color: #fca5a5; width: 24px; height: 24px; border-radius: 4px; cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center; }
.golden-remove:hover { background: #7f1d1d; }
.golden-field { margin-bottom: 10px; }
.golden-field:last-child { margin-bottom: 0; }
.golden-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #8b949e; margin-bottom: 4px; }
.golden-input { width: 100%; background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 8px 12px; font-size: 13px; color: #e1e4e8; font-family: inherit; }
.golden-input:focus { outline: none; border-color: #58a6ff; }
.golden-textarea { width: 100%; background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 8px 12px; font-size: 13px; color: #e1e4e8; font-family: inherit; min-height: 50px; resize: vertical; }
.golden-textarea:focus { outline: none; border-color: #58a6ff; }
.goldens-empty { color: #6b7280; text-align: center; padding: 30px; font-size: 14px; }

/* AI Generation Section */
.ai-section { background: #1c1917; border: 1px solid #da7821; border-radius: 8px; padding: 16px; margin-bottom: 16px; display: none; }
.ai-section.visible { display: block; }
.ai-label { font-size: 13px; color: #fbbf24; margin-bottom: 8px; font-weight: 600; }
.ai-textarea { width: 100%; min-height: 80px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; font-size: 13px; color: #e1e4e8; resize: vertical; font-family: inherit; }
.ai-textarea:focus { outline: none; border-color: #da7821; }
.ai-row { display: flex; gap: 12px; margin-top: 12px; align-items: center; }
.ai-count-label { font-size: 13px; color: #8b949e; }
.ai-count-input { width: 60px; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 6px 10px; font-size: 13px; color: #e1e4e8; text-align: center; }
.ai-count-input:focus { outline: none; border-color: #da7821; }
.btn-generate { background: #da7821; color: #fff; padding: 8px 18px; border-radius: 6px; border: none; font-size: 13px; font-weight: 600; cursor: pointer; }
.btn-generate:hover { background: #e8923b; }
.btn-generate:disabled { opacity: 0.5; cursor: not-allowed; }
.ai-cancel { background: transparent; border: 1px solid #30363d; color: #8b949e; padding: 8px 14px; border-radius: 6px; font-size: 13px; cursor: pointer; }
.ai-cancel:hover { border-color: #8b949e; color: #e1e4e8; }
.ai-status { margin-top: 12px; padding: 12px 16px; border-radius: 6px; background: #1c2128; border: 1px solid #da7821; font-size: 13px; color: #fbbf24; display: none; align-items: center; }
.ai-status.visible { display: flex; }
.btn-save-goldens { background: #238636; color: #fff; padding: 10px 20px; border-radius: 6px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; margin-top: 12px; }
.btn-save-goldens:hover { background: #2ea043; }

/* Divider */
.divider { border: none; border-top: 1px solid #21262d; margin: 32px 0 24px; }
.section-header { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #fff; }

/* Run Rows */
.empty { color: #6b7280; text-align: center; padding: 60px; font-size: 16px; }
.run-row { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 12px; overflow: hidden; transition: border-color 0.2s; }
.run-row:hover { border-color: #58a6ff; }
.run-header { display: flex; align-items: center; padding: 16px 20px; cursor: pointer; gap: 16px; }
.run-header .arrow { transition: transform 0.2s; color: #6b7280; font-size: 12px; }
.run-row.open .arrow { transform: rotate(90deg); }
.run-name { flex: 1; font-weight: 600; font-size: 15px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.run-date { color: #8b949e; font-size: 13px; white-space: nowrap; }
.run-duration { font-size: 12px; color: #58a6ff; font-family: monospace; background: #0d1117; padding: 2px 8px; border-radius: 4px; }
.run-badge { font-size: 11px; padding: 3px 8px; border-radius: 12px; font-weight: 600; }
.badge-pass { background: #1b4332; color: #6ee7b7; }
.badge-fail { background: #4c1d1d; color: #fca5a5; }
.badge-opt { background: #1e3a5f; color: #93c5fd; }
.badge-tools { background: #3b1f63; color: #c4b5fd; }
.run-details { display: none; padding: 0 20px 20px; border-top: 1px solid #30363d; }
.run-row.open .run-details { display: block; }
.section { margin-top: 16px; }
.section-title { font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #8b949e; margin-bottom: 8px; }
.prompt-box { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; font-family: 'SF Mono', Menlo, monospace; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; max-height: 200px; overflow-y: auto; }
.test-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.test-table th { text-align: left; padding: 8px 12px; background: #0d1117; color: #8b949e; font-weight: 500; border-bottom: 1px solid #30363d; }
.test-table td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
.test-table tr:last-child td { border-bottom: none; }
.score { font-weight: 600; font-family: monospace; }
.score-pass { color: #6ee7b7; }
.score-fail { color: #fca5a5; }
.reason-text { color: #8b949e; font-size: 12px; margin-top: 4px; }
.improvement { background: #1c1917; border-left: 3px solid #f59e0b; padding: 8px 12px; margin-top: 8px; border-radius: 0 4px 4px 0; font-size: 12px; color: #fbbf24; }
.tab-row { display: flex; gap: 0; margin-bottom: 12px; border-bottom: 1px solid #30363d; }
.tab { padding: 8px 16px; font-size: 13px; cursor: pointer; color: #8b949e; border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab.active { color: #58a6ff; border-bottom-color: #58a6ff; }
.tab-content { display: none; }
.tab-content.active { display: block; }
</style>
</head>
<body>
<h1>Revnyx DeepEval Dashboard</h1>
<p class="subtitle">Evaluate and optimize your LLM prompts</p>

<!-- Control Panel -->
<div class="control-panel">
  <div class="control-title">System Prompt</div>
  <textarea class="prompt-input" id="promptInput" placeholder="Paste your system prompt here... (leave empty to send only user input from goldens)"></textarea>
  <div class="btn-row">
    <button class="btn btn-test" id="btnTest" onclick="runAction('test_prompts')" disabled>Test Prompt</button>
    <button class="btn btn-tools" id="btnTools" onclick="runAction('test_tools')" disabled>Test Tools</button>
    <button class="btn btn-optimize" id="btnOptimize" onclick="runAction('optimize')" disabled>Optimize Prompt</button>
  </div>
  <div class="disabled-notice" id="disabledNotice">Add at least one golden below to enable evaluation.</div>
  <div class="status-bar" id="statusBar"></div>
</div>

<!-- Goldens Panel -->
<div class="goldens-panel">
  <div class="goldens-header">
    <div style="display:flex;align-items:center;">
      <span class="goldens-title">Goldens (Test Cases)</span>
      <span class="goldens-count" id="goldensCount">(0)</span>
    </div>
    <div class="goldens-actions">
      <div class="btn-ai-wrap">
        <button class="btn-sm btn-ai" onclick="toggleAiSection()"><img src="https://img.icons8.com/?size=100&id=rYb1JFR9WLSh&format=png&color=000000" class="ai-icon">Goldens</button>
        <span class="btn-ai-desc">Generate goldens with AI</span>
      </div>
      <button class="btn-sm btn-add" onclick="addGolden()">+ Add</button>
    </div>
  </div>

  <!-- AI Generation Section -->
  <div class="ai-section" id="aiSection">
    <div class="ai-label">What should the goldens test? (required)</div>
    <textarea class="ai-textarea" id="aiDescription" placeholder="e.g. Customer support questions about refunds, shipping, account management, billing issues..."></textarea>
    <div class="ai-row">
      <span class="ai-count-label">Number of goldens:</span>
      <input type="number" class="ai-count-input" id="aiCount" value="5" min="1" max="20">
      <button class="btn-generate" id="btnGenerate" onclick="generateGoldens()">Generate</button>
      <button class="ai-cancel" onclick="toggleAiSection()">Cancel</button>
    </div>
    <div class="ai-status" id="aiStatus"><span class="spinner"></span>Generating goldens with AI...</div>
  </div>

  <div id="goldensContainer"></div>
  <div class="goldens-empty" id="goldensEmpty">No goldens yet. Click "+ Add" to create manually or "Generate with AI" to auto-create.</div>
  <button class="btn-save-goldens" id="btnSaveGoldens" onclick="saveGoldens()" style="display:none;">Save Goldens</button>
</div>

<!-- Tool Goldens Panel -->
<div class="goldens-panel">
  <div class="goldens-header">
    <div style="display:flex;align-items:center;">
      <span class="goldens-title">Tool Test Cases</span>
      <span class="goldens-count" id="toolGoldensCount">(0)</span>
    </div>
    <div class="goldens-actions">
      <button class="btn-sm btn-add" onclick="addToolGolden()">+ Add</button>
    </div>
  </div>
  <div id="toolGoldensContainer"></div>
  <div class="goldens-empty" id="toolGoldensEmpty">No tool test cases yet. Click "+ Add" to define tool call expectations.</div>
  <button class="btn-save-goldens" id="btnSaveToolGoldens" onclick="saveToolGoldens()" style="display:none;">Save Tool Tests</button>
</div>

<hr class="divider">
<div class="section-header">Run History</div>
<div id="app"></div>

<script>
let goldens = [];
let toolGoldens = [];
let running = false;

// Load initial goldens
fetch('/api/goldens')
  .then(r => r.json())
  .then(data => { goldens = data; renderGoldens(); })
  .catch(() => renderGoldens());

// Load initial tool goldens
fetch('/api/tool_goldens')
  .then(r => r.json())
  .then(data => { toolGoldens = data; renderToolGoldens(); })
  .catch(() => renderToolGoldens());

function updateButtonState() {
  const hasGoldens = goldens.length > 0;
  const hasToolGoldens = toolGoldens.length > 0;
  document.getElementById('btnTest').disabled = !hasGoldens || running;
  document.getElementById('btnTools').disabled = !hasToolGoldens || running;
  document.getElementById('btnOptimize').disabled = !hasGoldens || running;
  document.getElementById('disabledNotice').style.display = hasGoldens ? 'none' : 'block';
}

function renderGoldens() {
  const container = document.getElementById('goldensContainer');
  const empty = document.getElementById('goldensEmpty');
  const count = document.getElementById('goldensCount');
  count.textContent = '(' + goldens.length + ')';

  if (!goldens.length) {
    container.innerHTML = '';
    empty.style.display = 'block';
    document.getElementById('btnSaveGoldens').style.display = 'none';
    updateButtonState();
    return;
  }

  empty.style.display = 'none';
  document.getElementById('btnSaveGoldens').style.display = 'block';

  container.innerHTML = goldens.map((g, i) => {
    const cls = g._aiGenerated ? ' ai-generated' : '';
    return `<div class="golden-item${cls}" data-index="${i}">
      <button class="golden-remove" onclick="removeGolden(${i})">&times;</button>
      <div class="golden-field"><div class="golden-label">Input (User Question)</div>
      <input class="golden-input" value="${escAttr(g.input || '')}" onchange="updateGolden(${i},'input',this.value)" placeholder="e.g. What is your refund policy?"></div>
      <div class="golden-field"><div class="golden-label">Expected Output (Ideal Answer)</div>
      <textarea class="golden-textarea" onchange="updateGolden(${i},'expected_output',this.value)" placeholder="e.g. We offer a 30-day full refund on all products.">${escHtml(g.expected_output || '')}</textarea></div>
      <div class="golden-field"><div class="golden-label">Context (Optional)</div>
      <textarea class="golden-textarea" onchange="updateGolden(${i},'context',this.value)" placeholder="Background info the LLM should stick to...">${escHtml((g.context || []).join('\\n'))}</textarea></div>
      </div>`;
  }).join('');

  updateButtonState();
}

function addGolden() {
  goldens.push({ input: '', expected_output: '', context: [] });
  renderGoldens();
  const items = document.querySelectorAll('.golden-item');
  const last = items[items.length - 1];
  last.scrollIntoView({ behavior: 'smooth', block: 'center' });
  setTimeout(() => { last.querySelector('.golden-input').focus(); }, 300);
}

function removeGolden(i) {
  goldens.splice(i, 1);
  renderGoldens();
}

function updateGolden(i, field, value) {
  if (field === 'context') {
    goldens[i].context = value.trim() ? value.split('\\n').filter(s => s.trim()) : [];
  } else {
    goldens[i][field] = value;
  }
  delete goldens[i]._aiGenerated;
}

function saveGoldens() {
  const clean = goldens.filter(g => g.input.trim()).map(g => {
    const obj = { input: g.input, expected_output: g.expected_output, context: g.context || [] };
    return obj;
  });
  fetch('/api/goldens', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(clean)
  })
  .then(r => r.json())
  .then(data => {
    if (data.success) {
      goldens = clean;
      renderGoldens();
      showQuickStatus('Goldens saved! (' + clean.length + ' entries)');
    }
  });
}

function showQuickStatus(msg) {
  const status = document.getElementById('statusBar');
  status.style.display = '';
  status.className = 'status-bar status-done';
  status.textContent = msg;
  setTimeout(() => { status.className = 'status-bar'; status.style.display = ''; }, 3000);
}

// Tool Goldens
function renderToolGoldens() {
  const container = document.getElementById('toolGoldensContainer');
  const empty = document.getElementById('toolGoldensEmpty');
  const count = document.getElementById('toolGoldensCount');
  count.textContent = '(' + toolGoldens.length + ')';

  if (!toolGoldens.length) {
    container.innerHTML = '';
    empty.style.display = 'block';
    document.getElementById('btnSaveToolGoldens').style.display = 'none';
    updateButtonState();
    return;
  }

  empty.style.display = 'none';
  document.getElementById('btnSaveToolGoldens').style.display = 'block';

  container.innerHTML = toolGoldens.map((g, i) => {
    return `<div class="golden-item" data-index="${i}">
      <button class="golden-remove" onclick="removeToolGolden(${i})">&times;</button>
      <div class="golden-field"><div class="golden-label">Input (User Message)</div>
      <input class="golden-input" value="${escAttr(g.input || '')}" onchange="updateToolGolden(${i},'input',this.value)" placeholder="e.g. Find me the latest news on AI"></div>
      <div class="golden-field"><div class="golden-label">Actual Output (Agent Response)</div>
      <textarea class="golden-textarea" onchange="updateToolGolden(${i},'actual_output',this.value)" placeholder="What the agent actually responded...">${escHtml(g.actual_output || '')}</textarea></div>
      <div class="golden-field"><div class="golden-label">Tools Called (Actual) — JSON array</div>
      <textarea class="golden-textarea" onchange="updateToolGolden(${i},'tools_called',this.value)" placeholder='[{"name": "WebSearch", "input": {"query": "AI news"}}]'>${escHtml(JSON.stringify(g.tools_called || [], null, 2))}</textarea></div>
      <div class="golden-field"><div class="golden-label">Expected Tools — JSON array</div>
      <textarea class="golden-textarea" onchange="updateToolGolden(${i},'expected_tools',this.value)" placeholder='[{"name": "WebSearch", "input": {"query": "latest AI news"}}]'>${escHtml(JSON.stringify(g.expected_tools || [], null, 2))}</textarea></div>
      </div>`;
  }).join('');

  updateButtonState();
}

function addToolGolden() {
  toolGoldens.push({ input: '', actual_output: '', tools_called: [], expected_tools: [] });
  renderToolGoldens();
  const items = document.querySelectorAll('#toolGoldensContainer .golden-item');
  const last = items[items.length - 1];
  last.scrollIntoView({ behavior: 'smooth', block: 'center' });
  setTimeout(() => { last.querySelector('.golden-input').focus(); }, 300);
}

function removeToolGolden(i) {
  toolGoldens.splice(i, 1);
  renderToolGoldens();
}

function updateToolGolden(i, field, value) {
  if (field === 'tools_called' || field === 'expected_tools') {
    try { toolGoldens[i][field] = JSON.parse(value); } catch(e) {}
  } else {
    toolGoldens[i][field] = value;
  }
}

function saveToolGoldens() {
  const clean = toolGoldens.filter(g => g.input.trim()).map(g => ({
    input: g.input, actual_output: g.actual_output,
    tools_called: g.tools_called || [], expected_tools: g.expected_tools || []
  }));
  fetch('/api/tool_goldens', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(clean)
  })
  .then(r => r.json())
  .then(data => {
    if (data.success) {
      toolGoldens = clean;
      renderToolGoldens();
      showQuickStatus('Tool tests saved! (' + clean.length + ' entries)');
    }
  });
}

// AI Generation
function toggleAiSection() {
  const section = document.getElementById('aiSection');
  section.classList.toggle('visible');
}

function generateGoldens() {
  const description = document.getElementById('aiDescription').value.trim();
  if (!description) {
    document.getElementById('aiDescription').style.borderColor = '#f87171';
    document.getElementById('aiDescription').focus();
    return;
  }
  document.getElementById('aiDescription').style.borderColor = '#30363d';

  const count = parseInt(document.getElementById('aiCount').value) || 5;
  const prompt = document.getElementById('promptInput').value;
  const btn = document.getElementById('btnGenerate');
  const statusEl = document.getElementById('aiStatus');

  btn.disabled = true;
  statusEl.classList.add('visible');

  fetch('/api/generate_goldens', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ description: description, count: count, system_prompt: prompt })
  })
  .then(r => r.json())
  .then(data => {
    btn.disabled = false;
    statusEl.classList.remove('visible');
    if (data.success && data.goldens) {
      data.goldens.forEach(g => { g._aiGenerated = true; });
      goldens = goldens.concat(data.goldens);
      renderGoldens();
      document.getElementById('aiSection').classList.remove('visible');
      document.getElementById('aiDescription').value = '';
      showQuickStatus('Generated ' + data.goldens.length + ' goldens! Review and edit them below, then click Save.');
    } else {
      alert('Error generating goldens: ' + (data.message || 'Unknown error'));
    }
  })
  .catch(err => {
    btn.disabled = false;
    statusEl.classList.remove('visible');
    alert('Error: ' + err.message);
  });
}

// Run Actions
function runAction(action) {
  if (running) return;

  // Check appropriate goldens exist
  if (action === 'test_tools') {
    const cleanTools = toolGoldens.filter(g => g.input.trim());
    if (!cleanTools.length) return;
  } else {
    const clean = goldens.filter(g => g.input.trim());
    if (!clean.length) return;
  }

  running = true;
  updateButtonState();
  const prompt = document.getElementById('promptInput').value;
  const status = document.getElementById('statusBar');
  status.style.display = '';
  status.className = 'status-bar status-running';
  status.innerHTML = '<span class="spinner"></span>Running ' + action.replace('_', ' ') + '...';
  const startTime = Date.now();

  // Update elapsed time every second
  const timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
    status.innerHTML = '<span class="spinner"></span>Running ' + action.replace('_', ' ') + '... (' + elapsed + 's)';
  }, 1000);

  // Save all data first, then run
  const savePromises = [
    fetch('/api/goldens', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(goldens.filter(g => g.input.trim()).map(g => ({
        input: g.input, expected_output: g.expected_output, context: g.context || []
      })))
    }),
    fetch('/api/tool_goldens', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(toolGoldens.filter(g => g.input.trim()).map(g => ({
        input: g.input, actual_output: g.actual_output,
        tools_called: g.tools_called || [], expected_tools: g.expected_tools || []
      })))
    })
  ];

  Promise.all(savePromises).then(() => {
    return fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: action, prompt: prompt})
    });
  })
  .then(r => r.json())
  .then(data => {
    clearInterval(timerInterval);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    running = false;
    updateButtonState();
    if (data.success) {
      status.className = 'status-bar status-done';
      status.textContent = 'Done in ' + elapsed + 's! ' + (data.message || '');
    } else {
      status.className = 'status-bar status-error';
      status.textContent = 'Error (' + elapsed + 's): ' + (data.message || 'Unknown error');
    }
    loadResults();
  })
  .catch(err => {
    clearInterval(timerInterval);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    running = false;
    updateButtonState();
    status.className = 'status-bar status-error';
    status.textContent = 'Error: ' + err.message;
  });
}

// Results
function loadResults() {
  fetch('/api/results')
    .then(r => r.json())
    .then(data => render(data.reverse()))
    .catch(() => render([]));
}
loadResults();

function render(runs) {
  const app = document.getElementById('app');
  if (!runs.length) {
    app.innerHTML = '<div class="empty">No runs yet. Paste a prompt above, add goldens, and click a button to get started.</div>';
    return;
  }
  app.innerHTML = runs.map((run, i) => renderRun(run, i)).join('');
  document.querySelectorAll('.run-header').forEach(h => {
    h.addEventListener('click', () => h.parentElement.classList.toggle('open'));
  });
  document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => {
      const group = t.closest('.run-details');
      group.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      group.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      group.querySelector('.tab-content[data-tab="' + t.dataset.tab + '"]').classList.add('active');
    });
  });
}

function renderRun(run, i) {
  const name = (run.system_prompt || 'No system prompt').substring(0, 50);
  const date = new Date(run.timestamp).toLocaleString();
  const duration = run.duration_seconds ? run.duration_seconds + 's' : '';
  const type = run.type;
  let badge = '';
  if (type === 'optimization') {
    badge = '<span class="run-badge badge-opt">OPTIMIZED</span>';
  } else if (type === 'tool_evaluation') {
    badge = '<span class="run-badge badge-tools">TOOLS</span>';
  } else if (run.tests && run.tests.every(t => t.passed)) {
    badge = '<span class="run-badge badge-pass">ALL PASSED</span>';
  } else {
    const failed = run.tests ? run.tests.filter(t => !t.passed).length : 0;
    badge = '<span class="run-badge badge-fail">' + failed + ' FAILED</span>';
  }

  let details = '';
  const hasPromptTests = run.tests && run.tests.length > 0;
  const hasToolTests = run.tool_tests && run.tool_tests.length > 0;
  const hasTabs = hasPromptTests && hasToolTests;

  details += '<div class="section"><div class="section-title">System Prompt Used</div>';
  details += '<div class="prompt-box">' + escHtml(run.system_prompt || '(empty)') + '</div></div>';

  if (run.optimized_prompt) {
    details += '<div class="section"><div class="section-title">Optimized Prompt</div>';
    details += '<div class="prompt-box">' + escHtml(run.optimized_prompt) + '</div></div>';
  }

  if (hasTabs) {
    details += '<div class="tab-row"><div class="tab active" data-tab="prompt">Prompt Tests</div><div class="tab" data-tab="tools">Tool Tests</div></div>';
  }

  if (hasPromptTests) {
    details += '<div class="tab-content' + (hasTabs ? ' active' : '') + '" data-tab="prompt">';
    details += '<div class="section"><div class="section-title">Prompt Test Results</div>';
    details += renderTestTable(run.tests);
    details += '</div>';
    const failures = run.tests.filter(t => !t.passed);
    if (failures.length) {
      details += '<div class="section"><div class="section-title">Areas of Improvement</div>';
      failures.forEach(f => {
        details += '<div class="improvement"><strong>' + escHtml(f.metric) + '</strong> on "' + escHtml(f.input.substring(0, 40)) + '"<br>' + escHtml(f.reason || 'No reason provided') + '</div>';
      });
      details += '</div>';
    }
    details += '</div>';
  }

  if (hasToolTests) {
    details += '<div class="tab-content' + (hasTabs ? '' : ' active') + '" data-tab="tools">';
    details += '<div class="section"><div class="section-title">Tool Test Results</div>';
    details += renderTestTable(run.tool_tests);
    details += '</div>';
    const toolFailures = run.tool_tests.filter(t => !t.passed);
    if (toolFailures.length) {
      details += '<div class="section"><div class="section-title">Tool Areas of Improvement</div>';
      toolFailures.forEach(f => {
        details += '<div class="improvement"><strong>' + escHtml(f.metric) + '</strong> on "' + escHtml(f.input.substring(0, 40)) + '"<br>' + escHtml(f.reason || 'No reason provided') + '</div>';
      });
      details += '</div>';
    }
    details += '</div>';
  }

  const durationHtml = duration ? '<span class="run-duration">' + duration + '</span>' : '';
  return '<div class="run-row"><div class="run-header"><span class="arrow">&#9654;</span><span class="run-name">' + escHtml(name) + '</span>' + badge + durationHtml + '<span class="run-date">' + date + '</span></div><div class="run-details">' + details + '</div></div>';
}

function renderTestTable(tests) {
  let html = '<table class="test-table"><thead><tr><th>Input</th><th>Metric</th><th>Score</th><th>Status</th></tr></thead><tbody>';
  tests.forEach(t => {
    const scoreClass = t.passed ? 'score-pass' : 'score-fail';
    const status = t.passed ? '&#10003; Pass' : '&#10007; Fail';
    html += '<tr>';
    html += '<td>' + escHtml(t.input.substring(0, 50)) + '</td>';
    html += '<td>' + escHtml(t.metric) + '</td>';
    html += '<td class="score ' + scoreClass + '">' + (t.score !== null && t.score !== undefined ? t.score.toFixed(2) : 'N/A') + '</td>';
    html += '<td class="' + scoreClass + '">' + status + '</td>';
    html += '</tr>';
    if (t.reason) {
      html += '<tr><td colspan="4"><div class="reason-text">' + escHtml(t.reason) + '</div></td></tr>';
    }
  });
  html += '</tbody></table>';
  return html;
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function escAttr(s) {
  return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == "/api/results":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE) as f:
                    self.wfile.write(f.read().encode())
            else:
                self.wfile.write(b"[]")
        elif self.path == "/api/goldens":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if os.path.exists(GOLDENS_FILE):
                with open(GOLDENS_FILE) as f:
                    self.wfile.write(f.read().encode())
            else:
                self.wfile.write(b"[]")
        elif self.path == "/api/tool_goldens":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if os.path.exists(TOOL_GOLDENS_FILE):
                with open(TOOL_GOLDENS_FILE) as f:
                    self.wfile.write(f.read().encode())
            else:
                self.wfile.write(b"[]")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(content_length))

        if self.path == "/api/goldens":
            with open(GOLDENS_FILE, "w") as f:
                json.dump(body, f, indent=2)
            self._json_response({"success": True})

        elif self.path == "/api/tool_goldens":
            with open(TOOL_GOLDENS_FILE, "w") as f:
                json.dump(body, f, indent=2)
            self._json_response({"success": True})

        elif self.path == "/api/generate_goldens":
            description = body.get("description", "")
            count = body.get("count", 5)
            system_prompt = body.get("system_prompt", "")

            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                gen_prompt = f"""Generate exactly {count} test cases (goldens) for evaluating an LLM.

Context about the LLM being tested:
System prompt: {system_prompt if system_prompt else '(no system prompt)'}

What the goldens should test:
{description}

Return a JSON array with exactly {count} objects, each having:
- "input": a realistic user question/message
- "expected_output": the ideal response the LLM should give
- "context": an array with 1-2 background facts that support the expected output

Return ONLY the JSON array, no other text."""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": gen_prompt}],
                    temperature=0.7,
                )
                content = response.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0]
                generated = json.loads(content)
                self._json_response({"success": True, "goldens": generated})
            except Exception as e:
                self._json_response({"success": False, "message": str(e)})

        elif self.path == "/api/run":
            action = body.get("action")
            prompt = body.get("prompt", "")

            with open("system_prompt.txt", "w") as f:
                f.write(prompt)

            try:
                if action == "test_prompts":
                    result = subprocess.run(
                        ["deepeval", "test", "run", "test_prompts.py", "--", "--tb=line", "-q"],
                        capture_output=True, text=True, timeout=300
                    )
                    success = result.returncode == 0
                    msg = "All tests passed!" if success else "Some tests failed. Check results below."
                elif action == "test_tools":
                    result = subprocess.run(
                        ["deepeval", "test", "run", "test_tools.py", "--", "--tb=line", "-q"],
                        capture_output=True, text=True, timeout=300
                    )
                    success = result.returncode == 0
                    msg = "All tool tests passed!" if success else "Some tool tests failed. Check results below."
                elif action == "optimize":
                    result = subprocess.run(
                        ["python3", "optimize_prompt.py"],
                        capture_output=True, text=True, timeout=600
                    )
                    success = result.returncode == 0
                    msg = "Optimization complete! Check the optimized prompt below." if success else result.stderr[-200:] if result.stderr else "Optimization failed."
                else:
                    success = False
                    msg = "Unknown action"

                self._json_response({"success": success, "message": msg})
            except subprocess.TimeoutExpired:
                self._json_response({"success": False, "message": "Timed out"})
            except Exception as e:
                self._json_response({"success": False, "message": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), DashboardHandler)
    print(f"Dashboard running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
