"""
Local dashboard server.
Run with: python3 dashboard.py
Then open: http://localhost:8050
"""

import json
import os
import re
import subprocess
import threading
import time as _time
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv
from openai import OpenAI

_ansi_re = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07')

load_dotenv()

PORT = 8050
RESULTS_FILE = "results.json"
GOLDENS_FILE = "goldens.json"
TOOL_GOLDENS_FILE = "tool_goldens.json"

_run_state = {"running": False, "log": "", "done": False, "success": False, "message": ""}

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Revnyx DeepEval</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg-deep: #09090b;
  --bg-surface: #111113;
  --bg-elevated: #18181b;
  --bg-hover: #1f1f23;
  --border: #27272a;
  --border-subtle: #1e1e21;
  --text-primary: #fafafa;
  --text-secondary: #a1a1aa;
  --text-muted: #71717a;
  --accent: #d4af37;
  --accent-dim: rgba(212,175,55,0.15);
  --success: #22c55e;
  --success-dim: rgba(34,197,94,0.12);
  --error: #ef4444;
  --error-dim: rgba(239,68,68,0.12);
  --blue: #3b82f6;
  --blue-dim: rgba(59,130,246,0.12);
  --purple: #a855f7;
  --purple-dim: rgba(168,85,247,0.12);
  --radius: 8px;
  --font: 'Outfit', sans-serif;
  --mono: 'JetBrains Mono', monospace;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { font-family: var(--font); background: var(--bg-deep); color: var(--text-primary); min-height: 100vh; line-height: 1.5; }

/* Fade-in animation */
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes glow { 0%, 100% { box-shadow: 0 0 0 rgba(212,175,55,0); } 50% { box-shadow: 0 0 20px rgba(212,175,55,0.08); } }

/* Layout */
.container { max-width: 900px; margin: 0 auto; padding: 48px 32px; }
@media (min-width: 1200px) { .container { max-width: 1000px; } }
@media (min-width: 1600px) { .container { max-width: 1100px; } }

/* Header */
.header { display: flex; align-items: center; gap: 16px; margin-bottom: 48px; animation: fadeUp 0.5s ease; }
.header img { width: 44px; height: 44px; border-radius: 10px; transition: transform 0.3s; }
.header img:hover { transform: scale(1.08) rotate(-2deg); }
.header-text h1 { font-size: 22px; font-weight: 600; letter-spacing: -0.3px; }
.header-text p { font-size: 13px; color: var(--text-muted); font-weight: 300; letter-spacing: 0.2px; }

/* Cards */
.card { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 28px; margin-bottom: 24px; transition: border-color 0.3s, box-shadow 0.3s, transform 0.2s; animation: fadeUp 0.5s ease backwards; }
.card:nth-child(2) { animation-delay: 0.08s; }
.card:nth-child(3) { animation-delay: 0.16s; }
.card:nth-child(4) { animation-delay: 0.24s; }
.card:hover { border-color: var(--border); box-shadow: 0 4px 24px rgba(0,0,0,0.3); }
.card-title { font-size: 13px; font-weight: 500; text-transform: uppercase; letter-spacing: 1.2px; color: var(--text-muted); margin-bottom: 16px; }

/* Prompt Input */
.prompt-area { width: 100%; min-height: 120px; background: var(--bg-deep); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 20px; font-family: var(--mono); font-size: 13px; color: var(--text-primary); resize: vertical; line-height: 1.7; transition: border-color 0.3s, box-shadow 0.3s; }
.prompt-area:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-dim); }

/* Buttons */
.btn-group { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
.btn { padding: 10px 22px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg-elevated); color: var(--text-primary); font-family: var(--font); font-size: 13px; font-weight: 500; cursor: pointer; transition: all 0.2s cubic-bezier(0.4,0,0.2,1); letter-spacing: 0.2px; }
.btn:hover:not(:disabled) { background: var(--bg-hover); border-color: var(--text-muted); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
.btn:active:not(:disabled) { transform: translateY(0); box-shadow: none; }
.btn:disabled { opacity: 0.3; cursor: not-allowed; }
.btn-accent { border-color: var(--accent); color: var(--accent); }
.btn-accent:hover:not(:disabled) { background: var(--accent-dim); }
.btn-success { border-color: var(--success); color: var(--success); }
.btn-success:hover:not(:disabled) { background: var(--success-dim); }
.btn-blue { border-color: var(--blue); color: var(--blue); }
.btn-blue:hover:not(:disabled) { background: var(--blue-dim); }
.btn-purple { border-color: var(--purple); color: var(--purple); }
.btn-purple:hover:not(:disabled) { background: var(--purple-dim); }
.btn-sm { padding: 7px 14px; font-size: 12px; }
.btn-gold { background: var(--accent); color: var(--bg-deep); border-color: var(--accent); font-weight: 600; }
.btn-gold:hover:not(:disabled) { background: #e6c84a; }

/* Status */
.status { margin-top: 16px; padding: 12px 16px; border-radius: var(--radius); font-size: 13px; display: none; align-items: center; gap: 10px; font-family: var(--mono); animation: fadeUp 0.3s ease; }
.status.visible { display: flex; }
.status-running { background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); }
.status-done { background: var(--success-dim); border: 1px solid rgba(34,197,94,0.25); color: var(--success); }
.status-error { background: var(--error-dim); border: 1px solid rgba(239,68,68,0.25); color: var(--error); }
.spinner { width: 14px; height: 14px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.disabled-msg { font-size: 12px; color: var(--accent); margin-top: 10px; display: none; font-weight: 400; }

/* Log Output */
.log-box { margin-top: 14px; background: var(--bg-deep); border: 1px solid var(--border-subtle); border-radius: var(--radius); padding: 16px; font-family: var(--mono); font-size: 11.5px; color: var(--text-muted); white-space: pre-wrap; word-break: break-word; max-height: 280px; overflow-y: auto; display: none; line-height: 1.8; }
.log-box.visible { display: block; }
.log-box::-webkit-scrollbar { width: 6px; }
.log-box::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Goldens */
.goldens-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }
.goldens-header-left { display: flex; align-items: baseline; gap: 8px; }
.goldens-count { font-size: 12px; color: var(--text-muted); font-weight: 400; }
.golden-item { background: var(--bg-deep); border: 1px solid var(--border-subtle); border-radius: var(--radius); padding: 18px; margin-bottom: 12px; position: relative; transition: all 0.25s cubic-bezier(0.4,0,0.2,1); animation: fadeUp 0.3s ease backwards; }
.golden-item:hover { border-color: var(--border); box-shadow: 0 2px 16px rgba(0,0,0,0.2); transform: translateY(-1px); }
.golden-item.ai-generated { border-color: var(--accent); border-style: dashed; animation: glow 2s ease infinite; }
.golden-remove { position: absolute; top: 12px; right: 12px; background: none; border: 1px solid var(--border); color: var(--text-muted); width: 22px; height: 22px; border-radius: 4px; cursor: pointer; font-size: 13px; display: flex; align-items: center; justify-content: center; transition: all 0.15s; }
.golden-remove:hover { border-color: var(--error); color: var(--error); }
.golden-field { margin-bottom: 12px; }
.golden-field:last-child { margin-bottom: 0; }
.golden-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 6px; font-weight: 500; }
.golden-input { width: 100%; background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 10px 14px; font-size: 13px; color: var(--text-primary); font-family: var(--font); transition: border-color 0.2s; }
.golden-input:focus { outline: none; border-color: var(--accent); }
.golden-textarea { width: 100%; background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 10px 14px; font-size: 13px; color: var(--text-primary); font-family: var(--mono); min-height: 60px; resize: vertical; line-height: 1.6; transition: border-color 0.2s; }
.golden-textarea:focus { outline: none; border-color: var(--accent); }
.goldens-empty { color: var(--text-muted); text-align: center; padding: 36px; font-size: 13px; }

/* AI Section */
.ai-section { background: var(--bg-deep); border: 1px dashed var(--accent); border-radius: var(--radius); padding: 18px; margin-bottom: 16px; display: none; }
.ai-section.visible { display: block; }
.ai-label { font-size: 12px; color: var(--accent); margin-bottom: 8px; font-weight: 500; letter-spacing: 0.3px; }
.ai-row { display: flex; gap: 12px; margin-top: 12px; align-items: center; flex-wrap: wrap; }
.ai-count-input { width: 56px; background: var(--bg-surface); border: 1px solid var(--border); border-radius: 6px; padding: 7px 10px; font-size: 13px; color: var(--text-primary); text-align: center; font-family: var(--mono); -moz-appearance: textfield; }
.ai-count-input:focus { outline: none; border-color: var(--accent); }
.ai-count-input::-webkit-inner-spin-button, .ai-count-input::-webkit-outer-spin-button { opacity: 1; filter: invert(1); }
.ai-status { margin-top: 10px; font-size: 12px; color: var(--text-muted); display: none; align-items: center; gap: 8px; }
.ai-status.visible { display: flex; }

/* Field Wrap + Expand */
.field-wrap { position: relative; }
.expand-btn { position: absolute; top: 8px; right: 8px; background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-muted); width: 22px; height: 22px; border-radius: 4px; cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center; opacity: 0; transition: all 0.15s; z-index: 2; }
.field-wrap:hover .expand-btn { opacity: 1; }
.expand-btn:hover { border-color: var(--accent); color: var(--accent); }

/* Divider */
.divider { border: none; border-top: 1px solid var(--border-subtle); margin: 40px 0 32px; }

/* Section Header */
.section-label { font-size: 13px; font-weight: 500; text-transform: uppercase; letter-spacing: 1.2px; color: var(--text-muted); margin-bottom: 20px; }

/* Run History */
.empty-state { color: var(--text-muted); text-align: center; padding: 60px 20px; font-size: 14px; border: 1px dashed var(--border); border-radius: var(--radius); }
.run-row { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 10px; margin-bottom: 10px; overflow: hidden; transition: all 0.25s cubic-bezier(0.4,0,0.2,1); animation: fadeUp 0.4s ease backwards; }
.run-row:nth-child(1) { animation-delay: 0s; }
.run-row:nth-child(2) { animation-delay: 0.05s; }
.run-row:nth-child(3) { animation-delay: 0.1s; }
.run-row:nth-child(4) { animation-delay: 0.15s; }
.run-row:nth-child(5) { animation-delay: 0.2s; }
.run-row:hover { border-color: var(--border); box-shadow: 0 4px 20px rgba(0,0,0,0.25); transform: translateY(-1px); }
.run-header { display: flex; align-items: center; padding: 16px 22px; cursor: pointer; gap: 14px; }
.run-header .arrow { color: var(--text-muted); font-size: 10px; transition: transform 0.2s; }
.run-row.open .arrow { transform: rotate(90deg); }
.run-name { flex: 1; font-size: 14px; font-weight: 400; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-secondary); }
.run-date { color: var(--text-muted); font-size: 12px; font-family: var(--mono); }
.run-duration { font-size: 11px; color: var(--accent); font-family: var(--mono); background: var(--accent-dim); padding: 2px 8px; border-radius: 4px; }
.badge { font-size: 10px; padding: 3px 10px; border-radius: 20px; font-weight: 600; letter-spacing: 0.3px; text-transform: uppercase; }
.badge-pass { background: var(--success-dim); color: var(--success); }
.badge-fail { background: var(--error-dim); color: var(--error); }
.badge-opt { background: var(--purple-dim); color: var(--purple); }
.badge-tools { background: var(--blue-dim); color: var(--blue); }
.run-details { display: none; padding: 0 22px 22px; border-top: 1px solid var(--border-subtle); }
.run-row.open .run-details { display: block; }

/* Summary Bar */
.summary-bar { display: flex; gap: 20px; padding: 14px 18px; background: var(--bg-deep); border: 1px solid var(--border-subtle); border-radius: var(--radius); margin-top: 18px; flex-wrap: wrap; align-items: center; }
.summary-item { font-size: 12px; font-weight: 500; font-family: var(--mono); }
.summary-pass { color: var(--success); }
.summary-fail { color: var(--error); }
.summary-total { color: var(--text-muted); }
.summary-score { color: var(--accent); }
.summary-time { color: var(--text-muted); }

/* Detail Sections */
.detail-section { margin-top: 18px; }
.detail-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 8px; font-weight: 500; }
.prompt-box { background: var(--bg-deep); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 14px 18px; font-family: var(--mono); font-size: 12px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; max-height: 180px; overflow-y: auto; color: var(--text-secondary); }

/* VS Code style diff */
.diff-container { display: grid; grid-template-columns: 1fr 1px 1fr; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; min-width: 0; }
.diff-pane { overflow-y: auto; max-height: 60vh; background: var(--bg-deep); }
.diff-pane-header { font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); padding: 10px 14px; background: var(--bg-elevated); border-bottom: 1px solid var(--border-subtle); font-weight: 500; position: sticky; top: 0; z-index: 1; }
.diff-line { display: flex; min-height: 22px; font-family: var(--mono); font-size: 12px; line-height: 22px; }
.diff-ln { min-width: 40px; text-align: right; padding-right: 12px; color: var(--text-muted); opacity: 0.5; user-select: none; flex-shrink: 0; }
.diff-text { flex: 1; padding: 0 8px; white-space: pre-wrap; word-break: break-word; }
.diff-del { background: rgba(239,68,68,0.13); }
.diff-del .diff-text { color: var(--error); }
.diff-add { background: rgba(34,197,94,0.13); }
.diff-add .diff-text { color: var(--success); }
.diff-empty { background: var(--bg-surface); opacity: 0.4; }
.diff-divider { width: 1px; background: var(--border); }

/* Test Table */
.test-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.test-table th { text-align: left; padding: 10px 14px; background: var(--bg-deep); color: var(--text-muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
.test-table td { padding: 10px 14px; border-top: 1px solid var(--border-subtle); }
.score { font-weight: 600; font-family: var(--mono); }
.score-pass { color: var(--success); }
.score-fail { color: var(--error); }
.reason-text { color: var(--text-muted); font-size: 11px; padding: 4px 14px 10px; font-style: italic; }
.improvement { background: var(--bg-deep); border-left: 2px solid var(--accent); padding: 10px 14px; margin-top: 8px; border-radius: 0 6px 6px 0; font-size: 12px; color: var(--text-secondary); }
.improvement strong { color: var(--accent); }

/* Tabs */
.tab-row { display: flex; gap: 0; margin-top: 14px; border-bottom: 1px solid var(--border-subtle); }
.tab { padding: 10px 18px; font-size: 12px; cursor: pointer; color: var(--text-muted); border-bottom: 2px solid transparent; transition: all 0.15s; font-weight: 500; }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Modal */
.modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 1000; align-items: center; justify-content: center; padding: 40px; backdrop-filter: blur(8px); opacity: 0; transition: opacity 0.2s; }
.modal-overlay.visible { display: flex; opacity: 1; }
.modal-box { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 14px; width: 100%; max-width: 860px; height: 75vh; display: flex; flex-direction: column; overflow: hidden; transform: scale(0.96); transition: transform 0.25s cubic-bezier(0.4,0,0.2,1); }
.modal-overlay.visible .modal-box { transform: scale(1); }
.modal-header { display: flex; align-items: center; justify-content: space-between; padding: 18px 24px; border-bottom: 1px solid var(--border-subtle); }
.modal-title { font-size: 13px; font-weight: 500; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
.modal-close { background: none; border: 1px solid var(--border); color: var(--text-muted); width: 28px; height: 28px; border-radius: 6px; cursor: pointer; font-size: 16px; display: flex; align-items: center; justify-content: center; transition: all 0.15s; }
.modal-close:hover { border-color: var(--error); color: var(--error); }
.modal-body { flex: 1; padding: 24px; overflow-y: auto; }
.modal-textarea { width: 100%; height: 100%; min-height: 100%; background: var(--bg-deep); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; font-family: var(--mono); font-size: 14px; color: var(--text-primary); resize: none; line-height: 1.8; }
.modal-textarea:focus { outline: none; border-color: var(--accent); }
.modal-footer { display: flex; justify-content: flex-end; gap: 10px; padding: 14px 24px; border-top: 1px solid var(--border-subtle); }
</style>
</head>
<body>
<div class="container">

<!-- Header -->
<div class="header">
  <img src="/logo.png" alt="Revnyx">
  <div class="header-text">
    <h1>Revnyx DeepEval</h1>
    <p>Prompt evaluation & optimization</p>
  </div>
</div>

<!-- System Prompt -->
<div class="card">
  <div class="card-title">System Prompt</div>
  <div class="field-wrap">
    <textarea class="prompt-area" id="promptInput" placeholder="Enter your system prompt here..."></textarea>
    <button class="expand-btn" onclick="openModal(document.getElementById('promptInput'))">&#x26F6;</button>
  </div>
  <div class="btn-group">
    <button class="btn btn-success" id="btnTest" onclick="runAction('test_prompts')" disabled>Test Prompt</button>
    <button class="btn btn-blue" id="btnTools" onclick="runAction('test_tools')" disabled>Test Tools</button>
    <button class="btn btn-accent" id="btnConv" onclick="runAction('test_conversation')" disabled>Test Conversation</button>
    <button class="btn btn-purple" id="btnOptimize" onclick="runAction('optimize')" disabled>Optimize Prompt</button>
    <button class="btn btn-accent" id="btnOptConv" onclick="runAction('optimize_conversation')" disabled>Optimize Conversation</button>
    <button class="btn btn-sm" onclick="document.getElementById('optSettings').classList.toggle('visible')" style="margin-left:auto;opacity:0.7;">&#9881; Settings</button>
  </div>
  <div class="ai-section" id="optSettings">
    <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:center;">
      <div><div class="golden-label">Algorithm</div>
        <select class="golden-input" id="optAlgo" style="width:140px;cursor:pointer;">
          <option value="GEPA">GEPA</option>
          <option value="MIPROv2">MIPROv2</option>
          <option value="COPRO">COPRO</option>
          <option value="SIMBA">SIMBA</option>
        </select></div>
      <div><div class="golden-label">Iterations</div>
        <input type="number" class="ai-count-input" id="optIter" value="10" min="1" max="30"></div>
      <div><div class="golden-label">Metrics</div>
        <div style="display:flex;flex-direction:column;gap:4px;">
          <label style="font-size:11px;color:var(--text-secondary);display:flex;align-items:center;gap:4px;"><input type="checkbox" id="optMetricRelevancy" checked> Answer Relevancy</label>
          <label style="font-size:11px;color:var(--text-secondary);display:flex;align-items:center;gap:4px;"><input type="checkbox" id="optMetricLanguage" checked> Language Compliance</label>
          <label style="font-size:11px;color:var(--text-secondary);display:flex;align-items:center;gap:4px;"><input type="checkbox" id="optMetricCorrectness" checked> Correctness</label>
          <label style="font-size:11px;color:var(--text-secondary);display:flex;align-items:center;gap:4px;"><input type="checkbox" id="optMetricHallucination"> Hallucination</label>
          <label style="font-size:11px;color:var(--text-secondary);display:flex;align-items:center;gap:4px;"><input type="checkbox" id="optMetricHelpfulness"> Helpfulness</label>
        </div></div>
      <div><div class="golden-label">Threshold</div>
        <input type="number" class="ai-count-input" id="optThreshold" value="0.85" min="0" max="1" step="0.05"></div>
      <button class="btn btn-sm btn-gold" onclick="saveOptConfig()">Save</button>
    </div>
  </div>
  <div class="disabled-msg" id="disabledNotice">Add at least one golden to enable evaluation.</div>
  <div class="status" id="statusBar"></div>
  <div class="log-box" id="logOutput"></div>
</div>

<!-- Goldens -->
<div class="card">
  <div class="goldens-header">
    <div class="goldens-header-left">
      <div class="card-title" style="margin:0;">Goldens</div>
      <span class="goldens-count" id="goldensCount">(0)</span>
    </div>
    <div style="display:flex;gap:8px;">
      <button class="btn btn-sm btn-accent" onclick="toggleAiSection()"><img src="https://img.icons8.com/?size=100&id=rYb1JFR9WLSh&format=png&color=000000" style="width:14px;height:14px;filter:invert(1);vertical-align:middle;margin-right:4px;">Generate</button>
      <button class="btn btn-sm" onclick="addGolden()">+ Add</button>
    </div>
  </div>

  <div class="ai-section" id="aiSection">
    <div class="ai-label">What should the goldens test?</div>
    <textarea class="golden-textarea" id="aiDescription" placeholder="e.g. Customer support questions about refunds, shipping, billing..."></textarea>
    <div class="ai-row">
      <span style="font-size:12px;color:var(--text-muted);">Count:</span>
      <input type="number" class="ai-count-input" id="aiCount" value="5" min="1" max="20">
      <button class="btn btn-sm btn-gold" id="btnGenerate" onclick="generateGoldens()">Generate</button>
      <button class="btn btn-sm" onclick="toggleAiSection()">Cancel</button>
    </div>
    <div class="ai-status" id="aiStatus"><span class="spinner"></span>Generating...</div>
  </div>

  <div id="goldensContainer"></div>
  <div class="goldens-empty" id="goldensEmpty">No goldens yet. Add manually or generate with AI.</div>
  <button class="btn btn-sm btn-gold" id="btnSaveGoldens" onclick="saveGoldens()" style="display:none;margin-top:12px;">Save Goldens</button>
</div>

<!-- Tool Goldens -->
<div class="card">
  <div class="goldens-header">
    <div class="goldens-header-left">
      <div class="card-title" style="margin:0;">Tool Test Cases</div>
      <span class="goldens-count" id="toolGoldensCount">(0)</span>
    </div>
    <div><button class="btn btn-sm" onclick="addToolGolden()">+ Add</button></div>
  </div>
  <div id="toolGoldensContainer"></div>
  <div class="goldens-empty" id="toolGoldensEmpty">No tool test cases yet.</div>
  <button class="btn btn-sm btn-gold" id="btnSaveToolGoldens" onclick="saveToolGoldens()" style="display:none;margin-top:12px;">Save Tool Tests</button>
</div>

<!-- Conversation Goldens -->
<div class="card">
  <div class="goldens-header">
    <div class="goldens-header-left">
      <div class="card-title" style="margin:0;">Conversation Scenarios</div>
      <span class="goldens-count" id="convGoldensCount">(0)</span>
    </div>
    <div style="display:flex;gap:8px;">
      <button class="btn btn-sm btn-accent" onclick="toggleConvAiSection()"><img src="https://img.icons8.com/?size=100&id=rYb1JFR9WLSh&format=png&color=000000" style="width:14px;height:14px;filter:invert(1);vertical-align:middle;margin-right:4px;">Generate</button>
      <button class="btn btn-sm btn-purple" onclick="toggleTranscriptSection()">From Transcript</button>
      <button class="btn btn-sm" onclick="addConvGolden()">+ Add Scenario</button>
    </div>
  </div>

  <div class="ai-section" id="convAiSection">
    <div class="ai-label">Describe the conversation scenarios to generate</div>
    <textarea class="golden-textarea" id="convAiDescription" placeholder="e.g. User calls about tax filing, gets qualified on income, receives a plan recommendation. Include objection handling and early hangup scenarios..."></textarea>
    <div style="margin-top:10px;">
      <div class="golden-label">Eval Criteria for generated scenarios</div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:10px;">
        <label style="font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:4px;"><input type="checkbox" id="convGenFlow" checked> Flow Correctness</label>
        <label style="font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:4px;"><input type="checkbox" id="convGenLang" checked> Language</label>
        <label style="font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:4px;"><input type="checkbox" id="convGenEdge" checked> Edge Cases</label>
      </div>
    </div>
    <div class="ai-row">
      <span style="font-size:12px;color:var(--text-muted);">Count:</span>
      <input type="number" class="ai-count-input" id="convAiCount" value="3" min="1" max="10">
      <button class="btn btn-sm btn-gold" id="btnGenerateConv" onclick="generateConvGoldens()">Generate</button>
      <button class="btn btn-sm" onclick="toggleConvAiSection()">Cancel</button>
    </div>
    <div class="ai-status" id="convAiStatus"><span class="spinner"></span>Generating scenarios...</div>
  </div>

  <div class="ai-section" id="transcriptSection" style="border-color:var(--purple);">
    <div class="ai-label" style="color:var(--purple);">Paste a conversation transcript</div>
    <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;">Paste a real conversation (agent/user messages) and we'll extract it into a testable scenario. Format: one message per line, prefixed with "User:" or "Agent:" (or similar role labels).</div>
    <textarea class="golden-textarea" id="transcriptInput" style="min-height:140px;" placeholder="User: Hello?
Agent: Namaste! Main Priya bol rahi hoon ClearTax se...
User: Haan boliye
Agent: Aapne recently ClearTax pe apni details dekhi thi...
User: Haan maine dekha tha
Agent: Great! Aapki salary kitni hai annually?
User: 12 lakh hai
Agent: Perfect, toh aapke liye humara premium plan best rahega..."></textarea>
    <div style="margin-top:10px;">
      <div class="golden-label">Scenario name (optional)</div>
      <input class="golden-input" id="transcriptScenario" placeholder="e.g. Happy path - tax filing inquiry" style="margin-bottom:10px;">
    </div>
    <div class="ai-row">
      <button class="btn btn-sm btn-gold" id="btnTranscript" onclick="convertTranscript()">Convert to Scenario</button>
      <button class="btn btn-sm btn-purple" id="btnTranscriptAi" onclick="convertTranscriptAi()">AI-Enhanced Convert</button>
      <button class="btn btn-sm" onclick="toggleTranscriptSection()">Cancel</button>
    </div>
    <div class="ai-status" id="transcriptAiStatus"><span class="spinner"></span>Processing transcript...</div>
  </div>

  <div id="convGoldensContainer"></div>
  <div class="goldens-empty" id="convGoldensEmpty">No conversation scenarios yet. Add a scenario to test multi-turn flows.</div>
  <button class="btn btn-sm btn-gold" id="btnSaveConvGoldens" onclick="saveConvGoldens()" style="display:none;margin-top:12px;">Save Scenarios</button>
</div>

<hr class="divider">
<div class="section-label">Run History</div>
<div id="app"></div>

</div>

<script>
let goldens = [];
let toolGoldens = [];
let convGoldens = [];
let running = false;

fetch('/api/goldens').then(r => r.json()).then(data => { goldens = data; renderGoldens(); }).catch(() => renderGoldens());
fetch('/api/tool_goldens').then(r => r.json()).then(data => { toolGoldens = data; renderToolGoldens(); }).catch(() => renderToolGoldens());
fetch('/api/conv_goldens').then(r => r.json()).then(data => { convGoldens = data; renderConvGoldens(); }).catch(() => renderConvGoldens());
fetch('/api/optimizer_config').then(r => r.json()).then(data => {
  document.getElementById('optAlgo').value = data.algorithm || 'GEPA';
  document.getElementById('optIter').value = data.iterations || 10;
  document.getElementById('optThreshold').value = data.threshold || 0.85;
  const metrics = data.metrics || [data.metric || 'AnswerRelevancy'];
  document.getElementById('optMetricRelevancy').checked = metrics.includes('AnswerRelevancy');
  document.getElementById('optMetricLanguage').checked = metrics.includes('LanguageCompliance');
  document.getElementById('optMetricCorrectness').checked = metrics.includes('Correctness');
  document.getElementById('optMetricHallucination').checked = metrics.includes('Hallucination');
  document.getElementById('optMetricHelpfulness').checked = metrics.includes('Helpfulness');
}).catch(() => {});

function updateButtonState() {
  const hasGoldens = goldens.length > 0;
  const hasToolGoldens = toolGoldens.length > 0;
  const hasConvGoldens = convGoldens.length > 0;
  const hasEnoughConv = convGoldens.length >= 2;
  document.getElementById('btnTest').disabled = !hasGoldens || running;
  document.getElementById('btnTools').disabled = !hasToolGoldens || running;
  document.getElementById('btnConv').disabled = !hasConvGoldens || running;
  document.getElementById('btnOptimize').disabled = !hasGoldens || running;
  document.getElementById('btnOptConv').disabled = !hasEnoughConv || running;
  document.getElementById('disabledNotice').style.display = hasGoldens ? 'none' : 'block';
}

function renderGoldens() {
  const container = document.getElementById('goldensContainer');
  const empty = document.getElementById('goldensEmpty');
  document.getElementById('goldensCount').textContent = '(' + goldens.length + ')';
  if (!goldens.length) { container.innerHTML = ''; empty.style.display = 'block'; document.getElementById('btnSaveGoldens').style.display = 'none'; updateButtonState(); return; }
  empty.style.display = 'none';
  document.getElementById('btnSaveGoldens').style.display = 'inline-block';
  container.innerHTML = goldens.map((g, i) => {
    const cls = g._aiGenerated ? ' ai-generated' : '';
    return `<div class="golden-item${cls}">
      <button class="golden-remove" onclick="removeGolden(${i})">&times;</button>
      <div class="golden-field"><div class="golden-label">Input</div>
      <div class="field-wrap"><input class="golden-input" value="${escAttr(g.input || '')}" onchange="updateGolden(${i},'input',this.value)" placeholder="User question..."><button class="expand-btn" onclick="openModal(this.previousElementSibling)">&#x26F6;</button></div></div>
      <div class="golden-field"><div class="golden-label">Expected Output</div>
      <div class="field-wrap"><textarea class="golden-textarea" onchange="updateGolden(${i},'expected_output',this.value)" placeholder="Ideal response...">${escHtml(g.expected_output || '')}</textarea><button class="expand-btn" onclick="openModal(this.previousElementSibling)">&#x26F6;</button></div></div>
      <div class="golden-field"><div class="golden-label">Context</div>
      <div class="field-wrap"><textarea class="golden-textarea" onchange="updateGolden(${i},'context',this.value)" placeholder="Background facts...">${escHtml((g.context || []).join('\\n'))}</textarea><button class="expand-btn" onclick="openModal(this.previousElementSibling)">&#x26F6;</button></div></div>
    </div>`;
  }).join('');
  updateButtonState();
}

function addGolden() {
  goldens.push({ input: '', expected_output: '', context: [] });
  renderGoldens();
  const items = document.querySelectorAll('#goldensContainer .golden-item');
  const last = items[items.length - 1];
  last.scrollIntoView({ behavior: 'smooth', block: 'center' });
  setTimeout(() => { last.querySelector('.golden-input').focus(); }, 300);
}
function removeGolden(i) { goldens.splice(i, 1); renderGoldens(); }
function updateGolden(i, field, value) {
  if (field === 'context') { goldens[i].context = value.trim() ? value.split('\\n').filter(s => s.trim()) : []; }
  else { goldens[i][field] = value; }
  delete goldens[i]._aiGenerated;
}
function saveGoldens() {
  const clean = goldens.filter(g => g.input.trim()).map(g => ({ input: g.input, expected_output: g.expected_output, context: g.context || [] }));
  fetch('/api/goldens', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(clean) })
  .then(r => r.json()).then(data => { if (data.success) { goldens = clean; renderGoldens(); showQuickStatus('Goldens saved!'); } });
}

function showQuickStatus(msg) {
  const s = document.getElementById('statusBar');
  s.className = 'status visible status-done';
  s.innerHTML = msg;
  setTimeout(() => { s.className = 'status'; }, 3000);
}

// Tool Goldens
function renderToolGoldens() {
  const container = document.getElementById('toolGoldensContainer');
  const empty = document.getElementById('toolGoldensEmpty');
  document.getElementById('toolGoldensCount').textContent = '(' + toolGoldens.length + ')';
  if (!toolGoldens.length) { container.innerHTML = ''; empty.style.display = 'block'; document.getElementById('btnSaveToolGoldens').style.display = 'none'; updateButtonState(); return; }
  empty.style.display = 'none';
  document.getElementById('btnSaveToolGoldens').style.display = 'inline-block';
  container.innerHTML = toolGoldens.map((g, i) => {
    return `<div class="golden-item">
      <button class="golden-remove" onclick="removeToolGolden(${i})">&times;</button>
      <div class="golden-field"><div class="golden-label">Input</div>
      <div class="field-wrap"><input class="golden-input" value="${escAttr(g.input || '')}" onchange="updateToolGolden(${i},'input',this.value)" placeholder="User message..."><button class="expand-btn" onclick="openModal(this.previousElementSibling)">&#x26F6;</button></div></div>
      <div class="golden-field"><div class="golden-label">Actual Output</div>
      <div class="field-wrap"><textarea class="golden-textarea" onchange="updateToolGolden(${i},'actual_output',this.value)" placeholder="Agent response...">${escHtml(g.actual_output || '')}</textarea><button class="expand-btn" onclick="openModal(this.previousElementSibling)">&#x26F6;</button></div></div>
      <div class="golden-field"><div class="golden-label">Tools Called (JSON)</div>
      <div class="field-wrap"><textarea class="golden-textarea" onchange="updateToolGolden(${i},'tools_called',this.value)">${escHtml(JSON.stringify(g.tools_called || [], null, 2))}</textarea><button class="expand-btn" onclick="openModal(this.previousElementSibling)">&#x26F6;</button></div></div>
      <div class="golden-field"><div class="golden-label">Expected Tools (JSON)</div>
      <div class="field-wrap"><textarea class="golden-textarea" onchange="updateToolGolden(${i},'expected_tools',this.value)">${escHtml(JSON.stringify(g.expected_tools || [], null, 2))}</textarea><button class="expand-btn" onclick="openModal(this.previousElementSibling)">&#x26F6;</button></div></div>
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
function removeToolGolden(i) { toolGoldens.splice(i, 1); renderToolGoldens(); }
function updateToolGolden(i, field, value) {
  if (field === 'tools_called' || field === 'expected_tools') { try { toolGoldens[i][field] = JSON.parse(value); } catch(e) {} }
  else { toolGoldens[i][field] = value; }
}
function saveToolGoldens() {
  const clean = toolGoldens.filter(g => g.input.trim()).map(g => ({ input: g.input, actual_output: g.actual_output, tools_called: g.tools_called || [], expected_tools: g.expected_tools || [] }));
  fetch('/api/tool_goldens', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(clean) })
  .then(r => r.json()).then(data => { if (data.success) { toolGoldens = clean; renderToolGoldens(); showQuickStatus('Tool tests saved!'); } });
}

// Conversation Goldens
function renderConvGoldens() {
  const container = document.getElementById('convGoldensContainer');
  const empty = document.getElementById('convGoldensEmpty');
  document.getElementById('convGoldensCount').textContent = '(' + convGoldens.length + ')';
  if (!convGoldens.length) { container.innerHTML = ''; empty.style.display = 'block'; document.getElementById('btnSaveConvGoldens').style.display = 'none'; updateButtonState(); return; }
  empty.style.display = 'none';
  document.getElementById('btnSaveConvGoldens').style.display = 'inline-block';
  container.innerHTML = convGoldens.map((c, i) => {
    const turnsHtml = (c.turns || []).map((t, ti) => {
      return `<div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;">
        <span style="font-size:11px;color:var(--text-muted);min-width:20px;">${ti+1}.</span>
        <input class="golden-input" value="${escAttr(t.content || '')}" onchange="updateConvTurn(${i},${ti},this.value)" placeholder="User message..." style="flex:1;">
        <button class="golden-remove" style="position:static;width:20px;height:20px;font-size:11px;" onclick="removeConvTurn(${i},${ti})">&times;</button>
      </div>`;
    }).join('');
    const cohort = c.cohort || 'inactive';
    const templateVars = c.template_vars || {};
    const tvDisplay = Object.entries(templateVars).map(([k,v]) => k + ': ' + v).join(', ');
    return `<div class="golden-item">
      <button class="golden-remove" onclick="removeConvGolden(${i})">&times;</button>
      <div class="golden-field"><div class="golden-label">Scenario Name</div>
      <input class="golden-input" value="${escAttr(c.scenario || '')}" onchange="convGoldens[${i}].scenario=this.value" placeholder="e.g. User objects mid-pitch"></div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <div class="golden-field" style="flex:1;min-width:120px;"><div class="golden-label">Cohort</div>
        <select class="golden-input" onchange="convGoldens[${i}].cohort=this.value" style="cursor:pointer;">
          <option value="inactive" ${cohort==='inactive'?'selected':''}>inactive</option>
          <option value="performance_drop" ${cohort==='performance_drop'?'selected':''}>performance_drop</option>
          <option value="d1_d2" ${cohort==='d1_d2'?'selected':''}>d1_d2</option>
          <option value="nudge" ${cohort==='nudge'?'selected':''}>nudge</option>
          <option value="campaign" ${cohort==='campaign'?'selected':''}>campaign</option>
          <option value="warning" ${cohort==='warning'?'selected':''}>warning</option>
        </select></div>
        <div class="golden-field" style="flex:2;min-width:200px;"><div class="golden-label">Template Variables <span style="font-weight:300;opacity:0.6">(JSON)</span></div>
        <input class="golden-input" value="${escAttr(JSON.stringify(templateVars))}" onchange="try{convGoldens[${i}].template_vars=JSON.parse(this.value)}catch(e){}" placeholder='{"participantName":"...", "cohort":"..."}'></div>
      </div>
      <div class="golden-field"><div class="golden-label">Eval Criteria</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <label style="font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:4px;"><input type="checkbox" ${c.eval_criteria?.includes('flow_correctness')?'checked':''} onchange="toggleConvCriteria(${i},'flow_correctness',this.checked)"> Flow</label>
        <label style="font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:4px;"><input type="checkbox" ${c.eval_criteria?.includes('language')?'checked':''} onchange="toggleConvCriteria(${i},'language',this.checked)"> Language</label>
        <label style="font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:4px;"><input type="checkbox" ${c.eval_criteria?.includes('edge_case')?'checked':''} onchange="toggleConvCriteria(${i},'edge_case',this.checked)"> Edge Cases</label>
      </div></div>
      <div class="golden-field"><div class="golden-label">User Turns (sequential messages)</div>
      ${turnsHtml}
      <button class="btn btn-sm" onclick="addConvTurn(${i})" style="margin-top:4px;">+ Add Turn</button>
      </div>
    </div>`;
  }).join('');
  updateButtonState();
}

function addConvGolden() {
  convGoldens.push({ scenario: '', cohort: 'inactive', template_vars: {}, eval_criteria: ['flow_correctness', 'language', 'edge_case'], turns: [{ role: 'user', content: '' }] });
  renderConvGoldens();
  const items = document.querySelectorAll('#convGoldensContainer .golden-item');
  const last = items[items.length - 1];
  last.scrollIntoView({ behavior: 'smooth', block: 'center' });
}
function removeConvGolden(i) { convGoldens.splice(i, 1); renderConvGoldens(); }
function addConvTurn(i) { convGoldens[i].turns.push({ role: 'user', content: '' }); renderConvGoldens(); }
function removeConvTurn(i, ti) { convGoldens[i].turns.splice(ti, 1); renderConvGoldens(); }
function updateConvTurn(i, ti, value) { convGoldens[i].turns[ti].content = value; }
function toggleConvCriteria(i, criterion, checked) {
  if (!convGoldens[i].eval_criteria) convGoldens[i].eval_criteria = [];
  if (checked && !convGoldens[i].eval_criteria.includes(criterion)) convGoldens[i].eval_criteria.push(criterion);
  if (!checked) convGoldens[i].eval_criteria = convGoldens[i].eval_criteria.filter(c => c !== criterion);
}
function saveConvGoldens() {
  const clean = convGoldens.filter(c => c.scenario && c.turns.some(t => t.content.trim())).map(c => ({
    scenario: c.scenario,
    cohort: c.cohort || 'inactive',
    template_vars: c.template_vars || {},
    eval_criteria: c.eval_criteria || [],
    turns: c.turns.filter(t => t.content.trim()).map(t => ({ role: 'user', content: t.content }))
  }));
  fetch('/api/conv_goldens', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(clean) })
  .then(r => r.json()).then(data => { if (data.success) { convGoldens = clean; renderConvGoldens(); showQuickStatus('Conversation scenarios saved!'); } });
}

// Generate Conversation Goldens
function toggleConvAiSection() { document.getElementById('convAiSection').classList.toggle('visible'); document.getElementById('transcriptSection').classList.remove('visible'); }
function toggleTranscriptSection() { document.getElementById('transcriptSection').classList.toggle('visible'); document.getElementById('convAiSection').classList.remove('visible'); }

function generateConvGoldens() {
  const desc = document.getElementById('convAiDescription').value.trim();
  if (!desc) { document.getElementById('convAiDescription').focus(); return; }
  const count = parseInt(document.getElementById('convAiCount').value) || 3;
  const prompt = document.getElementById('promptInput').value;
  const criteria = [];
  if (document.getElementById('convGenFlow').checked) criteria.push('flow_correctness');
  if (document.getElementById('convGenLang').checked) criteria.push('language');
  if (document.getElementById('convGenEdge').checked) criteria.push('edge_case');

  document.getElementById('btnGenerateConv').disabled = true;
  document.getElementById('convAiStatus').classList.add('visible');
  fetch('/api/generate_conv_goldens', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ description: desc, count: count, system_prompt: prompt, eval_criteria: criteria }) })
  .then(r => r.json()).then(data => {
    document.getElementById('btnGenerateConv').disabled = false;
    document.getElementById('convAiStatus').classList.remove('visible');
    if (data.success && data.scenarios) {
      convGoldens = convGoldens.concat(data.scenarios);
      renderConvGoldens();
      document.getElementById('convAiSection').classList.remove('visible');
      document.getElementById('convAiDescription').value = '';
      showQuickStatus('Generated ' + data.scenarios.length + ' scenarios!');
    } else { alert('Error: ' + (data.message || 'Unknown')); }
  }).catch(err => {
    document.getElementById('btnGenerateConv').disabled = false;
    document.getElementById('convAiStatus').classList.remove('visible');
    alert('Error: ' + err.message);
  });
}

function convertTranscript() {
  const raw = document.getElementById('transcriptInput').value.trim();
  if (!raw) { document.getElementById('transcriptInput').focus(); return; }
  const scenarioName = document.getElementById('transcriptScenario').value.trim() || 'Transcript scenario';

  const lines = raw.split('\\n').filter(l => l.trim());
  const turns = [];
  for (const line of lines) {
    const match = line.match(/^(user|customer|caller|human)\\s*[:>-]\\s*(.+)/i);
    if (match) {
      turns.push({ role: 'user', content: match[2].trim() });
    }
  }

  if (!turns.length) {
    alert('Could not find any user messages. Make sure lines start with "User:", "Customer:", "Caller:", or "Human:"');
    return;
  }

  convGoldens.push({ scenario: scenarioName, eval_criteria: ['flow_correctness', 'language', 'edge_case'], turns: turns });
  renderConvGoldens();
  document.getElementById('transcriptSection').classList.remove('visible');
  document.getElementById('transcriptInput').value = '';
  document.getElementById('transcriptScenario').value = '';
  showQuickStatus('Extracted ' + turns.length + ' turns from transcript!');
}

function convertTranscriptAi() {
  const raw = document.getElementById('transcriptInput').value.trim();
  if (!raw) { document.getElementById('transcriptInput').focus(); return; }
  const scenarioName = document.getElementById('transcriptScenario').value.trim() || '';
  const prompt = document.getElementById('promptInput').value;

  document.getElementById('btnTranscriptAi').disabled = true;
  document.getElementById('transcriptAiStatus').classList.add('visible');
  fetch('/api/transcript_to_golden', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ transcript: raw, scenario_name: scenarioName, system_prompt: prompt }) })
  .then(r => r.json()).then(data => {
    document.getElementById('btnTranscriptAi').disabled = false;
    document.getElementById('transcriptAiStatus').classList.remove('visible');
    if (data.success && data.scenario) {
      convGoldens.push(data.scenario);
      renderConvGoldens();
      document.getElementById('transcriptSection').classList.remove('visible');
      document.getElementById('transcriptInput').value = '';
      document.getElementById('transcriptScenario').value = '';
      showQuickStatus('AI converted transcript to scenario!');
    } else { alert('Error: ' + (data.message || 'Unknown')); }
  }).catch(err => {
    document.getElementById('btnTranscriptAi').disabled = false;
    document.getElementById('transcriptAiStatus').classList.remove('visible');
    alert('Error: ' + err.message);
  });
}

// Optimizer Config
function saveOptConfig() {
  const metrics = [];
  if (document.getElementById('optMetricRelevancy').checked) metrics.push('AnswerRelevancy');
  if (document.getElementById('optMetricLanguage').checked) metrics.push('LanguageCompliance');
  if (document.getElementById('optMetricCorrectness').checked) metrics.push('Correctness');
  if (document.getElementById('optMetricHallucination').checked) metrics.push('Hallucination');
  if (document.getElementById('optMetricHelpfulness').checked) metrics.push('Helpfulness');
  const data = {
    algorithm: document.getElementById('optAlgo').value,
    iterations: parseInt(document.getElementById('optIter').value) || 10,
    metrics: metrics.length ? metrics : ['AnswerRelevancy'],
    threshold: parseFloat(document.getElementById('optThreshold').value) || 0.85
  };
  fetch('/api/optimizer_config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) })
  .then(r => r.json()).then(d => { if (d.success) showQuickStatus('Optimizer settings saved!'); });
}

// AI Generation
function toggleAiSection() { document.getElementById('aiSection').classList.toggle('visible'); }
function generateGoldens() {
  const desc = document.getElementById('aiDescription').value.trim();
  if (!desc) { document.getElementById('aiDescription').focus(); return; }
  const count = parseInt(document.getElementById('aiCount').value) || 5;
  const prompt = document.getElementById('promptInput').value;
  document.getElementById('btnGenerate').disabled = true;
  document.getElementById('aiStatus').classList.add('visible');
  fetch('/api/generate_goldens', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ description: desc, count: count, system_prompt: prompt }) })
  .then(r => r.json()).then(data => {
    document.getElementById('btnGenerate').disabled = false;
    document.getElementById('aiStatus').classList.remove('visible');
    if (data.success && data.goldens) {
      data.goldens.forEach(g => { g._aiGenerated = true; });
      goldens = goldens.concat(data.goldens);
      renderGoldens();
      document.getElementById('aiSection').classList.remove('visible');
      document.getElementById('aiDescription').value = '';
      showQuickStatus('Generated ' + data.goldens.length + ' goldens!');
    } else { alert('Error: ' + (data.message || 'Unknown')); }
  }).catch(err => {
    document.getElementById('btnGenerate').disabled = false;
    document.getElementById('aiStatus').classList.remove('visible');
    alert('Error: ' + err.message);
  });
}

// Run Actions
function runAction(action) {
  if (running) return;
  if (action === 'test_tools') { if (!toolGoldens.filter(g => g.input.trim()).length) return; }
  else if (action === 'test_conversation' || action === 'optimize_conversation') { if (convGoldens.length < (action === 'optimize_conversation' ? 2 : 1)) return; }
  else { if (!goldens.filter(g => g.input.trim()).length) return; }

  running = true;
  updateButtonState();
  const prompt = document.getElementById('promptInput').value;
  const status = document.getElementById('statusBar');
  status.className = 'status visible status-running';
  const logEl = document.getElementById('logOutput');
  logEl.textContent = '';
  logEl.classList.add('visible');
  const startTime = Date.now();

  const labels = { test_prompts: 'Evaluating prompts', test_tools: 'Evaluating tools', test_conversation: 'Testing conversation flow', optimize: 'Optimizing prompt', optimize_conversation: 'Optimizing for conversation flow' };
  status.innerHTML = '<span class="spinner"></span>' + (labels[action] || 'Running') + '...';

  const timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
    status.innerHTML = '<span class="spinner"></span>' + (labels[action] || 'Running') + '... ' + elapsed + 's';
  }, 1000);

  const saves = [
    fetch('/api/goldens', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(goldens.filter(g => g.input.trim()).map(g => ({ input: g.input, expected_output: g.expected_output, context: g.context || [] }))) }),
    fetch('/api/tool_goldens', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(toolGoldens.filter(g => g.input.trim()).map(g => ({ input: g.input, actual_output: g.actual_output, tools_called: g.tools_called || [], expected_tools: g.expected_tools || [] }))) })
  ];

  Promise.all(saves).then(() => fetch('/api/run', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action, prompt}) }))
  .then(r => r.json()).then(() => {
    const poll = setInterval(() => {
      fetch('/api/log').then(r => r.json()).then(data => {
        if (data.log) { logEl.textContent = data.log; logEl.scrollTop = logEl.scrollHeight; }
        if (data.done) {
          clearInterval(poll); clearInterval(timerInterval);
          const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
          running = false; updateButtonState();
          if (data.success) { status.className = 'status visible status-done'; status.innerHTML = '&#10003; ' + data.message + ' <span style="opacity:0.7;margin-left:8px;">' + elapsed + 's</span>'; }
          else { status.className = 'status visible status-error'; status.innerHTML = '&#10007; ' + data.message + ' <span style="opacity:0.7;margin-left:8px;">' + elapsed + 's</span>'; }
          loadResults();
        }
      });
    }, 500);
  }).catch(err => {
    clearInterval(timerInterval); running = false; updateButtonState();
    status.className = 'status visible status-error'; status.innerHTML = 'Error: ' + err.message;
  });
}

// Results
let _allRuns = [];
function loadResults() { fetch('/api/results').then(r => r.json()).then(data => { _allRuns = data.reverse(); render(_allRuns); }).catch(() => { _allRuns = []; render([]); }); }
loadResults();

function render(runs) {
  const app = document.getElementById('app');
  if (!runs.length) { app.innerHTML = '<div class="empty-state">No runs yet. Add goldens and run an evaluation.</div>'; return; }
  app.innerHTML = runs.map(renderRun).join('');
  document.querySelectorAll('.run-header').forEach(h => h.addEventListener('click', () => h.parentElement.classList.toggle('open')));
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
    const g = t.closest('.run-details');
    g.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    g.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    g.querySelector('.tab-content[data-tab="' + t.dataset.tab + '"]').classList.add('active');
  }));
}

function renderRun(run) {
  const name = (run.system_prompt || 'No system prompt').substring(0, 50);
  const date = new Date(run.timestamp).toLocaleString();
  const duration = run.duration_seconds ? run.duration_seconds + 's' : '';
  let badge = '';
  if (run.type === 'optimization') badge = '<span class="badge badge-opt">Optimized</span>';
  else if (run.type === 'conversation_evaluation') badge = '<span class="badge badge-tools">Conversation</span>';
  else if (run.type === 'tool_evaluation') badge = '<span class="badge badge-tools">Tools</span>';
  else if (run.tests && run.tests.every(t => t.passed)) badge = '<span class="badge badge-pass">Passed</span>';
  else { const f = run.tests ? run.tests.filter(t => !t.passed).length : 0; badge = '<span class="badge badge-fail">' + f + ' Failed</span>'; }

  let details = '';
  const hasPromptTests = run.tests && run.tests.length > 0;
  const hasToolTests = run.tool_tests && run.tool_tests.length > 0;
  const allTests = (run.tests || []).concat(run.tool_tests || []);

  if (allTests.length) {
    const passed = allTests.filter(t => t.passed).length;
    const failed = allTests.filter(t => !t.passed).length;
    const avg = (allTests.reduce((s, t) => s + (t.score || 0), 0) / allTests.length).toFixed(2);
    details += '<div class="summary-bar"><span class="summary-item summary-pass">&#10003; ' + passed + ' passed</span>';
    if (failed) details += '<span class="summary-item summary-fail">&#10007; ' + failed + ' failed</span>';
    details += '<span class="summary-item summary-total">' + allTests.length + ' total</span><span class="summary-item summary-score">avg ' + avg + '</span>';
    if (duration) details += '<span class="summary-item summary-time">' + duration + '</span>';
    details += '<button class="btn btn-sm btn-accent" style="margin-left:auto;" onclick="openResultsModal(' + _allRuns.indexOf(run) + ')">View Results</button>';
    details += '</div>';
  }

  details += '<div class="detail-section"><div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;"><div class="detail-title" style="margin:0;">Prompt</div><button class="btn btn-sm" onclick="copyFromSibling(this)">Copy</button></div><div class="prompt-box">' + escHtml(run.system_prompt || '(empty)') + '</div></div>';
  if (run.optimized_prompt) {
    details += '<div class="detail-section"><div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;"><div class="detail-title" style="margin:0;">Optimized Prompt</div><button class="btn btn-sm btn-accent" onclick="openDiffModal(' + _allRuns.indexOf(run) + ')">View Diff</button><button class="btn btn-sm" onclick="copyFromSibling(this)">Copy</button></div><div class="prompt-box">' + escHtml(run.optimized_prompt) + '</div></div>';
  }

  const hasTabs = hasPromptTests && hasToolTests;
  if (hasTabs) details += '<div class="tab-row"><div class="tab active" data-tab="prompt">Prompt Tests</div><div class="tab" data-tab="tools">Tool Tests</div></div>';

  if (hasPromptTests) {
    details += '<div class="tab-content' + (hasTabs ? ' active' : '') + '" data-tab="prompt"><div class="detail-section"><div class="detail-title">Results</div>' + renderTestTable(run.tests) + '</div>';
    const fails = run.tests.filter(t => !t.passed);
    if (fails.length) { details += '<div class="detail-section"><div class="detail-title">Improvements</div>'; fails.forEach(f => { details += '<div class="improvement"><strong>' + escHtml(f.metric) + '</strong> on "' + escHtml(f.input.substring(0,40)) + '"<br>' + escHtml(f.reason || '') + '</div>'; }); details += '</div>'; }
    details += '</div>';
  }
  if (hasToolTests) {
    details += '<div class="tab-content' + (hasTabs ? '' : ' active') + '" data-tab="tools"><div class="detail-section"><div class="detail-title">Results</div>' + renderTestTable(run.tool_tests) + '</div>';
    const fails = run.tool_tests.filter(t => !t.passed);
    if (fails.length) { details += '<div class="detail-section"><div class="detail-title">Improvements</div>'; fails.forEach(f => { details += '<div class="improvement"><strong>' + escHtml(f.metric) + '</strong> on "' + escHtml(f.input.substring(0,40)) + '"<br>' + escHtml(f.reason || '') + '</div>'; }); details += '</div>'; }
    details += '</div>';
  }

  const dur = duration ? '<span class="run-duration">' + duration + '</span>' : '';
  return '<div class="run-row"><div class="run-header"><span class="arrow">&#9654;</span><span class="run-name">' + escHtml(name) + '</span>' + badge + dur + '<span class="run-date">' + date + '</span></div><div class="run-details">' + details + '</div></div>';
}

function renderTestTable(tests) {
  let h = '<table class="test-table"><thead><tr><th>Input</th><th>Metric</th><th>Score</th><th>Status</th></tr></thead><tbody>';
  tests.forEach(t => {
    const c = t.passed ? 'score-pass' : 'score-fail';
    h += '<tr><td>' + escHtml(t.input.substring(0,50)) + '</td><td>' + escHtml(t.metric) + '</td><td class="score ' + c + '">' + (t.score != null ? t.score.toFixed(2) : '-') + '</td><td class="' + c + '">' + (t.passed ? '&#10003;' : '&#10007;') + '</td></tr>';
    if (t.reason) h += '<tr><td colspan="4" class="reason-text">' + escHtml(t.reason) + '</td></tr>';
  });
  return h + '</tbody></table>';
}

function copyFromSibling(btn) {
  const box = btn.closest('.detail-section').querySelector('.prompt-box');
  if (!box) return;
  navigator.clipboard.writeText(box.textContent).then(() => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    btn.style.borderColor = 'var(--success)';
    btn.style.color = 'var(--success)';
    setTimeout(() => { btn.textContent = orig; btn.style.borderColor = ''; btn.style.color = ''; }, 1500);
  });
}

function copyDiffText(btn, which, idx) {
  const run = _allRuns[idx];
  if (!run) return;
  const text = which === 'optimized' ? run.optimized_prompt : run.system_prompt;
  navigator.clipboard.writeText(text || '').then(() => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    btn.style.borderColor = 'var(--success)';
    btn.style.color = 'var(--success)';
    setTimeout(() => { btn.textContent = orig; btn.style.borderColor = ''; btn.style.color = ''; }, 1500);
  });
}

function escHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function escAttr(s) { return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// Diff Modal
function openDiffModal(idx) {
  const run = _allRuns[idx];
  if (!run || !run.optimized_prompt) return;
  const modal = document.getElementById('resultsModal');
  const box = modal.querySelector('.modal-box');
  const body = document.getElementById('resultsModalBody');
  const original = run.system_prompt || '';
  const optimized = run.optimized_prompt || '';

  const diff = computeDiff(original, optimized);

  let html = '<div style="margin-bottom:16px;display:flex;align-items:center;gap:12px;"><span style="font-size:13px;color:var(--text-muted);">Prompt Diff</span><span style="font-size:11px;color:var(--error);background:var(--error-dim);padding:2px 8px;border-radius:3px;">Removed</span><span style="font-size:11px;color:var(--success);background:var(--success-dim);padding:2px 8px;border-radius:3px;">Added</span></div>';

  // Side-by-side: Original left, Optimized right
  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;">';
  html += '<div class="detail-section"><div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;"><div style="font-size:11px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);font-weight:500;">Original</div><button class="btn btn-sm" onclick="copyDiffText(this,&quot;original&quot;,' + idx + ')">Copy</button></div><div class="prompt-box" style="max-height:40vh;min-height:200px;">' + diff.originalHtml + '</div></div>';
  html += '<div class="detail-section"><div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;"><div style="font-size:11px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);font-weight:500;">Optimized</div><button class="btn btn-sm btn-accent" onclick="copyDiffText(this,&quot;optimized&quot;,' + idx + ')">Copy</button></div><div class="prompt-box" style="max-height:40vh;min-height:200px;">' + diff.optimizedHtml + '</div></div>';
  html += '</div>';

  // Unified diff below
  html += '<div><div style="font-size:11px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text-muted);margin-bottom:8px;font-weight:500;">Unified Changes</div>';
  html += '<div class="prompt-box" style="max-height:35vh;">' + diff.unifiedHtml + '</div></div>';

  body.innerHTML = html;
  modal.querySelector('.modal-title').textContent = 'Prompt Diff';
  box.style.maxWidth = '95vw';
  box.style.height = '90vh';
  modal.classList.add('visible');
}
function closeResultsModal() {
  const modal = document.getElementById('resultsModal');
  modal.classList.remove('visible');
  const box = modal.querySelector('.modal-box');
  box.style.maxWidth = '';
  box.style.height = '';
}

function computeDiff(a, b) {
  const aLines = a.split('\\n');
  const bLines = b.split('\\n');

  // DP-based LCS for reliable diff
  const n = aLines.length, m = bLines.length;
  // For large files, use a row-optimized DP to find the edit script
  const prev = new Array(m + 1).fill(0);
  const curr = new Array(m + 1).fill(0);
  // Build LCS length table (space-optimized, but we need backtrack so use full matrix for <= 1000 lines)
  let dp;
  if (n <= 1000 && m <= 1000) {
    dp = Array.from({length: n + 1}, () => new Uint16Array(m + 1));
    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        if (aLines[i-1] === bLines[j-1]) dp[i][j] = dp[i-1][j-1] + 1;
        else dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
      }
    }
  }

  // Backtrack to get diff ops
  const ops = [];
  if (dp) {
    let i = n, j = m;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && aLines[i-1] === bLines[j-1]) {
        ops.push({ type: 'same', aLine: aLines[i-1], bLine: bLines[j-1] });
        i--; j--;
      } else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) {
        ops.push({ type: 'add', bLine: bLines[j-1] });
        j--;
      } else {
        ops.push({ type: 'remove', aLine: aLines[i-1] });
        i--;
      }
    }
    ops.reverse();
  } else {
    // Fallback for very large files: simple line-by-line
    const max = Math.max(n, m);
    for (let i = 0; i < max; i++) {
      if (i < n && i < m && aLines[i] === bLines[i]) ops.push({ type: 'same', aLine: aLines[i], bLine: bLines[i] });
      else { if (i < n) ops.push({ type: 'remove', aLine: aLines[i] }); if (i < m) ops.push({ type: 'add', bLine: bLines[i] }); }
    }
  }

  // VS Code style: side-by-side with aligned rows
  let leftHtml = '';
  let rightHtml = '';
  let unifiedHtml = '';
  let leftLine = 1, rightLine = 1;

  ops.forEach(op => {
    if (op.type === 'same') {
      leftHtml += '<div class="diff-line"><span class="diff-ln">' + leftLine + '</span><span class="diff-text">' + escHtml(op.aLine || ' ') + '</span></div>';
      rightHtml += '<div class="diff-line"><span class="diff-ln">' + rightLine + '</span><span class="diff-text">' + escHtml(op.bLine || ' ') + '</span></div>';
      unifiedHtml += '<div class="diff-line"><span class="diff-text" style="color:var(--text-muted);">  ' + escHtml(op.aLine) + '</span></div>';
      leftLine++; rightLine++;
    } else if (op.type === 'remove') {
      leftHtml += '<div class="diff-line diff-del"><span class="diff-ln">' + leftLine + '</span><span class="diff-text">' + escHtml(op.aLine || ' ') + '</span></div>';
      rightHtml += '<div class="diff-line diff-empty"><span class="diff-ln"></span><span class="diff-text"></span></div>';
      unifiedHtml += '<div class="diff-line diff-del"><span class="diff-text">- ' + escHtml(op.aLine) + '</span></div>';
      leftLine++;
    } else if (op.type === 'add') {
      leftHtml += '<div class="diff-line diff-empty"><span class="diff-ln"></span><span class="diff-text"></span></div>';
      rightHtml += '<div class="diff-line diff-add"><span class="diff-ln">' + rightLine + '</span><span class="diff-text">' + escHtml(op.bLine || ' ') + '</span></div>';
      unifiedHtml += '<div class="diff-line diff-add"><span class="diff-text">+ ' + escHtml(op.bLine) + '</span></div>';
      rightLine++;
    }
  });

  return { originalHtml: leftHtml, optimizedHtml: rightHtml, unifiedHtml };
}

// Results Modal
function renderConversationTranscript(t) {
  const inputs = (t.input || '').split('\\n').filter(l => l.trim());
  const outputs = (t.output || '').split('\\n').filter(l => l.trim());
  let html = '<div style="background:var(--bg-deep);border:1px solid var(--border-subtle);border-radius:8px;padding:14px;margin:8px 0;max-height:240px;overflow-y:auto;">';
  const maxTurns = Math.max(inputs.length, outputs.length);
  for (let i = 0; i < maxTurns; i++) {
    if (inputs[i]) html += '<div style="margin-bottom:6px;"><span style="font-size:10px;font-weight:600;color:var(--blue);text-transform:uppercase;letter-spacing:0.5px;">User</span><div style="font-size:12px;color:var(--text-secondary);padding:4px 0 4px 10px;border-left:2px solid var(--blue);">' + escHtml(inputs[i].replace(/^User:\\s*/i, '')) + '</div></div>';
    if (outputs[i]) html += '<div style="margin-bottom:10px;"><span style="font-size:10px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:0.5px;">Agent</span><div style="font-size:12px;color:var(--text-secondary);padding:4px 0 4px 10px;border-left:2px solid var(--accent);">' + escHtml(outputs[i].replace(/^Agent:\\s*/i, '')) + '</div></div>';
  }
  html += '</div>';
  return html;
}

function openResultsModal(idx) {
  const run = _allRuns[idx];
  if (!run) return;
  const modal = document.getElementById('resultsModal');
  const body = document.getElementById('resultsModalBody');
  const allTests = (run.tests || []).concat(run.tool_tests || []);
  const passed = allTests.filter(t => t.passed);
  const failed = allTests.filter(t => !t.passed);
  const isConversation = run.type === 'conversation_evaluation';

  let html = '';

  // Header summary
  html += '<div style="display:flex;gap:20px;align-items:center;margin-bottom:24px;flex-wrap:wrap;">';
  html += '<div style="font-size:28px;font-weight:700;color:var(--text-primary);">' + passed.length + '/' + allTests.length + '</div>';
  html += '<div style="font-size:13px;color:var(--text-muted);">tests passed</div>';
  const avg = allTests.length ? (allTests.reduce((s,t) => s + (t.score||0), 0) / allTests.length).toFixed(2) : '0';
  html += '<div style="margin-left:auto;font-family:var(--mono);font-size:14px;color:var(--accent);">avg score: ' + avg + '</div>';
  if (run.duration_seconds) html += '<div style="font-family:var(--mono);font-size:13px;color:var(--text-muted);">' + run.duration_seconds + 's</div>';
  html += '</div>';

  // Failed tests section
  if (failed.length) {
    html += '<div style="margin-bottom:28px;">';
    html += '<div style="font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--error);margin-bottom:12px;font-weight:600;">Failed Tests (' + failed.length + ')</div>';
    failed.forEach(t => {
      html += '<div style="background:var(--error-dim);border:1px solid rgba(239,68,68,0.2);border-radius:8px;padding:16px;margin-bottom:12px;">';
      html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">';
      html += '<span style="font-size:13px;font-weight:500;color:var(--text-primary);">' + escHtml(t.metric) + (t.scenario ? ' &mdash; ' + escHtml(t.scenario) : '') + '</span>';
      html += '<span class="score score-fail" style="font-size:14px;">' + (t.score != null ? t.score.toFixed(2) : '-') + '</span>';
      html += '</div>';
      if (isConversation && t.output) {
        html += renderConversationTranscript(t);
      } else {
        html += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:6px;"><strong>Input:</strong> ' + escHtml(t.input.substring(0, 200)) + '</div>';
      }
      if (t.reason) html += '<div style="font-size:11px;color:var(--text-muted);font-style:italic;margin-top:8px;padding-top:8px;border-top:1px solid var(--border-subtle);">' + escHtml(t.reason) + '</div>';
      html += '</div>';
    });
    html += '</div>';
  }

  // Passed tests section
  if (passed.length) {
    html += '<div style="margin-bottom:28px;">';
    html += '<div style="font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--success);margin-bottom:12px;font-weight:600;">Passed Tests (' + passed.length + ')</div>';
    passed.forEach(t => {
      html += '<div style="background:var(--success-dim);border:1px solid rgba(34,197,94,0.15);border-radius:8px;padding:16px;margin-bottom:12px;">';
      html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">';
      html += '<span style="font-size:13px;font-weight:500;color:var(--text-primary);">' + escHtml(t.metric) + (t.scenario ? ' &mdash; ' + escHtml(t.scenario) : '') + '</span>';
      html += '<span class="score score-pass" style="font-size:14px;">' + (t.score != null ? t.score.toFixed(2) : '-') + '</span>';
      html += '</div>';
      if (isConversation && t.output) {
        html += renderConversationTranscript(t);
      } else {
        html += '<div style="font-size:12px;color:var(--text-muted);"><strong>Input:</strong> ' + escHtml(t.input.substring(0, 200)) + '</div>';
      }
      if (t.reason) html += '<div style="font-size:11px;color:var(--text-muted);font-style:italic;margin-top:8px;padding-top:8px;border-top:1px solid var(--border-subtle);">' + escHtml(t.reason) + '</div>';
      html += '</div>';
    });
    html += '</div>';
  }

  // Improvements section
  if (failed.length) {
    html += '<div>';
    html += '<div style="font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--accent);margin-bottom:12px;font-weight:600;">Suggested Improvements</div>';
    failed.forEach(t => {
      html += '<div class="improvement" style="margin-bottom:10px;"><strong>' + escHtml(t.metric) + '</strong> on "' + escHtml((t.scenario || t.input).substring(0,60)) + '"<br><span style="color:var(--text-secondary);">' + escHtml(t.reason || 'No suggestion available') + '</span></div>';
    });
    html += '</div>';
  }

  body.innerHTML = html;
  modal.classList.add('visible');
}

// Modal
let modalTarget = null;
function openModal(el) {
  modalTarget = el;
  const modal = document.getElementById('modalOverlay');
  document.getElementById('modalTextarea').value = el.value !== undefined ? el.value : el.textContent;
  document.getElementById('modalTitle').textContent = el.closest('.golden-field')?.querySelector('.golden-label')?.textContent || 'Edit';
  modal.classList.add('visible');
  document.getElementById('modalTextarea').focus();
}
function closeModal() { document.getElementById('modalOverlay').classList.remove('visible'); modalTarget = null; }
function saveModal() {
  if (modalTarget) { modalTarget.value = document.getElementById('modalTextarea').value; modalTarget.dispatchEvent(new Event('change')); }
  closeModal();
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeModal(); closeResultsModal(); } });
document.getElementById('promptInput').addEventListener('dblclick', function() { openModal(this); });
</script>

<div class="modal-overlay" id="modalOverlay" onclick="if(event.target===this)closeModal()">
  <div class="modal-box">
    <div class="modal-header">
      <span class="modal-title" id="modalTitle">Edit</span>
      <button class="modal-close" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body"><textarea class="modal-textarea" id="modalTextarea"></textarea></div>
    <div class="modal-footer">
      <button class="btn btn-sm" onclick="closeModal()">Cancel</button>
      <button class="btn btn-sm btn-gold" onclick="saveModal()">Save</button>
    </div>
  </div>
</div>

<div class="modal-overlay" id="resultsModal" onclick="if(event.target===this)closeResultsModal()">
  <div class="modal-box" style="max-width:960px;height:85vh;">
    <div class="modal-header">
      <span class="modal-title">Test Results</span>
      <button class="modal-close" onclick="closeResultsModal()">&times;</button>
    </div>
    <div class="modal-body" id="resultsModalBody" style="padding:28px;"></div>
  </div>
</div>

</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == "/logo.png":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            with open("logo.png", "rb") as f:
                self.wfile.write(f.read())
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
        elif self.path == "/api/conv_goldens":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if os.path.exists("conversation_goldens.json"):
                with open("conversation_goldens.json") as f:
                    self.wfile.write(f.read().encode())
            else:
                self.wfile.write(b"[]")
        elif self.path == "/api/log":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            clean_log = _ansi_re.sub('', _run_state.get("log", ""))
            self.wfile.write(json.dumps({
                "log": clean_log,
                "done": _run_state.get("done", False),
                "success": _run_state.get("success", False),
                "message": _run_state.get("message", ""),
            }).encode())
        elif self.path == "/api/optimizer_config":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if os.path.exists("optimizer_config.json"):
                with open("optimizer_config.json") as f:
                    self.wfile.write(f.read().encode())
            else:
                self.wfile.write(b'{"algorithm":"GEPA","iterations":10,"metrics":["AnswerRelevancy","LanguageCompliance","Correctness"],"threshold":0.85}')
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

        elif self.path == "/api/optimizer_config":
            with open("optimizer_config.json", "w") as f:
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

        elif self.path == "/api/conv_goldens":
            with open("conversation_goldens.json", "w") as f:
                json.dump(body, f, indent=2)
            self._json_response({"success": True})

        elif self.path == "/api/generate_conv_goldens":
            description = body.get("description", "")
            count = body.get("count", 3)
            system_prompt = body.get("system_prompt", "")
            eval_criteria = body.get("eval_criteria", ["flow_correctness", "language", "edge_case"])

            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                gen_prompt = f"""Generate exactly {count} conversation test scenarios for evaluating a voice/chat agent.

Context about the agent being tested:
System prompt: {system_prompt if system_prompt else '(no system prompt provided)'}

What the scenarios should cover:
{description}

Each scenario represents a multi-turn conversation where only the USER messages are provided (the agent's responses will be generated live during testing).

Return a JSON array with exactly {count} objects, each having:
- "scenario": a short descriptive name for this test case (e.g. "Happy path - full call flow", "User objects mid-pitch")
- "eval_criteria": {json.dumps(eval_criteria)}
- "turns": an array of objects with {{"role": "user", "content": "the user message"}}

Each scenario should have 3-7 user turns that simulate a realistic conversation flow.
Make the messages natural and conversational (use Hinglish if the system prompt suggests a Hindi-speaking audience).
Vary the scenarios: include happy paths, objection handling, confusion, early exits, off-topic tangents.

Return ONLY the JSON array, no other text."""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": gen_prompt}],
                    temperature=0.8,
                )
                content = response.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0]
                generated = json.loads(content)
                self._json_response({"success": True, "scenarios": generated})
            except Exception as e:
                self._json_response({"success": False, "message": str(e)})

        elif self.path == "/api/transcript_to_golden":
            transcript = body.get("transcript", "")
            scenario_name = body.get("scenario_name", "")
            system_prompt = body.get("system_prompt", "")

            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                gen_prompt = f"""Analyze this conversation transcript and convert it into a test scenario for evaluating an AI agent.

Transcript:
{transcript}

Agent's system prompt (for context): {system_prompt if system_prompt else '(not provided)'}

Extract the user messages from this transcript and determine:
1. A descriptive scenario name (use the provided name if given: "{scenario_name}")
2. Which eval criteria are most relevant: "flow_correctness" (does agent follow logical flow), "language" (proper language/tone), "edge_case" (handles objections/interruptions)
3. The user turns only (the agent responses will be regenerated during testing)

Return a single JSON object with:
- "scenario": descriptive name for this test case
- "eval_criteria": array of relevant criteria from ["flow_correctness", "language", "edge_case"]
- "turns": array of objects with {{"role": "user", "content": "extracted user message"}}

Only include user/customer/caller messages in turns, not the agent/assistant messages.
Clean up the messages if needed (fix typos, remove timestamps) but keep the natural language style.

Return ONLY the JSON object, no other text."""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": gen_prompt}],
                    temperature=0.3,
                )
                content = response.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0]
                scenario = json.loads(content)
                self._json_response({"success": True, "scenario": scenario})
            except Exception as e:
                self._json_response({"success": False, "message": str(e)})

        elif self.path == "/api/run":
            action = body.get("action")
            prompt = body.get("prompt", "")

            with open("system_prompt.txt", "w") as f:
                f.write(prompt)

            _run_state["running"] = True
            _run_state["log"] = ""
            _run_state["done"] = False
            _run_state["success"] = False
            _run_state["message"] = ""

            def run_in_bg():
                try:
                    if action == "test_prompts":
                        cmd = ["deepeval", "test", "run", "test_prompts.py", "--", "--tb=short"]
                    elif action == "test_tools":
                        cmd = ["deepeval", "test", "run", "test_tools.py", "--", "--tb=short"]
                    elif action == "test_conversation":
                        cmd = ["deepeval", "test", "run", "test_conversation.py", "--", "--tb=short"]
                    elif action == "optimize":
                        cmd = ["python3", "-u", "optimize_prompt.py"]
                    elif action == "optimize_conversation":
                        cmd = ["python3", "-u", "optimize_conversation.py"]
                    else:
                        _run_state["done"] = True
                        _run_state["success"] = False
                        _run_state["message"] = "Unknown action"
                        _run_state["running"] = False
                        return

                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        bufsize=0, env=env
                    )
                    buf = b""
                    while True:
                        chunk = proc.stdout.read(256)
                        if not chunk:
                            break
                        buf += chunk
                        try:
                            text = buf.decode("utf-8", errors="replace")
                            _run_state["log"] = text
                        except:
                            pass
                    proc.wait()

                    success = proc.returncode == 0
                    if action == "test_prompts":
                        msg = "All tests passed!" if success else "Some tests failed. Check results below."
                    elif action == "test_tools":
                        msg = "All tool tests passed!" if success else "Some tool tests failed. Check results below."
                    elif action == "test_conversation":
                        msg = "All conversation tests passed!" if success else "Some conversation tests failed. Check results below."
                    elif action == "optimize":
                        msg = "Optimization complete! Check the optimized prompt below." if success else "Optimization failed."
                    elif action == "optimize_conversation":
                        msg = "Conversation optimization complete! Check the optimized prompt below." if success else "Conversation optimization failed."

                    _run_state["success"] = success
                    _run_state["message"] = msg
                except Exception as e:
                    _run_state["success"] = False
                    _run_state["message"] = str(e)
                finally:
                    _run_state["done"] = True
                    _run_state["running"] = False

            threading.Thread(target=run_in_bg, daemon=True).start()
            self._json_response({"success": True, "message": "Started"})

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


class QuietHTTPServer(HTTPServer):
    def handle_error(self, request, client_address):
        pass


if __name__ == "__main__":
    server = QuietHTTPServer(("0.0.0.0", PORT), DashboardHandler)
    print(f"Dashboard running at http://0.0.0.0:{PORT}")
    print(f"Accessible on your network at http://192.168.68.57:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
