"""Live HTML tree visualization for MCTS search.

Writes two files to log_dir:
  - tree_report.html  (self-contained viewer, written once)
  - tree_data.json    (updated after each iteration)

The HTML polls tree_data.json every 2 seconds and re-renders
the tree without a page reload.
"""

import json
import os
from typing import List, Optional

from .search_algo.base_algo import OptimNode
from .search_algo.mcts_tree_node import MCTSNode

# Sentinel value MCTSNode uses for "no test yet"
_NO_TEST_METRIC = -1.0


class HtmlTreeReport:
    """Writes a self-contained HTML viewer and incrementally-updated JSON data file."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._html_path = os.path.join(log_dir, "tree_report.html")
        self._json_path = os.path.join(log_dir, "tree_data.json")
        self._html_written = False

    def update(
        self,
        nodes: List[MCTSNode],
        optim_nodes: List[OptimNode],
        status: str = "running",
        selected_node_id: int = -1,
        best_q_path: Optional[List[int]] = None,
        best_reward_path: Optional[List[int]] = None,
    ) -> None:
        """Refresh the JSON data file. Write the HTML shell on first call."""
        os.makedirs(self.log_dir, exist_ok=True)
        if not self._html_written:
            self._write_html()
            self._html_written = True
        self._write_json(
            nodes, optim_nodes, status, selected_node_id,
            best_q_path, best_reward_path,
        )

    def _write_html(self) -> None:
        with open(self._html_path, "w", encoding="utf-8") as f:
            f.write(_HTML_TEMPLATE)

    def _write_json(
        self,
        nodes: List[MCTSNode],
        optim_nodes: List[OptimNode],
        status: str,
        selected_node_id: int,
        best_q_path: Optional[List[int]],
        best_reward_path: Optional[List[int]],
    ) -> None:
        data = _build_data(
            nodes, optim_nodes, status, selected_node_id,
            best_q_path, best_reward_path,
        )
        tmp_path = self._json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self._json_path)


def _serialize_node(node: MCTSNode) -> dict:
    """Convert an MCTSNode to a JSON-friendly dict, clearing sentinel values."""
    d = node.to_dict()
    if d["test_metric"] == _NO_TEST_METRIC:
        d["test_metric"] = None
    return d


def _collect_edges(optim_nodes: List[OptimNode]) -> List[dict]:
    """Extract parent-to-child edges from optimization nodes."""
    edges = []
    for optim_node in optim_nodes:
        if optim_node.kind != "optim":
            continue
        for child_id in optim_node.children_id:
            edges.append({
                "from_id": optim_node.parent,
                "to_id": child_id,
                "gradient": optim_node.gradient,
                "gradient_prompt": optim_node.prompt,
            })
    return edges


def _build_data(
    nodes: List[MCTSNode],
    optim_nodes: List[OptimNode],
    status: str,
    selected_node_id: int,
    best_q_path: Optional[List[int]],
    best_reward_path: Optional[List[int]],
) -> dict:
    return {
        "status": status,
        "selected_node_id": selected_node_id,
        "best_q_path": best_q_path or [],
        "best_reward_path": best_reward_path or [],
        "nodes": [_serialize_node(n) for n in nodes],
        "edges": _collect_edges(optim_nodes),
    }


# ---------------------------------------------------------------------------
# Self-contained HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BREAD — MCTS Tree</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
/* ── Reset & base ────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f1117; color: #e0e0e0; }

/* ── Layout ──────────────────────────────────────────── */
#app { display: flex; height: 100vh; }
#tree-pane { flex: 1; overflow: auto; position: relative; }
#detail-pane {
  width: 480px; min-width: 380px; background: #181b22; border-left: 1px solid #2a2d35;
  display: flex; flex-direction: column; overflow: hidden;
  transition: width 0.2s;
}
#detail-pane.hidden { width: 0; min-width: 0; border: none; }

/* ── Header bar ──────────────────────────────────────── */
#header {
  position: sticky; top: 0; z-index: 100; background: #181b22;
  padding: 12px 20px; display: flex; align-items: center; gap: 14px;
  border-bottom: 1px solid #2a2d35; flex-wrap: wrap;
}
#header h1 { font-size: 16px; font-weight: 600; color: #fff; }
.status-badge {
  font-size: 12px; padding: 3px 10px; border-radius: 12px; font-weight: 500;
}
.status-running { background: #1a3a2a; color: #4ade80; }
.status-running::before { content: ''; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #4ade80; margin-right: 6px; animation: pulse 1.5s infinite; }
.status-complete { background: #1a2a3a; color: #60a5fa; }
.status-complete::before { content: '✓ '; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
#node-count { font-size: 12px; color: #888; }

/* ── Path toggle buttons ─────────────────────────────── */
.path-toggles { display: flex; gap: 6px; margin-left: auto; }
.path-btn {
  font-size: 11px; padding: 4px 12px; border-radius: 6px; cursor: pointer;
  border: 1.5px solid transparent; background: #23262e; color: #888;
  transition: all 0.15s; font-weight: 500;
}
.path-btn:hover { color: #ccc; }
.path-btn.active-q { border-color: #a78bfa; color: #a78bfa; background: rgba(167,139,250,0.12); }
.path-btn.active-r { border-color: #34d399; color: #34d399; background: rgba(52,211,153,0.12); }

/* ── Tree container ──────────────────────────────────── */
#tree-container { position: relative; min-height: 100%; }
#edges-svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
#edges-svg line { pointer-events: stroke; cursor: pointer; }

/* ── Node cards ──────────────────────────────────────── */
.node-card {
  position: absolute; width: 86px; text-align: center; border-radius: 10px;
  padding: 8px 4px 6px; cursor: pointer; font-size: 11px; line-height: 1.35;
  background: #1e2028; color: #ddd;
  border: 2px solid #2e3140; transition: transform 0.15s, box-shadow 0.15s;
  z-index: 2; user-select: none;
}
.node-card:hover { transform: scale(1.12); box-shadow: 0 4px 20px rgba(0,0,0,0.5); z-index: 3; border-color: #555; }
.node-card.selected-node { border: 2.5px dashed #ef4444; box-shadow: 0 0 16px rgba(239,68,68,0.5); animation: glow-selected 2s ease-in-out infinite; }
.node-card.selected-node .selected-badge {
  display: block; position: absolute; top: -10px; left: 50%; transform: translateX(-50%);
  background: #ef4444; color: #fff; font-size: 9px; font-weight: 700; padding: 1px 7px;
  border-radius: 6px; white-space: nowrap; letter-spacing: 0.5px;
}
@keyframes glow-selected { 0%,100% { box-shadow: 0 0 12px rgba(239,68,68,0.4); } 50% { box-shadow: 0 0 24px rgba(239,68,68,0.7); } }
.node-card.active { border-color: #60a5fa; box-shadow: 0 0 12px rgba(96,165,250,0.4); }
.node-card.on-q-path { border-color: #a78bfa; box-shadow: 0 0 8px rgba(167,139,250,0.3); }
.node-card.on-r-path { border-color: #34d399; box-shadow: 0 0 8px rgba(52,211,153,0.3); }
.node-card.on-q-path.on-r-path { border-color: #fbbf24; box-shadow: 0 0 8px rgba(251,191,36,0.3); }

.node-id { font-weight: 700; font-size: 13px; display: flex; align-items: center; justify-content: center; gap: 5px; }
.reward-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; border: 1px solid rgba(255,255,255,0.15); flex-shrink: 0; }
.node-reward { font-weight: 600; }
.node-test { font-size: 10px; opacity: 0.8; }
.selected-badge { display: none; }

/* ── Path-highlighted edges ──────────────────────────── */
.edge-q { stroke: #a78bfa !important; stroke-width: 3 !important; }
.edge-r { stroke: #34d399 !important; stroke-width: 3 !important; }
.edge-both { stroke: #fbbf24 !important; stroke-width: 3.5 !important; }

/* ── Detail panel ────────────────────────────────────── */
#detail-header {
  padding: 14px 16px; border-bottom: 1px solid #2a2d35;
  display: flex; justify-content: space-between; align-items: center;
}
#detail-header h2 { font-size: 15px; font-weight: 600; }
#detail-close { background: none; border: none; color: #888; font-size: 20px; cursor: pointer; padding: 4px 8px; }
#detail-close:hover { color: #fff; }
#detail-body { flex: 1; overflow-y: auto; padding: 0; }

.detail-section { padding: 14px 16px; border-bottom: 1px solid #2a2d35; }
.detail-section h3 { font-size: 13px; font-weight: 600; color: #aaa; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
.detail-meta { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 16px; font-size: 13px; }
.detail-meta .label { color: #888; }
.detail-meta .value { font-weight: 600; }

/* ── Prompt view ─────────────────────────────────────── */
.prompt-tabs { display: flex; gap: 4px; margin-bottom: 8px; }
.prompt-tab {
  font-size: 11px; padding: 4px 10px; border-radius: 6px; cursor: pointer;
  background: #23262e; color: #aaa; border: none; transition: background 0.15s;
}
.prompt-tab.active { background: #3b82f6; color: #fff; }
.prompt-content {
  max-height: 300px; overflow-y: auto; background: #12141a; border-radius: 8px;
  padding: 12px; font-size: 13px; line-height: 1.6;
}
.prompt-content pre { white-space: pre-wrap; word-break: break-word; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 12px; }
.prompt-content.markdown p { margin-bottom: 8px; }
.prompt-content.markdown ul, .prompt-content.markdown ol { padding-left: 20px; margin-bottom: 8px; }
.prompt-content.markdown code { background: #23262e; padding: 1px 5px; border-radius: 3px; font-size: 12px; }

/* ── Gradient flow ───────────────────────────────────── */
.gradient-flow { display: flex; flex-direction: column; gap: 10px; }
.gf-step {
  background: #12141a; border-radius: 8px; padding: 10px 12px;
  border-left: 3px solid #3b82f6;
}
.gf-step.gf-gradient { border-left-color: #f59e0b; }
.gf-step-label {
  font-size: 11px; font-weight: 600; color: #888; text-transform: uppercase;
  letter-spacing: 0.5px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;
}
.gf-step-label .arrow { color: #555; }
.gf-step-content {
  max-height: 200px; overflow-y: auto; font-size: 12px; line-height: 1.5;
}
.gf-step-content pre { white-space: pre-wrap; word-break: break-word; font-family: 'Cascadia Code', 'Fira Code', monospace; }

details.gp-details { margin-top: 8px; }
details.gp-details summary { font-size: 11px; color: #888; cursor: pointer; }
details.gp-details summary:hover { color: #bbb; }
details.gp-details .gf-step-content { margin-top: 6px; }

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #555; }

/* ── File-protocol banner ────────────────────────────── */
#file-banner {
  display: none; position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
  background: #44403c; color: #fbbf24; padding: 10px 20px; border-radius: 8px;
  font-size: 13px; z-index: 200; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
#file-banner code { background: #1c1917; padding: 2px 6px; border-radius: 4px; }
</style>
</head>
<body>

<div id="app">
  <div id="tree-pane">
    <div id="header">
      <h1>BREAD — MCTS Tree</h1>
      <span id="status" class="status-badge status-running">running</span>
      <span id="node-count"></span>
      <div class="path-toggles" id="path-toggles" style="display:none;">
        <button class="path-btn" id="btn-q-path" onclick="togglePath('q')">Best Q Path</button>
        <button class="path-btn" id="btn-r-path" onclick="togglePath('r')">Best Reward Path</button>
      </div>
    </div>
    <div id="tree-container">
      <svg id="edges-svg"></svg>
    </div>
  </div>

  <div id="detail-pane" class="hidden">
    <div id="detail-header">
      <h2 id="detail-title">Node</h2>
      <button id="detail-close">&times;</button>
    </div>
    <div id="detail-body"></div>
  </div>
</div>

<div id="file-banner">
  Tree not loading? Run <code>python -m http.server 8000</code> in this directory and open
  <code>http://localhost:8000/tree_report.html</code>
</div>

<script>
// ─── State ──────────────────────────────────────────────
let DATA = null;
let prevJSON = '';
let activeNodeId = null;
let promptMode = {};  // nodeId -> 'rendered' | 'raw'
let showQPath = false;
let showRPath = false;
const NODE_W = 86, NODE_H = 64, V_GAP = 95, H_PAD = 30, TOP_PAD = 70;
let fetchFails = 0;

// ─── Data fetching ──────────────────────────────────────
function fetchData() {
  const url = 'tree_data.json?' + Date.now();
  fetch(url)
    .then(r => { if (!r.ok) throw new Error(r.status); return r.text(); })
    .then(onData)
    .catch(() => {
      try {
        const xhr = new XMLHttpRequest();
        xhr.open('GET', url, true);
        xhr.onload = () => { if (xhr.status === 200 || xhr.status === 0) onData(xhr.responseText); else onFetchFail(); };
        xhr.onerror = onFetchFail;
        xhr.send();
      } catch(e) { onFetchFail(); }
    });
}

function onData(text) {
  fetchFails = 0;
  document.getElementById('file-banner').style.display = 'none';
  if (text === prevJSON) return;
  prevJSON = text;
  DATA = JSON.parse(text);
  render();
}

function onFetchFail() {
  fetchFails++;
  if (fetchFails > 3) document.getElementById('file-banner').style.display = 'block';
}

setInterval(fetchData, 2000);
fetchData();

// ─── Path toggles ───────────────────────────────────────
function togglePath(which) {
  if (which === 'q') showQPath = !showQPath;
  if (which === 'r') showRPath = !showRPath;
  document.getElementById('btn-q-path').classList.toggle('active-q', showQPath);
  document.getElementById('btn-r-path').classList.toggle('active-r', showRPath);
  renderTree();
}

// ─── Render ─────────────────────────────────────────────
function render() {
  if (!DATA) return;
  updateStatus();
  renderTree();
  if (activeNodeId !== null) showDetail(activeNodeId);
}

function updateStatus() {
  const el = document.getElementById('status');
  el.textContent = DATA.status;
  el.className = 'status-badge status-' + DATA.status;
  document.getElementById('node-count').textContent = DATA.nodes.length + ' nodes';
  // Show path buttons only when paths are available
  const hasPaths = DATA.best_q_path.length > 0 || DATA.best_reward_path.length > 0;
  document.getElementById('path-toggles').style.display = hasPaths ? 'flex' : 'none';
}

// ─── Tree layout ────────────────────────────────────────
function computePositions(nodes) {
  const byDepth = {};
  for (const n of nodes) {
    (byDepth[n.depth] = byDepth[n.depth] || []).push(n.id);
  }
  const maxDepth = Math.max(...Object.keys(byDepth).map(Number), 0);
  const maxWidth = Math.max(...Object.values(byDepth).map(a => a.length), 1);
  const containerW = Math.max(maxWidth * (NODE_W + H_PAD) + H_PAD, 400);

  const pos = {};
  for (const [d, ids] of Object.entries(byDepth)) {
    const depth = Number(d);
    const rowW = ids.length * (NODE_W + H_PAD) - H_PAD;
    const startX = (containerW - rowW) / 2;
    ids.forEach((id, i) => {
      pos[id] = {
        x: startX + i * (NODE_W + H_PAD),
        y: TOP_PAD + depth * V_GAP,
      };
    });
  }
  return { positions: pos, width: containerW, height: TOP_PAD + (maxDepth + 1) * V_GAP + 40 };
}

// ─── Build path edge sets ───────────────────────────────
function buildPathEdgeSets() {
  const qEdges = new Set();
  const rEdges = new Set();
  if (showQPath) {
    for (let i = 1; i < DATA.best_q_path.length; i++)
      qEdges.add(DATA.best_q_path[i-1] + '->' + DATA.best_q_path[i]);
  }
  if (showRPath) {
    for (let i = 1; i < DATA.best_reward_path.length; i++)
      rEdges.add(DATA.best_reward_path[i-1] + '->' + DATA.best_reward_path[i]);
  }
  return { qEdges, rEdges };
}

function renderTree() {
  const { positions, width, height } = computePositions(DATA.nodes);
  const container = document.getElementById('tree-container');
  container.style.width = width + 'px';
  container.style.height = height + 'px';

  const qSet = new Set(showQPath ? DATA.best_q_path : []);
  const rSet = new Set(showRPath ? DATA.best_reward_path : []);
  const { qEdges, rEdges } = buildPathEdgeSets();

  // ── Edges (SVG) ──
  const svg = document.getElementById('edges-svg');
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  let svgHTML = '';

  for (const n of DATA.nodes) {
    if (n.parent >= 0 && positions[n.parent] && positions[n.id]) {
      const p = positions[n.parent], c = positions[n.id];
      const x1 = p.x + NODE_W/2, y1 = p.y + NODE_H;
      const x2 = c.x + NODE_W/2, y2 = c.y;
      const edgeKey = n.parent + '->' + n.id;
      const onQ = qEdges.has(edgeKey);
      const onR = rEdges.has(edgeKey);
      let cls = '';
      if (onQ && onR) cls = 'edge-both';
      else if (onQ) cls = 'edge-q';
      else if (onR) cls = 'edge-r';
      // Invisible wide stroke for click target
      svgHTML += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="transparent" stroke-width="14" data-to="${n.id}" />`;
      // Visible stroke
      svgHTML += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#3a3d45" stroke-width="2" class="${cls}" data-to="${n.id}" style="pointer-events:none;" />`;
    }
  }
  svg.innerHTML = svgHTML;
  svg.querySelectorAll('line[data-to]').forEach(line => {
    if (line.getAttribute('stroke') === 'transparent') {
      line.addEventListener('click', () => { const id = +line.dataset.to; activeNodeId = id; showDetail(id); highlightNode(id); });
    }
  });

  // ── Node cards ──
  const existing = {};
  container.querySelectorAll('.node-card').forEach(el => { existing[el.dataset.id] = el; });

  for (const n of DATA.nodes) {
    const el = existing[n.id] || createCard(n);
    updateCard(el, n, positions[n.id], qSet, rSet);
    if (!existing[n.id]) container.appendChild(el);
    delete existing[n.id];
  }
  for (const el of Object.values(existing)) el.remove();
}

function createCard(n) {
  const el = document.createElement('div');
  el.className = 'node-card';
  el.dataset.id = n.id;
  el.innerHTML = `<span class="selected-badge">\u2605 SELECTED</span><div class="node-id"><span class="reward-dot"></span>#${n.id}</div><div class="node-reward"></div><div class="node-test"></div>`;
  el.addEventListener('click', () => { activeNodeId = n.id; showDetail(n.id); highlightNode(n.id); });
  return el;
}

function updateCard(el, n, pos, qSet, rSet) {
  el.style.left = pos.x + 'px';
  el.style.top = pos.y + 'px';
  el.querySelector('.reward-dot').style.background = rewardColor(n.reward);
  el.querySelector('.node-reward').textContent = 'R: ' + n.reward.toFixed(4);
  el.querySelector('.node-test').textContent = 'T: ' + (n.test_metric !== null ? (typeof n.test_metric === 'number' ? n.test_metric.toFixed(4) : n.test_metric) : 'N/A');
  el.classList.toggle('selected-node', n.id === DATA.selected_node_id);
  el.classList.toggle('on-q-path', qSet.has(n.id));
  el.classList.toggle('on-r-path', rSet.has(n.id));
}

function highlightNode(id) {
  document.querySelectorAll('.node-card').forEach(el => el.classList.toggle('active', +el.dataset.id === id));
  document.getElementById('detail-pane').classList.remove('hidden');
}

// ─── Color ──────────────────────────────────────────────
function rewardColor(reward) {
  const t = Math.max(0, Math.min(1, reward));
  let r, g, b;
  if (t < 0.5) {
    const s = t / 0.5;
    r = 200 + Math.round(55 * (1 - s));
    g = Math.round(180 * s);
    b = 50;
  } else {
    const s = (t - 0.5) / 0.5;
    r = Math.round(200 * (1 - s));
    g = 140 + Math.round(40 * s);
    b = 50 + Math.round(30 * s);
  }
  return `rgb(${r},${g},${b})`;
}

// ─── Detail panel ───────────────────────────────────────
document.getElementById('detail-close').addEventListener('click', () => {
  document.getElementById('detail-pane').classList.add('hidden');
  activeNodeId = null;
  document.querySelectorAll('.node-card.active').forEach(el => el.classList.remove('active'));
});

function showDetail(nodeId) {
  const node = DATA.nodes.find(n => n.id === nodeId);
  if (!node) return;

  document.getElementById('detail-title').textContent = `Node #${node.id}`;
  const body = document.getElementById('detail-body');

  const edge = DATA.edges.find(e => e.to_id === nodeId);
  const parentNode = node.parent >= 0 ? DATA.nodes.find(n => n.id === node.parent) : null;

  let html = '';

  // ── Meta section ──
  html += `<div class="detail-section"><h3>Properties</h3><div class="detail-meta">
    <span class="label">ID</span><span class="value">${node.id}</span>
    <span class="label">Depth</span><span class="value">${node.depth}</span>
    <span class="label">Reward</span><span class="value" style="color:${rewardColor(node.reward)}">${node.reward.toFixed(4)}</span>
    <span class="label">Test</span><span class="value">${node.test_metric !== null ? (typeof node.test_metric === 'number' ? node.test_metric.toFixed(4) : node.test_metric) : 'N/A'}</span>
    <span class="label">Q</span><span class="value">${node.q.toFixed(4)}</span>
    <span class="label">UCT</span><span class="value">${node.uct.toFixed(4)}</span>
  </div></div>`;

  // ── Prompt section ──
  const mode = promptMode[nodeId] || 'rendered';
  html += `<div class="detail-section"><h3>Prompt</h3>
    <div class="prompt-tabs">
      <button class="prompt-tab ${mode==='rendered'?'active':''}" onclick="setPromptMode(${nodeId},'rendered')">Rendered</button>
      <button class="prompt-tab ${mode==='raw'?'active':''}" onclick="setPromptMode(${nodeId},'raw')">Raw</button>
    </div>
    <div class="prompt-content ${mode==='rendered'?'markdown':''}">${mode==='rendered' ? renderMd(node.prompt) : '<pre>'+esc(node.prompt)+'</pre>'}</div>
  </div>`;

  // ── Gradient flow section ──
  if (edge && parentNode) {
    html += `<div class="detail-section"><h3>Optimization Step</h3>
      <div class="gradient-flow">
        <div class="gf-step">
          <div class="gf-step-label">Parent Prompt <span style="color:#888">(node #${parentNode.id})</span></div>
          <div class="gf-step-content"><pre>${esc(parentNode.prompt)}</pre></div>
        </div>
        <div class="gf-step gf-gradient">
          <div class="gf-step-label">\u2193 Gradient Feedback</div>
          <div class="gf-step-content"><pre>${esc(edge.gradient || '(none)')}</pre></div>
        </div>
        <div class="gf-step">
          <div class="gf-step-label">\u2193 Child Prompt <span style="color:#888">(this node #${node.id})</span></div>
          <div class="gf-step-content"><pre>${esc(node.prompt)}</pre></div>
        </div>
      </div>`;
    if (edge.gradient_prompt) {
      html += `<details class="gp-details"><summary>Show gradient prompt (LLM input)</summary>
        <div class="gf-step-content"><pre>${esc(edge.gradient_prompt)}</pre></div>
      </details>`;
    }
    html += `</div>`;
  }

  body.innerHTML = html;
  document.getElementById('detail-pane').classList.remove('hidden');
}

function setPromptMode(nodeId, mode) {
  promptMode[nodeId] = mode;
  showDetail(nodeId);
}

// ─── Markdown / escape ──────────────────────────────────
function renderMd(text) {
  if (typeof marked !== 'undefined') {
    try { return marked.parse(text || ''); } catch(e) {}
  }
  return '<pre>' + esc(text) + '</pre>';
}

function esc(text) {
  if (!text) return '';
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}
</script>
</body>
</html>
"""
