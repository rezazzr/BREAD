"""Live HTML report for APE search.

Writes two files to log_dir:
  - ape_report.html      (self-contained viewer, written once)
  - ape_report_data.json  (updated incrementally as the pipeline progresses)

The HTML polls ape_report_data.json every 2 seconds and re-renders
without a page reload, so you can watch candidates appear in real time.

To view: run `python -m http.server 8000` in the log directory
and open http://localhost:8000/ape_report.html
"""

import json
import os


class ApeHtmlReport:
    """Writes an HTML viewer and incrementally-updated JSON data file."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._html_path = os.path.join(log_dir, "ape_report.html")
        self._json_path = os.path.join(log_dir, "ape_report_data.json")
        self._html_written = False
        self._data = {
            "status": "running",
            "candidates": [],
            "best_candidate": None,
            "best_candidate_eval_details": {},
            "best_candidate_test_details": {},
            "generation_queries": [],
            "timing": {},
            "num_generated": 0,
            "num_unique": 0,
        }

    def _ensure_html(self):
        """Write the HTML shell on first call."""
        if not self._html_written:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(self._html_path, "w", encoding="utf-8") as f:
                f.write(_HTML_TEMPLATE)
            self._html_written = True

    def _flush_json(self):
        """Atomically write the current data to the JSON file."""
        self._ensure_html()
        tmp = self._json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self._json_path)

    # ------------------------------------------------------------------
    # Incremental update methods (called during the APE pipeline)
    # ------------------------------------------------------------------

    def update_generation(self, query_log: dict):
        """Called after each generation query completes."""
        self._data["generation_queries"].append({
            "query_idx": query_log["query_idx"],
            "demos": query_log["demos"],
            "query": query_log["query"],
            "candidates": query_log["candidates"],
        })
        self._data["num_generated"] += len(query_log["candidates"])
        self._flush_json()

    def update_dedup(self, num_generated: int, num_unique: int):
        """Called after deduplication."""
        self._data["num_generated"] = num_generated
        self._data["num_unique"] = num_unique
        self._data["status"] = "evaluating"
        self._flush_json()

    def update_candidate_score(self, candidate_data: dict):
        """Called after each candidate is scored."""
        self._data["candidates"].append(candidate_data)
        # Keep sorted by score descending for live display
        self._data["candidates"].sort(key=lambda x: x.get("eval_score", 0), reverse=True)
        self._flush_json()

    def update_timing(self, timing: dict):
        """Called to update timing info."""
        self._data["timing"].update(timing)
        self._flush_json()

    def finalize(self, data: dict):
        """Called at the end — write the final complete data."""
        self._data = data
        self._data["status"] = "complete"
        self._flush_json()


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>APE — Pipeline Report</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f1117; color: #e0e0e0; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }

/* Header */
.header { padding: 20px 0; border-bottom: 1px solid #2a2d35; margin-bottom: 24px; display: flex; align-items: center; gap: 14px; }
.header h1 { font-size: 22px; font-weight: 600; color: #fff; }
.status-badge { font-size: 12px; padding: 3px 10px; border-radius: 12px; font-weight: 500; }
.status-running { background: #1a3a2a; color: #4ade80; }
.status-running::before { content: ''; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #4ade80; margin-right: 6px; animation: pulse 1.5s infinite; }
.status-evaluating { background: #1a2a3a; color: #60a5fa; }
.status-evaluating::before { content: ''; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #60a5fa; margin-right: 6px; animation: pulse 1.5s infinite; }
.status-complete { background: #1a2a3a; color: #60a5fa; }
.status-complete::before { content: '✓ '; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

/* Pipeline flow */
.pipeline { display: flex; gap: 8px; align-items: center; margin-bottom: 28px; flex-wrap: wrap; }
.pipeline-step {
  padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600;
  background: #1e2028; border: 1.5px solid #2e3140; color: #888;
}
.pipeline-step.active { border-color: #3b82f6; color: #60a5fa; background: rgba(59,130,246,0.1); }
.pipeline-step.done { border-color: #22c55e; color: #4ade80; background: rgba(34,197,94,0.1); }
.pipeline-step.pending { opacity: 0.4; }
.pipeline-arrow { color: #555; font-size: 18px; }
.pipeline-stat { font-size: 11px; color: #666; display: block; font-weight: 400; }

/* Cards */
.card {
  background: #181b22; border: 1px solid #2a2d35; border-radius: 12px;
  margin-bottom: 20px; overflow: hidden;
}
.card-header {
  padding: 14px 18px; border-bottom: 1px solid #2a2d35; cursor: pointer;
  display: flex; justify-content: space-between; align-items: center;
}
.card-header h2 { font-size: 15px; font-weight: 600; }
.card-header .badge { font-size: 12px; padding: 2px 10px; border-radius: 10px; background: #23262e; color: #888; }
.card-body { padding: 16px 18px; }
.card-body.collapsed { display: none; }

/* Bar chart */
.bar-chart { display: flex; flex-direction: column; gap: 6px; }
.bar-row { display: flex; align-items: center; gap: 10px; font-size: 13px; }
.bar-rank { width: 28px; text-align: right; color: #888; flex-shrink: 0; }
.bar-container { flex: 1; height: 28px; background: #12141a; border-radius: 6px; overflow: hidden; position: relative; }
.bar-fill { height: 100%; border-radius: 6px; transition: width 0.3s; display: flex; align-items: center; padding: 0 8px; }
.bar-fill.best { background: linear-gradient(90deg, #22c55e33, #22c55e55); border: 1px solid #22c55e; }
.bar-fill.normal { background: linear-gradient(90deg, #3b82f633, #3b82f655); border: 1px solid #3b82f6; }
.bar-score { flex-shrink: 0; width: 120px; text-align: right; font-weight: 600; font-size: 12px; }
.bar-prompt { font-size: 11px; color: #aaa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Table */
.detail-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.detail-table th { text-align: left; padding: 8px 10px; color: #888; font-weight: 600; border-bottom: 1px solid #2a2d35; font-size: 12px; text-transform: uppercase; }
.detail-table td { padding: 6px 10px; border-bottom: 1px solid #1e2028; }
.detail-table tr:hover { background: #1a1d24; }
.correct { color: #4ade80; } .incorrect { color: #f87171; }

/* Generation queries */
.query-card {
  background: #12141a; border-radius: 8px; padding: 12px; margin-bottom: 10px;
  border-left: 3px solid #3b82f6;
}
.query-header { font-size: 12px; font-weight: 600; color: #60a5fa; margin-bottom: 8px; }
.query-demos { font-size: 12px; color: #888; max-height: 120px; overflow-y: auto; white-space: pre-wrap; font-family: monospace; margin-bottom: 8px; }
.query-candidates { font-size: 12px; }
.query-candidates li { margin: 4px 0; padding: 4px 8px; background: #1e2028; border-radius: 4px; list-style: none; }

/* Summary grid */
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.summary-item { background: #12141a; border-radius: 8px; padding: 14px; text-align: center; }
.summary-value { font-size: 24px; font-weight: 700; color: #fff; }
.summary-label { font-size: 11px; color: #888; margin-top: 4px; }

/* Prompt display */
.prompt-box {
  background: #12141a; border-radius: 8px; padding: 14px; font-size: 13px;
  line-height: 1.6; white-space: pre-wrap; font-family: 'Cascadia Code', monospace;
  max-height: 300px; overflow-y: auto; border: 1px solid #2a2d35;
}

/* Timing */
.timing-bar { display: flex; height: 24px; border-radius: 6px; overflow: hidden; margin-top: 8px; }
.timing-segment { display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600; }
.timing-gen { background: #3b82f644; color: #60a5fa; }
.timing-eval { background: #22c55e44; color: #4ade80; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

#file-banner {
  display: none; position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
  background: #44403c; color: #fbbf24; padding: 10px 20px; border-radius: 8px;
  font-size: 13px; z-index: 200;
}
#file-banner code { background: #1c1917; padding: 2px 6px; border-radius: 4px; }
</style>
</head>
<body>
<div class="container" id="app">
  <div class="header">
    <h1>APE — Automatic Prompt Engineer</h1>
    <span id="status" class="status-badge status-running">running</span>
  </div>
  <div id="content">Waiting for data...</div>
</div>

<div id="file-banner">
  Report not loading? Run <code>python -m http.server 8000</code> in this directory and open
  <code>http://localhost:8000/ape_report.html</code>
</div>

<script>
let DATA = null;
let prevJSON = '';
let fetchFails = 0;

function fetchData() {
  const url = 'ape_report_data.json?' + Date.now();
  fetch(url)
    .then(r => { if (!r.ok) throw new Error(r.status); return r.text(); })
    .then(text => {
      fetchFails = 0;
      document.getElementById('file-banner').style.display = 'none';
      if (text === prevJSON) return;
      prevJSON = text;
      DATA = JSON.parse(text);
      render();
    })
    .catch(() => {
      fetchFails++;
      if (fetchFails > 3) document.getElementById('file-banner').style.display = 'block';
    });
}

setInterval(fetchData, 2000);
fetchData();

function esc(t) { const d = document.createElement('div'); d.textContent = t || ''; return d.innerHTML; }

function render() {
  if (!DATA) return;
  const c = DATA.candidates || [];
  const best = DATA.best_candidate || {};
  const queries = DATA.generation_queries || [];
  const timing = DATA.timing || {};
  const evalD = DATA.best_candidate_eval_details || {};
  const testD = DATA.best_candidate_test_details || {};
  const status = DATA.status || 'running';

  // Update status badge
  const statusEl = document.getElementById('status');
  statusEl.textContent = status;
  statusEl.className = 'status-badge status-' + status;

  let html = '';

  // Pipeline flow
  const isComplete = status === 'complete';
  const isEval = status === 'evaluating' || isComplete;
  const hasTest = best.test_metric != null && best.test_metric >= 0;
  html += `<div class="pipeline">
    <div class="pipeline-step ${queries.length > 0 ? (isEval ? 'done' : 'active') : 'pending'}">Generate<span class="pipeline-stat">${queries.length} queries, ${DATA.num_generated} candidates</span></div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step ${isEval ? 'done' : 'pending'}">Dedup<span class="pipeline-stat">${DATA.num_generated} → ${DATA.num_unique}</span></div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step ${c.length > 0 ? (isComplete ? 'done' : 'active') : 'pending'}">Evaluate<span class="pipeline-stat">${c.length}/${DATA.num_unique} scored</span></div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step ${isComplete ? 'done' : 'pending'}">Select Best<span class="pipeline-stat">${best.eval_score != null ? best.eval_score.toFixed(4) : '—'}</span></div>
    ${hasTest ? `<span class="pipeline-arrow">→</span><div class="pipeline-step done">Test<span class="pipeline-stat">${typeof best.test_metric === 'number' ? best.test_metric.toFixed(4) : best.test_metric}</span></div>` : ''}
  </div>`;

  // Summary
  const bestEval = best.eval_score != null ? best.eval_score.toFixed(4) : '—';
  const bestTest = hasTest ? (typeof best.test_metric === 'number' ? best.test_metric.toFixed(4) : best.test_metric) : '—';
  html += `<div class="summary-grid" style="margin-bottom:20px;">
    <div class="summary-item"><div class="summary-value">${DATA.num_generated || 0}</div><div class="summary-label">Generated</div></div>
    <div class="summary-item"><div class="summary-value">${DATA.num_unique || 0}</div><div class="summary-label">Unique</div></div>
    <div class="summary-item"><div class="summary-value" style="color:#4ade80">${bestEval}</div><div class="summary-label">Best Eval</div></div>
    <div class="summary-item"><div class="summary-value" style="color:#60a5fa">${bestTest}</div><div class="summary-label">Test</div></div>
    <div class="summary-item"><div class="summary-value">${(timing.generation_time || 0).toFixed(2)}s</div><div class="summary-label">Gen Time</div></div>
    <div class="summary-item"><div class="summary-value">${(timing.evaluation_time || 0).toFixed(2)}s</div><div class="summary-label">Eval Time</div></div>
  </div>`;

  // Timing bar
  const totalTime = (timing.generation_time || 0) + (timing.evaluation_time || 0);
  if (totalTime > 0) {
    const genPct = ((timing.generation_time || 0) / totalTime * 100);
    const evalPct = ((timing.evaluation_time || 0) / totalTime * 100);
    html += `<div class="card"><div class="card-header"><h2>Time Breakdown</h2></div><div class="card-body">
      <div class="timing-bar">
        <div class="timing-segment timing-gen" style="width:${genPct}%">Gen ${genPct.toFixed(0)}%</div>
        <div class="timing-segment timing-eval" style="width:${evalPct}%">Eval ${evalPct.toFixed(0)}%</div>
      </div>
    </div></div>`;
  }

  // Best candidate prompt
  if (best.prompt) {
    html += `<div class="card"><div class="card-header" onclick="toggle(this)"><h2>Best Candidate</h2><span class="badge">eval ${bestEval} | test ${bestTest}</span></div>
      <div class="card-body"><div class="prompt-box">${esc(best.prompt)}</div></div></div>`;
  }

  // Candidate ranking bar chart
  if (c.length > 0) {
    const maxScore = Math.max(...c.map(x => x.eval_score), 0.001);
    html += `<div class="card"><div class="card-header" onclick="toggle(this)"><h2>Candidate Ranking</h2><span class="badge">${c.length} candidates</span></div><div class="card-body"><div class="bar-chart">`;
    c.forEach((cand, i) => {
      const w = Math.max(2, cand.eval_score / maxScore * 100);
      const cls = i === 0 ? 'best' : 'normal';
      const correct = cand.num_correct != null ? ` (${cand.num_correct}/${cand.num_total})` : '';
      const origin = cand.origin ? ` [q${cand.origin.query_idx}:c${cand.origin.candidate_idx_in_query}]` : '';
      html += `<div class="bar-row">
        <span class="bar-rank">#${i+1}</span>
        <div class="bar-container"><div class="bar-fill ${cls}" style="width:${w}%"><span class="bar-prompt">${esc(cand.prompt.substring(0,80))}</span></div></div>
        <span class="bar-score">${cand.eval_score.toFixed(4)}${correct}${origin}</span>
      </div>`;
    });
    html += `</div></div></div>`;
  }

  // Eval per-example details
  if (evalD.correct && evalD.correct.length > 0) {
    html += buildExampleTable('Eval Set Details', evalD);
  }
  if (testD.correct && testD.correct.length > 0) {
    html += buildExampleTable('Test Set Details', testD);
  }

  // Generation queries
  if (queries.length > 0) {
    html += `<div class="card"><div class="card-header" onclick="toggle(this)"><h2>Generation Queries</h2><span class="badge">${queries.length} queries</span></div><div class="card-body collapsed">`;
    queries.forEach((q, i) => {
      html += `<div class="query-card">
        <div class="query-header">Query ${i+1} (${(q.candidates||[]).length} candidates)</div>
        <details><summary style="font-size:12px;color:#888;cursor:pointer">Show demos</summary>
          <div class="query-demos">${esc(q.demos)}</div>
        </details>
        <div class="query-candidates"><ul>${(q.candidates||[]).map(c => `<li>${esc(c.substring(0,200))}</li>`).join('')}</ul></div>
      </div>`;
    });
    html += `</div></div>`;
  }

  document.getElementById('content').innerHTML = html;
}

function buildExampleTable(title, details) {
  const correct = details.correct || [];
  const preds = details.preds || [];
  const labels = details.labels || [];
  const numCorrect = correct.filter(c => c === 1).length;
  let h = `<div class="card"><div class="card-header" onclick="toggle(this)"><h2>${title}</h2><span class="badge">${numCorrect}/${correct.length} correct (${(numCorrect/Math.max(correct.length,1)*100).toFixed(1)}%)</span></div><div class="card-body collapsed">`;
  h += `<table class="detail-table"><thead><tr><th>#</th><th>Result</th><th>Prediction</th><th>Label</th></tr></thead><tbody>`;
  correct.forEach((c, i) => {
    const cls = c === 1 ? 'correct' : 'incorrect';
    const mark = c === 1 ? '✓' : '✗';
    h += `<tr><td>${i}</td><td class="${cls}">${mark}</td><td>${esc(preds[i] || '?')}</td><td>${esc(labels[i] || '?')}</td></tr>`;
  });
  h += `</tbody></table></div></div>`;
  return h;
}

function toggle(headerEl) {
  const body = headerEl.nextElementSibling;
  body.classList.toggle('collapsed');
}
</script>
</body>
</html>
"""
