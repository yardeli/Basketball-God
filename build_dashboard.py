"""
Build Dashboard — Generate a standalone HTML dashboard for NCAA predictions.

Reads model outputs and generates a single-file HTML dashboard with:
  - Today's game predictions with confidence bars
  - Walk-forward backtest results across seasons
  - Feature importances visualization
  - Elo ratings top 25
  - Model training metrics
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config


def build_dashboard():
    """Generate the dashboard HTML file."""
    print("[Dashboard] Building dashboard...")

    # Load data
    predictions = _load_json(config.OUTPUTS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d')}.json", [])
    wf_results = _load_json(config.OUTPUTS_DIR / "walk_forward_results.json", {})
    model_meta = _load_json(config.MODELS_DIR / "model_meta.json", {})

    predictions_json = json.dumps(predictions, default=str)
    wf_results_json = json.dumps(wf_results, default=str)
    feature_importances_json = json.dumps(model_meta.get("feature_importances", {}), default=str)
    training_metrics_json = json.dumps(model_meta.get("training_metrics", {}), default=str)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NCAA Generator v2.0 — Game Predictions</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a1a; color: #e0e0e0; }}
.header {{ background: linear-gradient(135deg, #1a1a3e 0%, #0d0d2b 100%); padding: 24px 32px; border-bottom: 2px solid #2a2a5a; }}
.header h1 {{ font-size: 28px; font-weight: 700; color: #fff; }}
.header p {{ color: #888; margin-top: 4px; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
.section {{ margin-bottom: 32px; }}
.section-title {{ font-size: 20px; font-weight: 600; margin-bottom: 16px; color: #fff; border-left: 4px solid #4a9eff; padding-left: 12px; }}

/* Predictions table */
.pred-table {{ width: 100%; border-collapse: collapse; background: #12122a; border-radius: 12px; overflow: hidden; }}
.pred-table th {{ background: #1a1a3e; padding: 12px 16px; text-align: left; font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
.pred-table td {{ padding: 12px 16px; border-bottom: 1px solid #1a1a3e; }}
.pred-table tr:hover {{ background: #1a1a3e; }}
.confidence-bar {{ height: 8px; border-radius: 4px; background: #1a1a3e; position: relative; overflow: hidden; }}
.confidence-fill {{ height: 100%; border-radius: 4px; }}
.conf-high {{ background: linear-gradient(90deg, #22c55e, #16a34a); }}
.conf-med {{ background: linear-gradient(90deg, #eab308, #ca8a04); }}
.conf-low {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
.winner {{ color: #22c55e; font-weight: 600; }}
.prob {{ font-size: 13px; color: #888; }}

/* Stats cards */
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
.stat-card {{ background: #12122a; border-radius: 12px; padding: 20px; border: 1px solid #2a2a5a; }}
.stat-card .label {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
.stat-card .value {{ font-size: 32px; font-weight: 700; color: #fff; margin-top: 4px; }}
.stat-card .sub {{ font-size: 13px; color: #666; margin-top: 4px; }}

/* Charts */
.chart-container {{ background: #12122a; border-radius: 12px; padding: 20px; border: 1px solid #2a2a5a; }}
.chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 900px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}

/* No data */
.no-data {{ text-align: center; padding: 40px; color: #666; font-size: 16px; }}

/* Season table */
.season-table {{ width: 100%; border-collapse: collapse; background: #12122a; border-radius: 12px; overflow: hidden; margin-top: 16px; }}
.season-table th {{ background: #1a1a3e; padding: 10px 14px; text-align: left; font-size: 12px; color: #888; text-transform: uppercase; }}
.season-table td {{ padding: 10px 14px; border-bottom: 1px solid #1a1a3e; font-size: 14px; }}
.season-table tr:hover {{ background: #1a1a3e; }}
.positive {{ color: #22c55e; }}
.negative {{ color: #ef4444; }}
</style>
</head>
<body>
<div class="header">
  <h1>NCAA Generator v2.0</h1>
  <p>D1 Men's Basketball Predictions — {datetime.now().strftime('%B %d, %Y')}</p>
</div>
<div class="container">

  <!-- Today's Predictions -->
  <div class="section" id="predictions-section">
    <h2 class="section-title">Today's Predictions</h2>
    <div id="predictions-container"></div>
  </div>

  <!-- Model Stats -->
  <div class="section">
    <h2 class="section-title">Model Performance</h2>
    <div class="stats-grid" id="stats-grid"></div>
  </div>

  <!-- Charts -->
  <div class="section">
    <h2 class="section-title">Walk-Forward Backtest</h2>
    <div class="chart-row">
      <div class="chart-container">
        <canvas id="accuracy-chart"></canvas>
      </div>
      <div class="chart-container">
        <canvas id="feature-chart"></canvas>
      </div>
    </div>
  </div>

  <!-- Season Details -->
  <div class="section">
    <h2 class="section-title">Season-by-Season Results</h2>
    <div id="season-table-container"></div>
  </div>
</div>

<script>
const predictions = {predictions_json};
const wfResults = {wf_results_json};
const featureImportances = {feature_importances_json};
const trainingMetrics = {training_metrics_json};

// ── Predictions Table ──
function renderPredictions() {{
  const container = document.getElementById('predictions-container');
  if (!predictions || predictions.length === 0) {{
    container.innerHTML = '<div class="no-data">No games today. Run <code>python predict.py --today</code> to generate predictions.</div>';
    return;
  }}

  let html = '<table class="pred-table"><thead><tr>';
  html += '<th>Away</th><th>Home</th><th>Pick</th><th>Confidence</th><th>Win Prob</th>';
  html += '</tr></thead><tbody>';

  for (const p of predictions) {{
    const confPct = (p.confidence * 100).toFixed(1);
    const confClass = p.confidence > 0.65 ? 'conf-high' : p.confidence > 0.55 ? 'conf-med' : 'conf-low';
    const isHomeWin = p.predicted_winner === p.home_team;

    html += '<tr>';
    html += `<td>${{!isHomeWin ? '<span class="winner">' : ''}}${{p.away_team}}${{!isHomeWin ? '</span>' : ''}}</td>`;
    html += `<td>${{isHomeWin ? '<span class="winner">' : ''}}${{p.home_team}}${{isHomeWin ? '</span>' : ''}}</td>`;
    html += `<td class="winner">${{p.predicted_winner}}</td>`;
    html += `<td><div class="confidence-bar"><div class="confidence-fill ${{confClass}}" style="width:${{confPct}}%"></div></div>${{confPct}}%</td>`;
    html += `<td class="prob">${{(p.home_win_prob * 100).toFixed(1)}}% / ${{(p.away_win_prob * 100).toFixed(1)}}%</td>`;
    html += '</tr>';
  }}

  html += '</tbody></table>';
  container.innerHTML = html;
}}

// ── Stats Cards ──
function renderStats() {{
  const grid = document.getElementById('stats-grid');
  const s = wfResults.summary || {{}};

  const cards = [
    {{ label: 'Avg Accuracy', value: s.avg_accuracy ? (s.avg_accuracy * 100).toFixed(1) + '%' : 'N/A', sub: 'Walk-forward OOS' }},
    {{ label: 'Home Baseline', value: s.avg_baseline ? (s.avg_baseline * 100).toFixed(1) + '%' : 'N/A', sub: 'Always pick home' }},
    {{ label: 'Improvement', value: s.avg_improvement ? (s.avg_improvement > 0 ? '+' : '') + (s.avg_improvement * 100).toFixed(1) + '%' : 'N/A', sub: 'vs baseline' }},
    {{ label: 'Seasons Tested', value: s.n_seasons || 'N/A', sub: 'Out-of-sample' }},
    {{ label: 'Train Accuracy', value: trainingMetrics.train_accuracy ? (trainingMetrics.train_accuracy * 100).toFixed(1) + '%' : 'N/A', sub: trainingMetrics.train_games ? trainingMetrics.train_games + ' games' : '' }},
    {{ label: 'Val Accuracy', value: trainingMetrics.val_accuracy ? (trainingMetrics.val_accuracy * 100).toFixed(1) + '%' : 'N/A', sub: trainingMetrics.val_games ? trainingMetrics.val_games + ' games' : '' }},
  ];

  grid.innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="label">${{c.label}}</div>
      <div class="value">${{c.value}}</div>
      <div class="sub">${{c.sub}}</div>
    </div>
  `).join('');
}}

// ── Accuracy Chart ──
function renderAccuracyChart() {{
  const seasons = wfResults.seasons || [];
  if (seasons.length === 0) return;

  new Chart(document.getElementById('accuracy-chart'), {{
    type: 'line',
    data: {{
      labels: seasons.map(s => s.season),
      datasets: [
        {{
          label: 'Model Accuracy',
          data: seasons.map(s => (s.accuracy * 100).toFixed(1)),
          borderColor: '#4a9eff',
          backgroundColor: 'rgba(74, 158, 255, 0.1)',
          fill: true,
          tension: 0.3,
        }},
        {{
          label: 'Home Baseline',
          data: seasons.map(s => (s.home_win_baseline * 100).toFixed(1)),
          borderColor: '#ef4444',
          borderDash: [5, 5],
          fill: false,
          tension: 0.3,
        }},
      ],
    }},
    options: {{
      responsive: true,
      plugins: {{ title: {{ display: true, text: 'Walk-Forward Accuracy by Season', color: '#fff' }}, legend: {{ labels: {{ color: '#888' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#1a1a3e' }} }},
        y: {{ ticks: {{ color: '#888', callback: v => v + '%' }}, grid: {{ color: '#1a1a3e' }}, min: 40, max: 80 }},
      }},
    }},
  }});
}}

// ── Feature Importance Chart ──
function renderFeatureChart() {{
  const entries = Object.entries(featureImportances).sort((a, b) => b[1] - a[1]).slice(0, 15);
  if (entries.length === 0) return;

  new Chart(document.getElementById('feature-chart'), {{
    type: 'bar',
    data: {{
      labels: entries.map(e => e[0]),
      datasets: [{{
        label: 'Importance',
        data: entries.map(e => e[1]),
        backgroundColor: 'rgba(74, 158, 255, 0.6)',
        borderColor: '#4a9eff',
        borderWidth: 1,
      }}],
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      plugins: {{ title: {{ display: true, text: 'Top Feature Importances', color: '#fff' }}, legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#1a1a3e' }} }},
        y: {{ ticks: {{ color: '#888', font: {{ size: 11 }} }}, grid: {{ display: false }} }},
      }},
    }},
  }});
}}

// ── Season Table ──
function renderSeasonTable() {{
  const seasons = wfResults.seasons || [];
  if (seasons.length === 0) return;

  const container = document.getElementById('season-table-container');
  let html = '<table class="season-table"><thead><tr>';
  html += '<th>Season</th><th>Trained On</th><th>Games</th><th>Accuracy</th><th>Baseline</th><th>Improvement</th><th>Log Loss</th>';
  html += '</tr></thead><tbody>';

  for (const s of seasons) {{
    const impClass = s.improvement > 0 ? 'positive' : 'negative';
    html += '<tr>';
    html += `<td>${{s.season}}</td>`;
    html += `<td>${{s.train_seasons}}</td>`;
    html += `<td>${{s.n_test}}</td>`;
    html += `<td>${{(s.accuracy * 100).toFixed(1)}}%</td>`;
    html += `<td>${{(s.home_win_baseline * 100).toFixed(1)}}%</td>`;
    html += `<td class="${{impClass}}">${{s.improvement > 0 ? '+' : ''}}${{(s.improvement * 100).toFixed(1)}}%</td>`;
    html += `<td>${{s.log_loss.toFixed(4)}}</td>`;
    html += '</tr>';
  }}

  html += '</tbody></table>';
  container.innerHTML = html;
}}

// Init
renderPredictions();
renderStats();
renderAccuracyChart();
renderFeatureChart();
renderSeasonTable();
</script>
</body>
</html>"""

    output_path = config.VIZ_DIR / "dashboard.html"
    config.VIZ_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[Dashboard] Saved to {output_path}")
    return output_path


def _load_json(path: Path, default):
    """Load JSON file or return default."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


if __name__ == "__main__":
    build_dashboard()
