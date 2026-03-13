"""
Basketball-God — Master Terminal Dashboard
==========================================
Run:  python dashboard.py
      python dashboard.py --refresh    (force-refresh odds from API)
      python dashboard.py --season 2026

Shows live betting odds, model predictions, bracket backtest,
feature importance, and calibration — all in one beautiful terminal view.
"""

import argparse
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import io
import os
import sys

import numpy as np
import pandas as pd
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).parent

# Force UTF-8 output on Windows so block-drawing chars render correctly
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

console = Console(force_terminal=True, highlight=False)


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_simulation(season: int) -> list[tuple[str, float]]:
    """Load Monte Carlo champion probabilities for a given season."""
    sim_path = ROOT / "phase4_tournament" / "output" / "bracket_simulations.json"
    if not sim_path.exists():
        return []
    sims = json.loads(sim_path.read_text())
    for s in sims:
        if s.get("season") == season:
            return [(t, p) for t, p in s.get("champion_probabilities", [])]
    return []


def load_backtest() -> dict:
    path = ROOT / "phase4_tournament" / "output" / "bracket_backtest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_phase5() -> dict:
    report = ROOT / "phase5_deploy" / "output" / "final_report.json"
    bootstrap = ROOT / "phase5_deploy" / "output" / "bootstrap_ci.json"
    calibration = ROOT / "phase5_deploy" / "output" / "calibration_audit.json"
    importance = ROOT / "phase5_deploy" / "output" / "feature_importance.json"
    worst = ROOT / "phase5_deploy" / "output" / "worst_case.json"
    result = {}
    for key, path in [("report", report), ("bootstrap", bootstrap),
                      ("calibration", calibration), ("importance", importance),
                      ("worst", worst)]:
        if path.exists():
            result[key] = json.loads(path.read_text())
    return result


def load_seed_matchups() -> dict:
    path = ROOT / "phase4_tournament" / "output" / "seed_matchup_stats.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_team_names() -> dict:
    try:
        conn = sqlite3.connect(ROOT / "phase1_data" / "output" / "basketball_god.db")
        df = pd.read_sql("SELECT team_id, team_name FROM teams", conn)
        conn.close()
        return dict(zip(df["team_id"], df["team_name"]))
    except Exception:
        return {}


# ── Section renderers ──────────────────────────────────────────────────────────

def render_header() -> Panel:
    title = Text()
    title.append("  BASKETBALL-GOD  ", style="bold white on dark_orange")
    title.append("  NCAA Tournament Prediction Engine  ", style="bold white")
    title.append("  Powered by 202,529 Games (1985-2026)  ", style="dim white")

    sub = Text(justify="center")
    sub.append("Phase 1: Data  ", style="green")
    sub.append("| ")
    sub.append("Phase 2: Features  ", style="green")
    sub.append("| ")
    sub.append("Phase 3: Models  ", style="green")
    sub.append("| ")
    sub.append("Phase 4: Tournament  ", style="green")
    sub.append("| ")
    sub.append("Phase 5: Robustness  ", style="green")

    content = Text(justify="center")
    content.append_text(title)
    content.append("\n")
    content.append_text(sub)

    return Panel(content, style="bold", border_style="dark_orange",
                 padding=(0, 2))


def render_model_stats(p5: dict) -> Panel:
    """Key model performance metrics."""
    bootstrap = p5.get("bootstrap", {})
    pe = bootstrap.get("point_estimates", {})
    cal = p5.get("calibration", {})
    lift = bootstrap.get("acc_lift", {})

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("metric", style="dim")
    table.add_column("value",  style="bold")
    table.add_column("context", style="dim italic")

    acc = pe.get("model_acc", 0)
    seed_acc = pe.get("seed_acc", 0)
    acc_color = "bright_green" if acc >= 0.72 else "yellow"

    table.add_row(
        "Overall Accuracy",
        Text(f"{acc:.1%}", style=acc_color),
        f"vs seed baseline {seed_acc:.1%}",
    )
    lo = lift.get("ci_lo", 0)
    hi = lift.get("ci_hi", 0)
    table.add_row(
        "Accuracy Lift",
        Text(f"+{lift.get('mean', 0):.1%}", style="bright_green"),
        f"95% CI [{lo:.1%}, {hi:.1%}]",
    )
    table.add_row(
        "Avg ESPN Bracket Pts",
        Text("126.2", style="bright_cyan"),
        "vs 51.1 seed baseline",
    )
    table.add_row(
        "Log-loss",
        Text(f"{pe.get('model_logloss', 0):.4f}", style="white"),
        f"Brier: {pe.get('model_brier', 0):.4f}",
    )
    ece = cal.get("ece", 0)
    cal_color = "bright_green" if ece < 0.04 else "yellow"
    table.add_row(
        "Calibration (ECE)",
        Text(f"{ece:.4f}", style=cal_color),
        "well-calibrated" if ece < 0.05 else "needs work",
    )
    table.add_row(
        "Test Window",
        Text("2015-2025", style="white"),
        f"{pe.get('n_games', 0):,} tournament games",
    )

    return Panel(table, title="[bold dark_orange]Model Performance[/]",
                 border_style="dark_orange", padding=(0, 1))


def render_backtest(backtest: dict) -> Panel:
    """Per-season ESPN bracket scoring backtest."""
    seasons = backtest.get("per_season", [])
    if not seasons:
        return Panel("[dim]No backtest data[/]", title="Bracket Backtest")

    table = Table(box=box.SIMPLE_HEAD, show_header=True,
                  header_style="bold dark_orange")
    table.add_column("Season", style="dim", width=7)
    table.add_column("Model pts", justify="right", width=10)
    table.add_column("Accuracy", justify="right", width=9)
    table.add_column("Seed pts",  justify="right", width=9)
    table.add_column("Seed acc",  justify="right", width=9)
    table.add_column("Delta", justify="right", width=8)

    for s in seasons:
        m_pts = s["model_total_pts"]
        s_pts = s["seed_total_pts"]
        delta = m_pts - s_pts
        acc   = s["model_accuracy"]

        acc_color = "bright_green" if acc >= 0.75 else ("yellow" if acc >= 0.70 else "red")
        delta_color = "bright_green" if delta > 0 else "red"

        table.add_row(
            str(s["season"]),
            Text(f"{m_pts}", style=acc_color),
            Text(f"{acc:.1%}", style=acc_color),
            str(s_pts),
            f"{s['seed_accuracy']:.1%}",
            Text(f"+{delta}" if delta >= 0 else str(delta), style=delta_color),
        )

    avg_m = backtest.get("avg_model_pts", 0)
    avg_s = backtest.get("avg_seed_pts", 0)
    table.add_row(
        "[bold]AVG[/]",
        Text(f"{avg_m:.1f}", style="bold bright_green"),
        Text(f"{backtest.get('avg_model_acc', 0):.1%}", style="bold bright_green"),
        f"{avg_s:.1f}",
        f"{backtest.get('avg_seed_acc', 0):.1%}",
        Text(f"+{avg_m - avg_s:.1f}", style="bold bright_green"),
    )

    return Panel(table, title="[bold dark_orange]ESPN Bracket Scoring Backtest (2015-2025)[/]",
                 border_style="dark_orange", padding=(0, 1))


def render_feature_importance(p5: dict) -> Panel:
    """Top features ranked by importance."""
    imp = p5.get("importance", {})
    features = imp.get("feature_importance", [])[:15]

    if not features:
        return Panel("[dim]No feature data[/]", title="Feature Importance")

    max_score = max(f["importance"] for f in features) if features else 1

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column("rank", style="dim", width=3)
    table.add_column("feature", width=32)
    table.add_column("bar", width=22)
    table.add_column("score", justify="right", width=7)

    colors = ["bright_green", "green", "yellow", "white", "dim white"]
    for i, f in enumerate(features):
        feat = f["feature"]
        score = f["importance"]
        pct = score / max_score
        bar_len = int(pct * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        color = colors[min(i // 3, len(colors) - 1)]
        # Clean up feature name for display
        display = feat.replace("diff_", "d_").replace("_", " ")
        table.add_row(
            f"{i+1}.",
            Text(display, style=color),
            Text(bar, style=color),
            Text(f"{score:.4f}", style="dim"),
        )

    method = imp.get("method", "xgb_fscore")
    return Panel(table,
                 title=f"[bold dark_orange]Feature Importance[/] [dim]({method})[/]",
                 border_style="dark_orange", padding=(0, 1))


def render_seed_upsets(matchups: dict) -> Panel:
    """Classic seed matchup upset rates."""
    classic = {
        "8v9":  "8 vs 9  (coin flip)",
        "5v12": "5 vs 12 (classic upset)",
        "1v16": "1 vs 16 (rare upset)",
        "6v11": "6 vs 11",
        "7v10": "7 vs 10",
        "4v13": "4 vs 13",
        "3v14": "3 vs 14",
    }

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column("matchup", width=22)
    table.add_column("upset%", justify="right", width=7)
    table.add_column("bar", width=16)
    table.add_column("games", justify="right", width=6, style="dim")

    for key, label in classic.items():
        data = matchups.get(key, {})
        if not data:
            continue
        pct = data.get("upset_rate", data.get("upset_pct", 0) / 100)
        n   = data.get("games", data.get("n_games", 0))
        bar_len = int(pct * 15)
        bar = "#" * bar_len + "." * (15 - bar_len)
        color = ("red" if pct > 0.45 else
                 "yellow" if pct > 0.25 else
                 "green")
        table.add_row(
            label,
            Text(f"{pct:.1%}", style=f"bold {color}"),
            Text(bar, style=color),
            str(n),
        )

    return Panel(table, title="[bold dark_orange]Historic Seed Upset Rates[/]",
                 border_style="dark_orange", padding=(0, 1))


def render_worst_seasons(p5: dict) -> Panel:
    """Seasons where model underperformed."""
    worst = p5.get("worst", {})
    by_round = worst.get("by_round", [])
    worst_seasons = worst.get("worst_seasons", [])[:5]

    table = Table(box=None, show_header=True, header_style="bold",
                  padding=(0, 1))
    table.add_column("Season", width=8)
    table.add_column("Accuracy", justify="right", width=10)
    table.add_column("Games", justify="right", width=6, style="dim")

    for r in worst_seasons:
        acc = r.get("acc", 0)
        color = "yellow" if acc >= 0.65 else "red"
        table.add_row(
            str(int(r["season"])),
            Text(f"{acc:.1%}", style=color),
            str(int(r.get("n", 0))),
        )

    r_table = Table(box=None, show_header=True, header_style="bold",
                    padding=(0, 1))
    r_table.add_column("Round", width=15)
    r_table.add_column("Acc", justify="right", width=7)
    r_table.add_column("n", justify="right", width=5, style="dim")

    for r in sorted(by_round, key=lambda x: x.get("acc", 0)):
        acc = r.get("acc", 0)
        color = "bright_green" if acc >= 0.80 else "yellow" if acc >= 0.70 else "white"
        table.add_row  # skip, use r_table
        r_table.add_row(
            str(r.get("round", "")),
            Text(f"{acc:.1%}", style=color),
            str(int(r.get("n", 0))),
        )

    combined = Columns([table, r_table], equal=False, expand=True)
    return Panel(combined,
                 title="[bold dark_orange]Worst Seasons | Accuracy by Round[/]",
                 border_style="dark_orange", padding=(0, 1))


def render_champion_odds(model_probs: list, odds_data: dict,
                         season: int) -> Panel:
    """
    Main table: model champion probabilities vs live market odds.
    """
    from odds_fetcher import compare_model_to_market

    comparison = compare_model_to_market(model_probs, odds_data)

    table = Table(box=box.SIMPLE_HEAD, show_header=True,
                  header_style="bold dark_orange", padding=(0, 1),
                  expand=True)
    table.add_column("#",           width=3,  style="dim", no_wrap=True)
    table.add_column("Team",        min_width=22, no_wrap=True)
    table.add_column("Model%",      justify="right", width=8)
    table.add_column("Mdl Odds",    justify="right", width=9)
    table.add_column("Mkt Odds",    justify="right", width=9)
    table.add_column("Mkt%",        justify="right", width=7)
    table.add_column("Edge",        justify="right", width=8)
    table.add_column("Signal",      width=16, no_wrap=True)

    signal_styles = {
        "MODEL HIGHER":  ("bold bright_green", "[^]"),
        "MARKET HIGHER": ("bold red",          "[v]"),
        "AGREE":         ("dim white",          "[=]"),
        "NO MARKET":     ("dim",                "[ ]"),
    }

    for i, row in enumerate(comparison[:20], 1):
        sig = row["signal"]
        sig_style, sig_icon = signal_styles.get(sig, ("dim", "   "))

        model_color = ("bright_green" if row["model_prob"] > 0.12 else
                       "green" if row["model_prob"] > 0.06 else
                       "white" if row["model_prob"] > 0.02 else "dim white")

        edge_str = ""
        edge_style = "dim"
        if row["edge"] is not None:
            edge_str = f"{row['edge']:+.1%}"
            edge_style = "bright_green" if row["edge"] > 0.02 else (
                "red" if row["edge"] < -0.02 else "dim white"
            )

        mkt_prob_str = ""
        if row["market_prob"] is not None:
            mkt_prob_str = f"{row['market_prob']:.1%}"

        table.add_row(
            str(i),
            Text(row["team"], style=model_color),
            Text(f"{row['model_prob']:.1%}", style=model_color),
            Text(row["model_odds"], style="dim"),
            Text(row["market_odds"], style="cyan"),
            Text(mkt_prob_str, style="dim cyan"),
            Text(edge_str, style=edge_style),
            Text(f"{sig_icon} {sig}", style=sig_style),
        )

    # Status bar (use Text, not a markup string, to avoid rendering issues)
    if odds_data.get("error") and not odds_data.get("cached"):
        status = Text()
        status.append("Odds unavailable: ", style="red")
        status.append(str(odds_data["error"]), style="dim red")
        status.append("  (showing model predictions only)", style="dim")
    elif odds_data.get("cached"):
        age = int((time.time() - odds_data.get("fetched_at", 0)) / 60)
        rem = odds_data.get("requests_remaining", -1)
        status = Text(f"Odds cached {age}m ago | {rem} API requests remaining",
                      style="dim")
    else:
        rem = odds_data.get("requests_remaining", -1)
        status = Text()
        status.append("Live odds  ", style="green")
        status.append(f"| {rem} API requests remaining", style="dim")

    content = Table.grid(padding=0)
    content.add_row(table)
    content.add_row(status)

    title = f"[bold dark_orange]Champion Odds — {season} NCAA Tournament[/bold dark_orange]"
    return Panel(content, title=title, border_style="dark_orange", padding=(0, 1))


def render_calibration_diagram(p5: dict) -> Panel:
    """ASCII reliability diagram."""
    cal = p5.get("calibration", {})
    bins = cal.get("reliability_diagram", [])

    if not bins:
        return Panel("[dim]No calibration data[/]", title="Calibration")

    lines = []
    lines.append(Text("  Predicted vs Actual Win Rate", style="bold"))
    lines.append(Text("  (diagonal = perfect calibration)\n", style="dim"))

    for b in bins:
        pred = b["mean_predicted"]
        actual = b["mean_actual"]
        n = b["n"]
        err = b["calibration_error"]

        bar_len = int(actual * 25)
        bar = "█" * bar_len
        pred_marker = int(pred * 25)

        row = Text()
        row.append(f"  {pred:.0%}  ", style="dim")
        row.append(bar, style="bright_green")
        if pred_marker <= bar_len:
            row.append("│", style="bright_red")
        row.append(f"  {actual:.0%}", style="white")
        row.append(f"  err={err:.3f}  n={n}", style="dim")
        lines.append(row)

    ece = cal.get("ece", 0)
    mce = cal.get("mce", 0)
    ece_color = "bright_green" if ece < 0.04 else "yellow"
    lines.append(Text(""))
    lines.append(Text(f"  ECE={ece:.4f}  MCE={mce:.4f}", style=ece_color))

    content = Text()
    for l in lines:
        content.append_text(l)
        content.append("\n")

    return Panel(content, title="[bold dark_orange]Calibration Diagram[/]",
                 border_style="dark_orange", padding=(0, 1))


def render_legend() -> Panel:
    items = [
        ("MODEL HIGHER", "bright_green", "Model gives better odds than market — potential value bet"),
        ("MARKET HIGHER", "red",         "Market more confident than model"),
        ("AGREE",         "dim white",   "Model and market within 3% of each other"),
        ("NO MARKET",     "dim",         "Team not listed in current futures market"),
    ]
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column("signal", width=14)
    t.add_column("meaning")
    for sig, color, desc in items:
        t.add_row(Text(sig, style=f"bold {color}"), Text(desc, style="dim"))

    note = Text("\n  Edge = Model Prob - Market Fair Prob  |  ", style="dim")
    note.append("Positive edge = model thinks team is undervalued by market", style="dim italic")

    g = Table.grid()
    g.add_row(t)
    g.add_row(note)

    return Panel(g, title="[bold]Signal Legend[/]", border_style="dim",
                 padding=(0, 1))


def render_dataset_info() -> Panel:
    """Quick reference: dataset size and model architecture."""
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column("k", style="dim", width=22)
    t.add_column("v", style="bold")

    rows = [
        ("Total games ingested", "202,529  (1985-2026)"),
        ("Teams", "381 D1 programs"),
        ("Massey ranking rows", "5.8M  (196 systems)"),
        ("Feature set", "44 differential features"),
        ("Training approach", "Transfer learning (D_transfer)"),
        ("Validation",  "CPCV walk-forward, 1-season embargo"),
        ("Round calibration", "Isotonic regression per round"),
        ("Simulation", "10,000 Monte Carlo bracket runs"),
    ]
    for k, v in rows:
        t.add_row(k, v)

    return Panel(t, title="[bold dark_orange]Dataset & Architecture[/]",
                 border_style="dark_orange", padding=(0, 1))


# ── Main dashboard ─────────────────────────────────────────────────────────────

def run_dashboard(season: int = 2025, refresh_odds: bool = False):
    console.clear()

    # Loading spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold dark_orange]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        t1 = progress.add_task("Loading model artifacts...", total=None)
        p5 = load_phase5()
        backtest = load_backtest()
        seed_matchups = load_seed_matchups()
        progress.update(t1, description="Loading simulation results...")
        model_probs = load_simulation(season)
        if not model_probs:
            # Try adjacent season
            for s in [2024, 2023]:
                model_probs = load_simulation(s)
                if model_probs:
                    season = s
                    break
        progress.update(t1, description="Fetching live odds from The Odds API...")
        from odds_fetcher import fetch_championship_odds
        odds_data = fetch_championship_odds(force_refresh=refresh_odds)

    # ── Render ─────────────────────────────────────────────────────────────────
    console.print(render_header())
    console.print()

    # Row 1: Model stats + Dataset info
    console.print(Columns([
        render_model_stats(p5),
        render_dataset_info(),
    ], equal=True, expand=True))
    console.print()

    # Row 2: Champion odds vs market (full width)
    if model_probs:
        console.print(render_champion_odds(model_probs, odds_data, season))
    else:
        # No simulation data — show market-only if available
        if odds_data.get("teams"):
            t = Table(box=box.SIMPLE_HEAD, header_style="bold dark_orange")
            t.add_column("#", width=3, style="dim")
            t.add_column("Team", width=26)
            t.add_column("Market Odds", justify="right", width=12)
            t.add_column("Implied Prob", justify="right", width=12)
            for i, team in enumerate(odds_data["teams"][:20], 1):
                t.add_row(str(i), team["name"], team["best_odds_fmt"],
                          f"{team['fair_prob']:.1%}")
            console.print(Panel(t, title=f"[bold dark_orange]Market Odds — {season}[/]",
                                border_style="dark_orange"))
        else:
            console.print(Panel(
                "[dim]No simulation data found for this season.\n"
                "Run phase4_tournament/bracket.py to generate predictions.[/dim]",
                title="Champion Predictions", border_style="dim"
            ))
    console.print()

    # Row 3: Backtest (full width)
    console.print(render_backtest(backtest))
    console.print()

    # Row 4: Feature importance + Calibration
    console.print(Columns([
        render_feature_importance(p5),
        render_calibration_diagram(p5),
    ], equal=True, expand=True))
    console.print()

    # Row 5: Seed upsets + Worst seasons
    console.print(Columns([
        render_seed_upsets(seed_matchups),
        render_worst_seasons(p5),
    ], equal=True, expand=True))
    console.print()

    # Footer legend
    console.print(render_legend())

    # Footer summary line
    console.print(Rule(style="dark_orange"))
    footer = Text(justify="center")
    footer.append("Basketball-God", style="bold dark_orange")
    footer.append("  |  5-phase build  |  ", style="dim")
    footer.append("202,529 games", style="white")
    footer.append("  |  ", style="dim")
    footer.append("74.1% tournament accuracy", style="bright_green")
    footer.append("  |  ", style="dim")
    footer.append("126.2 avg ESPN bracket pts", style="bright_cyan")
    footer.append("  |  ", style="dim")
    if odds_data.get("error") and not odds_data.get("cached"):
        footer.append(f"Odds unavailable", style="dim red")
    else:
        rem = odds_data.get("requests_remaining", -1)
        footer.append(f"API quota: {rem} req remaining", style="dim")
    console.print(footer)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Basketball-God Terminal Dashboard")
    parser.add_argument("--season",  type=int, default=2025,
                        help="Season year for champion predictions (default: 2025)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force refresh odds from API (uses 1 API request)")
    args = parser.parse_args()

    run_dashboard(season=args.season, refresh_odds=args.refresh)


if __name__ == "__main__":
    main()
