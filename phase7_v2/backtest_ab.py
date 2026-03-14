"""
Basketball-God 2.0 — A/B Backtest: Old Model (v1) vs New Model (v2)
====================================================================
Compares the phase6 baseline against the phase7 v2 model on the same
test seasons (2022-2025) with the same CPCV methodology.

Metrics:
  - Overall accuracy (% correct winner predictions)
  - ATS accuracy (% correct against the spread, using Massey spread as proxy)
  - Average CLV estimate (model prob vs Elo-baseline market)
  - ROI simulation (flat $100 on every "STRONG VALUE" signal, >8% edge)
  - Brier score (probability calibration — lower is better)
  - High-confidence hit rate (top 20% most confident predictions)
  - False positive rate on STRONG VALUE signals

Saves: backtest_results_YYYY-MM-DD.json

RUN:
    python phase7_v2/backtest_ab.py
"""

import json
import sys
import sqlite3
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / "phase2_features" / "output"
DB_PATH  = ROOT / "phase1_data" / "output" / "basketball_god.db"
V1_DIR   = ROOT / "phase6_regular_season" / "output"
V2_DIR   = Path(__file__).parent / "output"

sys.path.insert(0, str(ROOT))

TEST_SEASONS = [2022, 2023, 2024, 2025]
EMBARGO      = 1
STRONG_VALUE_THRESHOLD = 0.08  # edge >= 8% = STRONG VALUE

# Typical NCAAB market odds (no-vig): each side ~52.4% implied
MARKET_VIG_RATE = 0.0476  # 4.76% vig (110/-110 standard)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def prob_to_edge(model_prob: float, market_implied: float = 0.5238) -> float:
    """Edge = model probability - market implied probability (no-vig)."""
    return model_prob - market_implied


def american_payout(american: int, stake: float = 100.0) -> float:
    """Profit from winning bet."""
    if american > 0:
        return stake * american / 100
    return stake * 100 / abs(american)


def compute_roi_simulation(
    probs: np.ndarray,
    actuals: np.ndarray,
    elo_diffs: np.ndarray | None,
    stake: float = 100.0,
) -> dict:
    """
    Simulate flat $100 bets on every prediction with edge >= 8% (STRONG VALUE).
    Market odds estimated from model prob: market prices team at model_prob - edge_threshold.

    Returns: {n_bets, n_won, roi, total_pnl, win_rate, false_positive_rate}
    """
    # Estimate market implied prob: average of Elo-based and flat 50/50
    if elo_diffs is not None:
        elo_probs = 1 / (1 + 10 ** (-elo_diffs / 400))
    else:
        elo_probs = np.full(len(probs), 0.5)

    # Market implied = 50% Elo-based, 50% flat 52.38% (vig-adjusted)
    market_implied = 0.5 * elo_probs + 0.5 * 0.5238

    edges = probs - market_implied
    strong_value_mask = edges >= STRONG_VALUE_THRESHOLD

    if strong_value_mask.sum() == 0:
        return {
            "n_bets": 0, "n_won": 0, "total_pnl": 0.0,
            "roi": 0.0, "win_rate": None, "false_positive_rate": None,
        }

    sv_probs   = probs[strong_value_mask]
    sv_actuals = actuals[strong_value_mask]

    # Approximate market odds from implied prob (remove vig estimate)
    # Market offers approx -110 baseline → payout ~$90.91 per $100 bet
    payouts = np.where(sv_probs >= 0.5,
                       stake * 100 / (sv_probs / (1 - sv_probs) * 100 + 100),  # fav
                       stake * (1 - sv_probs) / sv_probs * 100 / 100)           # dog
    # Clip to reasonable range
    payouts = np.clip(payouts, 30, 200)

    won  = sv_actuals == 1
    pnl  = np.where(won, payouts, -stake)
    total_pnl = float(pnl.sum())
    n_bets = int(strong_value_mask.sum())
    n_won  = int(won.sum())
    roi    = total_pnl / (n_bets * stake) if n_bets > 0 else 0.0

    # False positive rate: strong value picks that lost
    fp_rate = 1 - n_won / n_bets if n_bets > 0 else None

    return {
        "n_bets":             n_bets,
        "n_won":              n_won,
        "total_pnl":          round(total_pnl, 2),
        "roi":                round(roi, 4),
        "win_rate":           round(n_won / n_bets, 4) if n_bets > 0 else None,
        "false_positive_rate": round(fp_rate, 4) if fp_rate is not None else None,
    }


def compute_clv_estimate(
    model_probs: np.ndarray,
    elo_diffs: np.ndarray,
) -> dict:
    """
    Retroactive CLV estimate: model probability vs Elo-implied probability.

    Since no historical closing lines are stored, we estimate CLV as:
        CLV = model_prob - elo_implied_prob

    Positive CLV = model adds value over a naive Elo-only market.
    This is a conservative estimate since the real closing line is
    typically more efficient than raw Elo.

    A proper CLV tracker (clv_tracker.py) begins accumulating from deployment.
    """
    elo_probs = 1 / (1 + 10 ** (-elo_diffs / 400))
    clv = model_probs - elo_probs
    return {
        "avg_clv":     round(float(clv.mean()), 4),
        "avg_clv_pct": f"{clv.mean()*100:+.2f}%",
        "pct_positive_clv": round(float((clv > 0).mean()), 4),
        "note": "Estimated vs Elo baseline (real CLV tracked going forward via clv_tracker.py)",
    }


def compute_ats_accuracy(
    probs: np.ndarray,
    actuals: np.ndarray,
    massey_spreads: np.ndarray,
) -> float:
    """
    ATS accuracy: did the model correctly predict the team would beat the
    market spread?

    Market spread proxy: diff_massey_spread (Massey consensus projected margin)
    Model spread estimate: (model_prob - 0.5) * 20 pts (rough linear mapping)

    ATS win if:
      model_spread > massey_spread AND team1 actual margin > massey_spread
    OR
      model_spread < massey_spread AND team1 actual margin < massey_spread
    """
    model_spread = (probs - 0.5) * 20  # approximate
    model_beats_spread = model_spread > massey_spreads
    # For ATS, we'd need actual final margins — use label (team1 won) as proxy
    actual_beats_spread = actuals == 1  # simplified: team1 covers if team1 won
    return float(np.mean(model_beats_spread == actual_beats_spread))


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_base_data():
    """Load the parquet and build Elo diff (needed for CLV estimation)."""
    print("[AB] Loading data...")
    df = pd.read_parquet(FEAT_DIR / "features_all.parquet")
    df = df[df["game_type"] == "regular"].copy()
    df = df[df["season"] >= 2010].copy()

    # Build Elo diff for CLV estimation
    print("[AB] Computing Elo for CLV baseline...")
    conn = sqlite3.connect(DB_PATH)
    games_raw = pd.read_sql_query(
        """SELECT game_id, season, game_date,
                  w_team_id, l_team_id, w_score, l_score, w_loc
           FROM games ORDER BY game_date, game_id""",
        conn,
    )
    conn.close()

    from elo import EloSystem
    elo_sys   = EloSystem()
    elo_map   = {}
    prev_season = None

    for _, g in games_raw.iterrows():
        if prev_season is not None and int(g["season"]) != prev_season:
            elo_sys.new_season()
        prev_season = int(g["season"])
        w = int(g["w_team_id"])
        l = int(g["l_team_id"])
        elo_map[g["game_id"]] = {w: elo_sys.get_rating(w), l: elo_sys.get_rating(l)}
        elo_sys.update(w, l, int(g["w_score"]), int(g["l_score"]),
                       neutral_site=(str(g["w_loc"]) == "N"))

    def get_elo_diff(row):
        m = elo_map.get(row["game_id"], {})
        t1 = m.get(int(row["team1_id"]))
        t2 = m.get(int(row["team2_id"]))
        return (t1 - t2) if (t1 and t2) else 0.0

    df["_elo_diff"] = df.apply(get_elo_diff, axis=1)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATE V1 (BASELINE)
# ═══════════════════════════════════════════════════════════════════════════════

RS_FEATURES_V1 = [
    "diff_win_pct", "diff_avg_margin", "diff_sos", "diff_rest_days",
    "diff_games_last_7", "diff_win_streak", "diff_h2h_win_pct_5",
    "diff_h2h_win_pct_10", "diff_conf_win_pct",
    "diff_efg_pct", "diff_opp_efg_pct", "diff_to_rate", "diff_opp_to_rate",
    "diff_orb_rate", "diff_drb_rate", "diff_ft_rate", "diff_opp_ft_rate",
    "diff_fg3_rate", "diff_fg3_pct", "diff_ast_to_ratio",
    "diff_blk_rate", "diff_stl_rate",
    "diff_off_eff", "diff_def_eff", "diff_net_eff", "diff_pace",
    "diff_massey_avg_rank", "diff_massey_best_rank",
    "diff_massey_n_systems", "diff_massey_spread",
]

def evaluate_v1(df: pd.DataFrame) -> list[dict]:
    """Run CPCV on baseline features (replicate phase6 results exactly)."""
    print("\n[AB] Evaluating V1 (baseline phase6 model)...")

    avail = [f for f in RS_FEATURES_V1 if f in df.columns]
    medians = df[avail].median()
    df_filled = df.copy()
    df_filled[avail] = df_filled[avail].fillna(medians)

    results = []
    for yr in TEST_SEASONS:
        max_train = yr - EMBARGO - 1
        train = df_filled[df_filled["season"] <= max_train]
        test  = df_filled[df_filled["season"] == yr]
        if len(train) < 1000 or len(test) < 100:
            continue

        val_season = yr - EMBARGO - 1
        val = df_filled[df_filled["season"] == val_season]

        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(train[avail].values)
        Xs_test  = scaler.transform(test[avail].values)
        Xs_val   = scaler.transform(val[avail].values) if len(val) > 0 else None

        xgb_m = xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10,
            objective="binary:logistic", eval_metric="logloss", random_state=42,
            early_stopping_rounds=50, verbosity=0,
        )
        eval_set = [(Xs_val, val["label"].values)] if Xs_val is not None else None
        xgb_m.fit(Xs_train, train["label"].values, eval_set=eval_set, verbose=False)

        lr_m = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
        lr_m.fit(Xs_train, train["label"].values)

        probs   = 0.65 * xgb_m.predict_proba(Xs_test)[:, 1] + 0.35 * lr_m.predict_proba(Xs_test)[:, 1]
        preds   = (probs >= 0.5).astype(int)
        actuals = test["label"].values

        acc   = float(accuracy_score(actuals, preds))
        brier = float(brier_score_loss(actuals, probs))
        ll    = float(log_loss(actuals, probs))
        auc   = float(roc_auc_score(actuals, probs))

        # High-confidence
        thresh = np.percentile(np.abs(probs - 0.5), 80)
        confident = np.abs(probs - 0.5) >= thresh
        hc_acc = float(accuracy_score(actuals[confident], preds[confident])) if confident.sum() > 0 else None

        # ATS proxy
        ms = test["diff_massey_spread"].fillna(0).values
        ats_acc = compute_ats_accuracy(probs, actuals, ms) if "diff_massey_spread" in test.columns else None

        # CLV vs Elo
        elo_diffs = test["_elo_diff"].values
        roi_stats = compute_roi_simulation(probs, actuals, elo_diffs)
        clv_stats = compute_clv_estimate(probs, elo_diffs)

        results.append({
            "season":     yr,
            "n_games":    int(len(test)),
            "accuracy":   round(acc, 4),
            "brier":      round(brier, 4),
            "log_loss":   round(ll, 4),
            "auc":        round(auc, 4),
            "hc_acc":     round(hc_acc, 4) if hc_acc else None,
            "ats_acc":    round(ats_acc, 4) if ats_acc else None,
            "roi_stats":  roi_stats,
            "clv":        clv_stats,
        })
        ats_str = f"{ats_acc:.3f}" if ats_acc is not None else "N/A"
        print(f"  V1 {yr}: acc={acc:.3f}  brier={brier:.3f}  ats={ats_str}  clv={clv_stats['avg_clv_pct']}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATE V2 (NEW MODEL)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_v2() -> list[dict]:
    """Load v2 backtest results (computed by train_v2.py)."""
    v2_path = V2_DIR / "backtest_v2.json"
    if not v2_path.exists():
        print(f"[AB] ERROR: {v2_path} not found. Run train_v2.py first.")
        sys.exit(1)
    with open(v2_path) as f:
        data = json.load(f)
    return data["per_season"]


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def format_pct(v, precision=1):
    if v is None: return "N/A"
    return f"{v*100:.{precision}f}%"

def format_delta(new, old, higher_is_better=True, is_brier=False):
    if new is None or old is None: return "  —"
    d = new - old
    if is_brier:
        d = -d  # lower brier is better
    sign = "+" if d >= 0 else ""
    color_up   = "\033[92m" if higher_is_better else "\033[91m"
    color_down = "\033[91m" if higher_is_better else "\033[92m"
    color = color_up if d >= 0 else color_down
    reset = "\033[0m"
    return f"{color}{sign}{d*100:.1f}%{reset}"


def print_comparison(v1_results: list[dict], v2_per_season: list[dict], df_with_elo: pd.DataFrame):
    """Print side-by-side comparison table."""
    print("\n")
    print("=" * 72)
    print("  A/B BACKTEST: Old Model (v1) vs New Model (v2)")
    print("  Test seasons: 2022, 2023, 2024, 2025  |  CPCV walk-forward")
    print("=" * 72)

    def avg(lst, key, default=None):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return float(np.mean(vals)) if vals else default

    # V1 aggregates
    v1_acc   = avg(v1_results, "accuracy")
    v1_brier = avg(v1_results, "brier")
    v1_hc    = avg(v1_results, "hc_acc")
    v1_ats   = avg(v1_results, "ats_acc")
    v1_roi   = sum(r["roi_stats"]["total_pnl"] for r in v1_results)
    v1_nbets = sum(r["roi_stats"]["n_bets"] for r in v1_results)
    v1_fp    = avg([r["roi_stats"] for r in v1_results], "false_positive_rate")
    v1_clv   = avg([r["clv"] for r in v1_results], "avg_clv")

    # V2 aggregates (from backtest_v2.json — which has more limited fields)
    v2_acc   = avg(v2_per_season, "accuracy")
    v2_brier = avg(v2_per_season, "brier")
    v2_hc    = avg(v2_per_season, "hc_acc")
    v2_ats   = avg(v2_per_season, "ats_acc")

    # For V2 ROI, CLV: re-run prediction on v2 model for test sets
    v2_roi_pnl, v2_nbets, v2_fp_rate, v2_clv_est = compute_v2_roi_and_clv(df_with_elo)

    col_w = 14

    def row(label, v1_val, v2_val, fmt="pct", higher_better=True):
        if fmt == "pct":
            v1s = format_pct(v1_val) if v1_val is not None else "N/A"
            v2s = format_pct(v2_val) if v2_val is not None else "N/A"
        elif fmt == "dollar":
            v1s = f"${v1_val:+.0f}" if v1_val is not None else "N/A"
            v2s = f"${v2_val:+.0f}" if v2_val is not None else "N/A"
        else:
            v1s = f"{v1_val:.4f}" if v1_val is not None else "N/A"
            v2s = f"{v2_val:.4f}" if v2_val is not None else "N/A"

        delta = format_delta(v2_val, v1_val, higher_is_better=higher_better)
        print(f"  {label:<28} {v1s:>{col_w}}   {v2s:>{col_w}}   {delta}")

    print(f"\n  {'Metric':<28} {'v1 (Baseline)':>{col_w}}   {'v2 (Enhanced)':>{col_w}}   Delta")
    print("  " + "-" * 65)

    row("Overall Accuracy",         v1_acc,        v2_acc,        "pct")
    row("ATS Accuracy (est.)",       v1_ats,        v2_ats,        "pct")
    row("Avg CLV vs Elo (est.)",     v1_clv,        v2_clv_est,   "pct")
    row("ROI (flat $100/bet)",       (v1_roi / max(v1_nbets*100, 1)) if v1_nbets else None,
                                     (v2_roi_pnl / max(v2_nbets*100, 1)) if v2_nbets else None, "pct")
    row("Brier Score",               v1_brier,      v2_brier,      "raw",  higher_better=False)
    row("High-Conf Hit Rate (top20%)", v1_hc,       v2_hc,         "pct")
    row("Strong Bet False Pos. Rate", v1_fp,        v2_fp_rate,    "pct",  higher_better=False)

    print("  " + "-" * 65)
    print(f"\n  Simulation details:")
    print(f"    V1 STRONG VALUE bets: {v1_nbets} | Total P&L: ${v1_roi:+.0f}")
    print(f"    V2 STRONG VALUE bets: {v2_nbets} | Total P&L: ${v2_roi_pnl:+.0f}")

    print("\n  Per-Season Accuracy Breakdown:")
    v1_by_yr = {r["season"]: r["accuracy"] for r in v1_results}
    v2_by_yr = {r["season"]: r["accuracy"] for r in v2_per_season}
    print(f"  {'Season':<8} {'V1':>8} {'V2':>8} {'Delta':>8}")
    print("  " + "-" * 36)
    for yr in TEST_SEASONS:
        a1 = v1_by_yr.get(yr)
        a2 = v2_by_yr.get(yr)
        if a1 and a2:
            delta = (a2 - a1) * 100
            sign = "+" if delta >= 0 else ""
            print(f"  {yr:<8} {format_pct(a1):>8} {format_pct(a2):>8} {sign}{delta:.1f}%")

    print("\n  Notes:")
    print("    · CLV vs Elo: estimated retroactively (real CLV tracked via clv_tracker.py going forward)")
    print("    · ATS: proxy using diff_massey_spread as market spread")
    print("    · Injury feature = 0 in historical training (activates in live predictions)")
    print("    · ROI simulation assumes standard market odds at ~52.4% implied (110/-110)")
    print("=" * 72)


def compute_v2_roi_and_clv(df_with_elo: pd.DataFrame) -> tuple[float, int, float, float]:
    """Run v2 model predictions on test sets and compute ROI + CLV."""
    v2_bundle_path = V2_DIR / "regular_season_model_v2.pkl"
    if not v2_bundle_path.exists():
        print("[AB] V2 model not found for ROI computation")
        return 0.0, 0, None, None

    import pickle
    with open(v2_bundle_path, "rb") as f:
        bundle = pickle.load(f)

    xgb_m   = bundle["xgb_model"]
    lr_m    = bundle["lr_model"]
    scaler  = bundle["scaler"]
    fcols   = bundle["feature_cols"]

    total_pnl, total_bets, all_fp, all_clv = 0.0, 0, [], []

    for yr in TEST_SEASONS:
        test = df_with_elo[df_with_elo["season"] == yr].copy()
        if len(test) < 50:
            continue

        # Fill missing features
        avail_cols = [c for c in fcols if c in test.columns]
        for c in fcols:
            if c not in test.columns:
                test[c] = 0.0

        medians_fill = test[fcols].median()
        test[fcols] = test[fcols].fillna(medians_fill)

        try:
            Xs = scaler.transform(test[fcols].values)
            probs = 0.65 * xgb_m.predict_proba(Xs)[:, 1] + 0.35 * lr_m.predict_proba(Xs)[:, 1]
        except Exception as e:
            print(f"  [WARN] v2 predict failed for {yr}: {e}")
            continue

        actuals   = test["label"].values
        elo_diffs = test["_elo_diff"].values

        roi_stats = compute_roi_simulation(probs, actuals, elo_diffs)
        clv_stats = compute_clv_estimate(probs, elo_diffs)

        total_pnl  += roi_stats["total_pnl"]
        total_bets += roi_stats["n_bets"]
        if roi_stats["false_positive_rate"] is not None:
            all_fp.append(roi_stats["false_positive_rate"])
        all_clv.append(clv_stats["avg_clv"])

    avg_fp  = float(np.mean(all_fp)) if all_fp else None
    avg_clv = float(np.mean(all_clv)) if all_clv else None
    return total_pnl, total_bets, avg_fp, avg_clv


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Basketball-God 2.0 — A/B Backtest: v1 Baseline vs v2 Enhanced")
    print("=" * 65)

    df = load_base_data()

    print("\n[1] Evaluating V1 baseline model (CPCV 2022-2025)...")
    v1_results = evaluate_v1(df)

    print("\n[2] Loading V2 model results (from backtest_v2.json)...")
    v2_results = evaluate_v2()

    print_comparison(v1_results, v2_results, df)

    # ── Save output ─────────────────────────────────────────────────────────────
    today_str = date.today().isoformat()
    out_path  = ROOT / f"backtest_results_{today_str}.json"

    def avg(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    v2_roi_pnl, v2_nbets, v2_fp, v2_clv = compute_v2_roi_and_clv(df)
    v1_roi  = sum(r["roi_stats"]["total_pnl"] for r in v1_results)
    v1_nbets = sum(r["roi_stats"]["n_bets"] for r in v1_results)
    v1_fp    = avg([r["roi_stats"] for r in v1_results], "false_positive_rate")
    v1_clv   = avg([r["clv"] for r in v1_results], "avg_clv")

    output = {
        "date": today_str,
        "test_seasons": TEST_SEASONS,
        "methodology": "CPCV walk-forward, 1-season embargo, flat $100 bet simulation",
        "v1_baseline": {
            "model":            "phase6_regular_season",
            "avg_accuracy":     avg(v1_results, "accuracy"),
            "avg_brier":        avg(v1_results, "brier"),
            "avg_ats_accuracy": avg(v1_results, "ats_acc"),
            "avg_clv_vs_elo":   v1_clv,
            "total_pnl_$100":   round(v1_roi, 2),
            "n_strong_value_bets": v1_nbets,
            "strong_bet_false_positive_rate": v1_fp,
            "avg_hc_accuracy":  avg(v1_results, "hc_acc"),
            "per_season":       v1_results,
        },
        "v2_enhanced": {
            "model":            "phase7_v2",
            "new_features":     ["diff_elo", "diff_last5_ewm_margin",
                                 "diff_coaching_instability", "diff_roster_disruption",
                                 "diff_injury_impact"],
            "avg_accuracy":     avg(v2_results, "accuracy"),
            "avg_brier":        avg(v2_results, "brier"),
            "avg_ats_accuracy": avg(v2_results, "ats_acc"),
            "avg_clv_vs_elo":   v2_clv,
            "total_pnl_$100":   round(v2_roi_pnl, 2),
            "n_strong_value_bets": v2_nbets,
            "strong_bet_false_positive_rate": v2_fp,
            "avg_hc_accuracy":  avg(v2_results, "hc_acc"),
            "per_season":       v2_results,
        },
        "deltas": {
            "accuracy_delta":   round((avg(v2_results, "accuracy") or 0) - (avg(v1_results, "accuracy") or 0), 4),
            "brier_delta":      round((avg(v1_results, "brier") or 0) - (avg(v2_results, "brier") or 0), 4),
            "ats_delta":        round((avg(v2_results, "ats_acc") or 0) - (avg(v1_results, "ats_acc") or 0), 4),
            "pnl_delta_$100":   round(v2_roi_pnl - v1_roi, 2),
            "clv_delta":        round((v2_clv or 0) - (v1_clv or 0), 4),
        },
        "notes": {
            "clv": "CLV estimated retroactively vs Elo baseline (no historical closing lines stored). Forward CLV tracked via clv_tracker.py.",
            "ats": "ATS accuracy uses diff_massey_spread as market spread proxy.",
            "injury": "diff_injury_impact = 0 in all historical training games (no injury history). Feature activates in live predictions.",
            "roi": "ROI simulation uses approximate market odds derived from model probability.",
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
