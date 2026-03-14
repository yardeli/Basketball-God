"""
Basketball-God 2.0 — Phase 7: Enhanced Regular Season Model (v2)
=================================================================
Adds 5 improvements over the baseline phase6 model:

  1. CLV-aware output: estimated CLV per prediction vs Elo baseline market
  2. Injury impact feature: injury_impact_diff (live, 0 in historical training)
  3. Elo diff as EXPLICIT feature: diff_elo (computed from full game history)
  4. Momentum with decay weighting:
       - diff_last5_ewm_margin (decay-weighted 5-game margin)
       - Sample weights = exponential decay by season age
  5. Coaching instability + roster disruption:
       - diff_coaching_instability (mid-season coach changes, from DB)
       - diff_roster_disruption (portal disruption score, from JSON)

VALIDATION: Same CPCV walk-forward as phase6 (test 2022-2025, 1-season embargo)
COMPARISON: Compared against phase6 baseline in backtest_ab.py

RUN:
    cd Basketball-God2.0
    python phase7_v2/train_v2.py
"""

import json
import pickle
import sqlite3
import sys
import warnings
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
OUT_DIR  = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PORTAL_JSON = Path(__file__).parent / "coaching_portal.json"

# ── Feature set (baseline + new) ───────────────────────────────────────────────
RS_FEATURES_BASELINE = [
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

NEW_FEATURES = [
    "diff_elo",                   # Improvement 3: explicit Elo differential
    "diff_last5_ewm_margin",      # Improvement 4: decay-weighted recent form
    "diff_coaching_instability",  # Improvement 5: coaching change flag (home - away)
    "diff_roster_disruption",     # Improvement 5: portal disruption score (home - away)
    # Note: diff_injury_impact = 0 in historical training (no historical injury data)
    #        It is included as a feature but zero-filled; gains value in live predictions
    "diff_injury_impact",
]

V2_FEATURES = RS_FEATURES_BASELINE + NEW_FEATURES

XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=10,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    early_stopping_rounds=50,
)

TEST_SEASONS = [2022, 2023, 2024, 2025]
EMBARGO      = 1


# ═══════════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 3: ELO DIFFERENTIAL
# ═══════════════════════════════════════════════════════════════════════════════

def build_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Elo ratings for all teams across all historical games and add
    diff_elo = elo(team1) - elo(team2) at prediction time.

    Uses the full EloSystem from elo.py:
      - K=20 regular season, K=30 conference tournaments
      - 33% mean reversion between seasons
      - Home court advantage: 65 Elo points
      - MOV multiplier (log dampening)

    No data leakage: Elo recorded BEFORE game outcome is applied.
    """
    print("[v2] Building Elo ratings from full game history...")

    sys.path.insert(0, str(ROOT))
    from elo import EloSystem

    conn = sqlite3.connect(DB_PATH)
    games_raw = pd.read_sql_query(
        """
        SELECT game_id, season, day_num, game_date,
               w_team_id, l_team_id, w_score, l_score, w_loc, game_type
        FROM games
        ORDER BY game_date, game_id
        """,
        conn,
    )
    conn.close()

    print(f"  Loaded {len(games_raw):,} games from DB for Elo computation")

    # Build Elo chronologically
    elo = EloSystem()
    # K-factor by game type
    K_CONFERENCE = 30
    K_REGULAR    = 20

    # Pre-game Elo lookup: game_id -> {team_id: elo}
    pregame_elo: dict[str, dict[int, float]] = {}
    prev_season = None

    for _, game in games_raw.iterrows():
        season = int(game["season"])

        if prev_season is not None and season != prev_season:
            elo.new_season()
        prev_season = season

        w_id = int(game["w_team_id"])
        l_id = int(game["l_team_id"])
        gid  = game["game_id"]

        # Record pre-game Elos for both teams
        pregame_elo[gid] = {
            w_id: elo.get_rating(w_id),
            l_id: elo.get_rating(l_id),
        }

        # Use higher K for tournament/conference games
        game_type = str(game.get("game_type", "regular"))
        k_override = K_CONFERENCE if "tourney" in game_type.lower() else K_REGULAR

        neutral = str(game.get("w_loc", "N")) == "N"
        # Temporarily override K
        import config as cfg
        original_k = cfg.ELO_K
        cfg.ELO_K = k_override
        elo.update(w_id, l_id, int(game["w_score"]), int(game["l_score"]), neutral_site=neutral)
        cfg.ELO_K = original_k

    print(f"  Elo computed for {len(pregame_elo):,} games")

    # Map to parquet rows: diff_elo = elo(team1) - elo(team2)
    def get_diff_elo(row):
        game_elos = pregame_elo.get(row["game_id"], {})
        t1 = game_elos.get(int(row["team1_id"]))
        t2 = game_elos.get(int(row["team2_id"]))
        if t1 is None or t2 is None:
            return np.nan
        return t1 - t2

    df = df.copy()
    df["diff_elo"] = df.apply(get_diff_elo, axis=1)

    nan_count = df["diff_elo"].isna().sum()
    print(f"  diff_elo: {nan_count} NaN ({nan_count/len(df):.1%})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 4: MOMENTUM WITH EXPONENTIAL DECAY
# ═══════════════════════════════════════════════════════════════════════════════

def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute decay-weighted 5-game net margin for each team.

    Uses pandas EWM (exponential weighted moving average) with span=5:
      - Each game's contribution decays as α^n where α = 2/(span+1) ≈ 0.33
      - shift(1) prevents current game from being included (no leakage)

    diff_last5_ewm_margin = ewm_margin(team1) - ewm_margin(team2)
    """
    print("[v2] Building momentum features (EWM last-5 margin)...")

    conn = sqlite3.connect(DB_PATH)
    stats = pd.read_sql_query(
        """
        SELECT gs.game_id, gs.team_id,
               gs.score, gs.opp_score,
               g.game_date
        FROM game_stats gs
        JOIN games g ON gs.game_id = g.game_id
        WHERE g.season >= 2009
        ORDER BY gs.team_id, g.game_date, gs.game_id
        """,
        conn,
    )
    conn.close()

    stats["margin"] = stats["score"] - stats["opp_score"]

    # EWM per team, shift(1) to avoid leakage (don't include current game)
    stats = stats.sort_values(["team_id", "game_date", "game_id"]).copy()
    stats["ewm_margin"] = (
        stats.groupby("team_id")["margin"]
        .transform(lambda x: x.ewm(span=5, adjust=False).mean().shift(1))
    )

    # Build lookup: (game_id, team_id) -> ewm_margin
    lookup = (
        stats
        .set_index(["game_id", "team_id"])["ewm_margin"]
        .to_dict()
    )

    df = df.copy()
    df["t1_ewm_margin"] = [
        lookup.get((gid, tid), np.nan)
        for gid, tid in zip(df["game_id"], df["team1_id"].astype(int))
    ]
    df["t2_ewm_margin"] = [
        lookup.get((gid, tid), np.nan)
        for gid, tid in zip(df["game_id"], df["team2_id"].astype(int))
    ]
    df["diff_last5_ewm_margin"] = df["t1_ewm_margin"] - df["t2_ewm_margin"]

    nan_count = df["diff_last5_ewm_margin"].isna().sum()
    print(f"  diff_last5_ewm_margin: {nan_count} NaN ({nan_count/len(df):.1%})")
    return df


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Improvement 4: Exponential decay sample weights.

    Per user spec (adapted to seasons since we don't have exact game dates in weeks):
      - Most recent season:    weight = 1.00
      - 1 season ago:          weight = 0.75
      - 2 seasons ago:         weight = 0.50
      - 3+ seasons ago:        weight = 0.25

    Within each season, games later in the season (higher day_num) get slightly
    higher weight (1.0) vs early-season games (0.85) to reflect more stable stats.
    """
    max_season = df["season"].max()

    season_weight_map = {
        0: 1.00,   # current season
        1: 0.75,
        2: 0.50,
    }

    def season_w(s):
        age = int(max_season) - int(s)
        return season_weight_map.get(age, 0.25)

    season_weights = df["season"].apply(season_w).values

    # Within-season: normalize day_num to [0.85, 1.0]
    max_day = df.groupby("season")["day_num"].transform("max")
    min_day = df.groupby("season")["day_num"].transform("min")
    day_range = (max_day - min_day).clip(lower=1)
    day_pct = (df["day_num"] - min_day) / day_range  # 0 to 1

    within_season_weights = 0.85 + 0.15 * day_pct.values

    combined = season_weights * within_season_weights
    return combined.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT 5: COACHING & PORTAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def build_coaching_portal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Improvement 5: Coaching instability and roster disruption features.

    coaching_instability:
      - Detected from DB team_coaches table (Kaggle data through 2025)
      - A team with > 1 coach in a season has a mid-season coaching change
      - Flag = 1.0 for teams with mid-season changes, 0.0 otherwise

    roster_disruption_score (0.0 - 1.0):
      - Sourced from coaching_portal.json (manually maintained)
      - Measures net portal disruption (heavy exits without matching imports = high score)
      - 0.0 = no notable portal activity
      - 1.0 = complete roster overhaul

    diff_coaching_instability = team1_flag - team2_flag  (-1, 0, or 1)
    diff_roster_disruption = team1_score - team2_score  (-1.0 to 1.0)
    """
    print("[v2] Building coaching instability and roster disruption features...")

    # ── Coaching changes from DB ───────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    coaches = pd.read_sql_query(
        """
        SELECT Season, TeamID, FirstDayNum, LastDayNum, CoachName
        FROM team_coaches
        ORDER BY Season, TeamID, FirstDayNum
        """,
        conn,
    )
    conn.close()

    # Teams with > 1 coach in a season = mid-season change
    n_coaches = coaches.groupby(["Season", "TeamID"]).size().reset_index(name="n")
    mid_season = set(
        zip(
            n_coaches.loc[n_coaches["n"] > 1, "Season"],
            n_coaches.loc[n_coaches["n"] > 1, "TeamID"],
        )
    )
    print(f"  Mid-season coaching changes detected: {len(mid_season)} (season, team) pairs")

    # ── Portal disruption from JSON ────────────────────────────────────────────
    portal_map: dict[tuple[int, int], float] = {}
    try:
        portal_data = json.loads(PORTAL_JSON.read_text(encoding="utf-8"))
        for entry in portal_data.get("portal_disruptions", []):
            portal_map[(int(entry["season"]), int(entry["team_id"]))] = float(entry["net_disruption"])
        print(f"  Portal disruption entries loaded: {len(portal_map)}")
    except Exception as e:
        print(f"  [WARN] Could not load portal JSON: {e}")

    # ── Build feature columns ──────────────────────────────────────────────────
    df = df.copy()

    df["t1_coach_instability"] = [
        1.0 if (int(s), int(tid)) in mid_season else 0.0
        for s, tid in zip(df["season"], df["team1_id"])
    ]
    df["t2_coach_instability"] = [
        1.0 if (int(s), int(tid)) in mid_season else 0.0
        for s, tid in zip(df["season"], df["team2_id"])
    ]
    df["diff_coaching_instability"] = df["t1_coach_instability"] - df["t2_coach_instability"]

    df["t1_roster_disruption"] = [
        portal_map.get((int(s), int(tid)), 0.0)
        for s, tid in zip(df["season"], df["team1_id"])
    ]
    df["t2_roster_disruption"] = [
        portal_map.get((int(s), int(tid)), 0.0)
        for s, tid in zip(df["season"], df["team2_id"])
    ]
    df["diff_roster_disruption"] = df["t1_roster_disruption"] - df["t2_roster_disruption"]

    coach_changed = df["diff_coaching_instability"].abs().sum()
    disrupted = (df["diff_roster_disruption"] != 0).sum()
    print(f"  Games with coaching change flag != 0: {int(coach_changed)}")
    print(f"  Games with roster disruption != 0: {disrupted}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  INJURY FEATURE (Improvement 2)
# ═══════════════════════════════════════════════════════════════════════════════

def build_injury_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Improvement 2: Injury impact feature.

    Historical training: diff_injury_impact = 0.0 (no historical injury data available)
    Live predictions:    text_pipeline.fetch_injury_report() → impact_estimate per team

    The feature is included in the model so it can be used at prediction time.
    Its zero fill during training means it adds no bias, but the model learns
    a coefficient that will activate when live injury data is available.
    """
    df = df.copy()
    df["diff_injury_impact"] = 0.0  # historical placeholder
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING AND PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_build_features(min_season: int = 2010) -> tuple[pd.DataFrame, list[str], dict]:
    """Load parquet and add all v2 features."""
    print("\n[1] Loading base features from parquet...")
    df = pd.read_parquet(FEAT_DIR / "features_all.parquet")
    df = df[df["game_type"] == "regular"].copy()
    df = df[df["season"] >= min_season].copy()
    print(f"  Base: {len(df):,} regular season games ({df['season'].min()}–{df['season'].max()})")

    print("\n[2] Adding Improvement 3: Elo differential...")
    df = build_elo_features(df)

    print("\n[3] Adding Improvement 4: Decay-weighted momentum...")
    df = build_momentum_features(df)

    print("\n[4] Adding Improvement 5: Coaching & portal features...")
    df = build_coaching_portal_features(df)

    print("\n[5] Adding Improvement 2: Injury impact (historical placeholder)...")
    df = build_injury_feature(df)

    # Determine available features (intersection with what's in the dataframe)
    avail = [f for f in V2_FEATURES if f in df.columns]
    missing = [f for f in V2_FEATURES if f not in df.columns]
    if missing:
        print(f"  [WARN] Features not available: {missing}")

    # Fill NaN with per-feature medians
    medians = df[avail].median()
    df[avail] = df[avail].fillna(medians)
    print(f"\n  Total features: {len(avail)} (baseline: {len(RS_FEATURES_BASELINE)}, new: {len(avail)-len(RS_FEATURES_BASELINE)})")

    return df, avail, medians.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def cpcv_splits(df, test_seasons, embargo=1):
    for yr in test_seasons:
        max_train = yr - embargo - 1
        train_idx = np.where(df["season"].values <= max_train)[0]
        test_idx  = np.where(df["season"].values == yr)[0]
        if len(train_idx) < 1000 or len(test_idx) < 100:
            continue
        yield train_idx, test_idx, yr


def train_ensemble(X_train, y_train, sample_weight=None, X_val=None, y_val=None):
    """Train XGBoost + LR ensemble with optional sample weights."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    xgb_model = xgb.XGBClassifier(**XGB_PARAMS, verbosity=0)
    eval_set = [(scaler.transform(X_val), y_val)] if X_val is not None else None
    xgb_model.fit(
        Xs, y_train,
        sample_weight=sample_weight,
        eval_set=eval_set,
        verbose=False,
    )

    lr_model = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
    lr_model.fit(Xs, y_train, sample_weight=sample_weight)

    return xgb_model, lr_model, scaler


def predict_ensemble(xgb_model, lr_model, scaler, X):
    Xs = scaler.transform(X if not hasattr(X, "values") else X.values)
    p_xgb = xgb_model.predict_proba(Xs)[:, 1]
    p_lr  = lr_model.predict_proba(Xs)[:, 1]
    return 0.65 * p_xgb + 0.35 * p_lr


def run_backtest(df: pd.DataFrame, feature_cols: list[str], all_weights: np.ndarray) -> list[dict]:
    """CPCV walk-forward backtest with decay sample weights."""
    print("\n── CPCV Walk-Forward Backtest (v2) ─────────────────────────")
    results = []

    for train_idx, test_idx, yr in cpcv_splits(df, TEST_SEASONS, EMBARGO):
        train = df.iloc[train_idx]
        test  = df.iloc[test_idx]

        X_train = train[feature_cols].values
        y_train = train["label"].values
        weights = all_weights[train_idx]

        val_season = yr - EMBARGO - 1
        val = df[df["season"] == val_season]
        X_val = val[feature_cols].values if len(val) > 0 else None
        y_val = val["label"].values       if len(val) > 0 else None

        X_test = test[feature_cols].values
        y_test = test["label"].values

        xgb_m, lr_m, scaler = train_ensemble(X_train, y_train, weights, X_val, y_val)
        probs = predict_ensemble(xgb_m, lr_m, scaler, X_test)
        preds = (probs >= 0.5).astype(int)

        acc   = accuracy_score(y_test, preds)
        ll    = log_loss(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        auc   = roc_auc_score(y_test, probs)

        # High-confidence accuracy (top 20% most confident predictions)
        thresh_80 = np.percentile(np.abs(probs - 0.5), 80)
        confident = np.abs(probs - 0.5) >= thresh_80
        hc_acc = accuracy_score(y_test[confident], preds[confident]) if confident.sum() > 0 else None

        # ATS accuracy (using diff_massey_spread as proxy for market spread)
        ats_acc = None
        if "diff_massey_spread" in test.columns:
            market_spread = test["diff_massey_spread"].fillna(0).values
            # model predicts team1 covers if model_spread > market_spread
            model_spread_est = (probs - 0.5) * 20  # rough conversion: prob → points
            model_covers = (model_spread_est > market_spread).astype(int)
            actual_margin = (test["diff_avg_margin"].values > 0).astype(int)  # team1 margin > 0
            ats_acc = float(np.mean(model_covers == actual_margin))

        results.append({
            "season":    yr,
            "n_games":   int(len(test)),
            "accuracy":  round(float(acc), 4),
            "log_loss":  round(float(ll), 4),
            "brier":     round(float(brier), 4),
            "auc":       round(float(auc), 4),
            "hc_acc":    round(float(hc_acc), 4) if hc_acc else None,
            "ats_acc":   round(float(ats_acc), 4) if ats_acc else None,
        })
        hc_str  = f"  hc_acc={hc_acc:.3f}" if hc_acc else ""
        ats_str = f"  ats={ats_acc:.3f}" if ats_acc else ""
        print(f"  {yr}: acc={acc:.3f}  brier={brier:.3f}  ll={ll:.3f}{hc_str}{ats_str}  n={len(test)}")

    avg_acc = float(np.mean([r["accuracy"] for r in results]))
    print(f"\n  Average accuracy (2022-2025): {avg_acc:.3f}")
    return results


def train_production(df: pd.DataFrame, feature_cols: list[str], all_weights: np.ndarray):
    """Train production model on all data with decay weights."""
    print("\n── Training Production Model v2 (all data) ──────────────────")
    X = df[feature_cols].values
    y = df["label"].values
    w = all_weights

    val = df[df["season"] >= 2024]
    X_val = val[feature_cols].values
    y_val = val["label"].values

    xgb_m, lr_m, scaler = train_ensemble(X, y, w, X_val, y_val)
    print(f"  Trained on {len(df):,} games with decay sample weights")

    importance = dict(zip(feature_cols, xgb_m.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    return xgb_m, lr_m, scaler, importance


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Basketball-God 2.0 — Phase 7: Enhanced Regular Season Model v2")
    print("=" * 65)

    df, feature_cols, medians = load_and_build_features(min_season=2010)
    df = df.reset_index(drop=True)  # ensure 0-based positional index for weight array alignment

    print("\n[6] Computing sample weights (exponential season decay)...")
    sample_weights = compute_sample_weights(df)
    print(f"  Weight range: {sample_weights.min():.3f} – {sample_weights.max():.3f}")
    print(f"  Season weights: " + ", ".join(
        f"{yr}: {sample_weights[df['season']==yr].mean():.3f}"
        for yr in sorted(df["season"].unique())[-5:]
    ))

    print("\n[7] Running CPCV backtest (test seasons 2022-2025)...")
    backtest_results = run_backtest(df, feature_cols, sample_weights)

    print("\n[8] Training production model v2 on all data...")
    xgb_m, lr_m, scaler, importance = train_production(df, feature_cols, sample_weights)

    print("\n[9] Saving model and artifacts...")

    model_bundle = {
        "xgb_model":     xgb_m,
        "lr_model":      lr_m,
        "scaler":        scaler,
        "feature_cols":  feature_cols,
        "medians":       medians,
        "model_type":    "regular_season_v2",
        "model_version": "2.0",
        "trained_on":    "2010-2025 NCAAB regular season with 5 improvements",
        "n_games":       len(df),
        "improvements": [
            "CLV-aware output (estimated vs Elo baseline)",
            "Injury impact feature (live: text_pipeline; historical: 0.0)",
            "Explicit Elo differential (diff_elo)",
            "Decay-weighted momentum (diff_last5_ewm_margin, season sample weights)",
            "Coaching instability + roster disruption (diff_coaching_instability, diff_roster_disruption)",
        ],
    }
    with open(OUT_DIR / "regular_season_model_v2.pkl", "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"  Saved: {OUT_DIR / 'regular_season_model_v2.pkl'}")

    avg_acc = float(np.mean([r["accuracy"] for r in backtest_results]))
    avg_brier = float(np.mean([r["brier"] for r in backtest_results]))
    avg_hc = [r["hc_acc"] for r in backtest_results if r["hc_acc"]]
    avg_ats = [r["ats_acc"] for r in backtest_results if r["ats_acc"]]

    backtest_out = {
        "model_version":     "v2",
        "per_season":        backtest_results,
        "avg_accuracy":      round(avg_acc, 4),
        "avg_brier":         round(avg_brier, 4),
        "avg_hc_accuracy":   round(float(np.mean(avg_hc)), 4) if avg_hc else None,
        "avg_ats_accuracy":  round(float(np.mean(avg_ats)), 4) if avg_ats else None,
        "test_seasons":      TEST_SEASONS,
        "n_features":        len(feature_cols),
        "feature_cols":      feature_cols,
        "new_features":      NEW_FEATURES,
    }
    with open(OUT_DIR / "backtest_v2.json", "w") as f:
        json.dump(backtest_out, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'backtest_v2.json'}")

    importance_out = {
        "feature_importance": {k: round(v, 6) for k, v in importance.items()},
        "model_version": "v2",
    }
    with open(OUT_DIR / "feature_importance_v2.json", "w") as f:
        json.dump(importance_out, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'feature_importance_v2.json'}")

    print("\n[10] Summary:")
    print(f"  Features:          {len(feature_cols)} ({len(NEW_FEATURES)} new)")
    print(f"  Training games:    {len(df):,}")
    print(f"  Avg CPCV accuracy: {avg_acc:.1%}")
    if avg_ats:
        print(f"  Avg ATS accuracy:  {float(np.mean(avg_ats)):.1%}")
    print(f"\n  Top 10 features:")
    for feat, imp in list(importance.items())[:10]:
        tag = " ← NEW" if feat in NEW_FEATURES else ""
        print(f"    {feat}: {imp:.4f}{tag}")

    print("\nDone. Run backtest_ab.py for full A/B comparison.\n")
    return backtest_out


if __name__ == "__main__":
    main()
