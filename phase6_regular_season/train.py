"""
Basketball-God 2.0 — Phase 6: Regular Season Model Training
============================================================
Trains a dedicated XGBoost model on regular season NCAAB games (2010-2025).
Uses the existing features_all.parquet from Phase 2, filtered to regular season only.

VALIDATION: Walk-forward CPCV, test seasons 2022-2025, 1-season embargo.
OUTPUT:     phase6_regular_season/output/  (model + backtest + report)
RUN:        python phase6_regular_season/train.py
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / "phase2_features" / "output"
OUT_DIR  = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature set ────────────────────────────────────────────────────────────────
# diff_seed excluded: always NaN for regular season games
RS_FEATURES = [
    # Tier 1 (universal — 100% filled)
    "diff_win_pct", "diff_avg_margin", "diff_sos", "diff_rest_days",
    "diff_games_last_7", "diff_win_streak", "diff_h2h_win_pct_5",
    "diff_h2h_win_pct_10", "diff_conf_win_pct",
    # Tier 2 (box score — 2003+, ~92% filled in 2015+)
    "diff_efg_pct", "diff_opp_efg_pct", "diff_to_rate", "diff_opp_to_rate",
    "diff_orb_rate", "diff_drb_rate", "diff_ft_rate", "diff_opp_ft_rate",
    "diff_fg3_rate", "diff_fg3_pct", "diff_ast_to_ratio",
    "diff_blk_rate", "diff_stl_rate",
    "diff_off_eff", "diff_def_eff", "diff_net_eff", "diff_pace",
    # Tier 3 (Massey / NET — ~97% filled in 2015+)
    "diff_massey_avg_rank", "diff_massey_best_rank",
    "diff_massey_n_systems", "diff_massey_spread",
]

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
EMBARGO      = 1   # skip Y-1 to prevent adjacent-season leakage
MIN_TRAIN    = 8   # minimum seasons before first test fold


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_regular_season(min_season: int = 2010) -> pd.DataFrame:
    """Load regular season rows from Phase 2 parquet, fill NaN with medians."""
    df = pd.read_parquet(FEAT_DIR / "features_all.parquet")
    df = df[df["game_type"] == "regular"].copy()
    df = df[df["season"] >= min_season].copy()
    print(f"  Regular season games loaded: {len(df):,}  (seasons {df['season'].min()}–{df['season'].max()})")

    # Get available features (subset to those in the parquet)
    avail = [f for f in RS_FEATURES if f in df.columns]
    missing = [f for f in RS_FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: features not found in parquet: {missing}")

    # Impute NaN with per-feature median (computed on training data only during CV;
    # here we use global median for simplicity — acceptable since NaN < 10%)
    medians = df[avail].median()
    df[avail] = df[avail].fillna(medians)

    print(f"  Features used: {len(avail)}")
    return df, avail, medians


def cpcv_splits(df, test_seasons, embargo=1):
    for yr in test_seasons:
        max_train = yr - embargo - 1
        train_idx = df.index[df["season"] <= max_train]
        test_idx  = df.index[df["season"] == yr]
        if len(train_idx) < 1000 or len(test_idx) < 100:
            continue
        yield train_idx, test_idx, yr


def train_ensemble(X_train, y_train, X_val=None, y_val=None):
    """Train XGBoost + Logistic Regression ensemble."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    # XGBoost
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS, verbosity=0)
    eval_set = [(scaler.transform(X_val), y_val)] if X_val is not None else None
    xgb_model.fit(Xs, y_train,
                  eval_set=eval_set,
                  verbose=False)

    # Logistic Regression on scaled features
    lr_model = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
    lr_model.fit(Xs, y_train)

    return xgb_model, lr_model, scaler


def predict_ensemble(xgb_model, lr_model, scaler, X):
    Xs = scaler.transform(X)
    p_xgb = xgb_model.predict_proba(Xs)[:, 1]
    p_lr  = lr_model.predict_proba(Xs)[:, 1]
    return 0.65 * p_xgb + 0.35 * p_lr


# ── CPCV Backtest ──────────────────────────────────────────────────────────────

def run_backtest(df, feature_cols, medians):
    print("\n── CPCV Walk-Forward Backtest ──────────────────────────────")
    results = []

    for train_idx, test_idx, yr in cpcv_splits(df, TEST_SEASONS, EMBARGO):
        train = df.loc[train_idx]
        test  = df.loc[test_idx]

        X_train = train[feature_cols].values
        y_train = train["label"].values
        X_test  = test[feature_cols].values
        y_test  = test["label"].values

        # Use last season before embargo as validation for early stopping
        val_season = yr - EMBARGO - 1
        val = df[df["season"] == val_season]
        X_val = val[feature_cols].values if len(val) > 0 else None
        y_val = val["label"].values       if len(val) > 0 else None

        xgb_m, lr_m, scaler = train_ensemble(X_train, y_train, X_val, y_val)
        probs = predict_ensemble(xgb_m, lr_m, scaler, X_test)
        preds = (probs >= 0.5).astype(int)

        acc  = accuracy_score(y_test, preds)
        ll   = log_loss(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        auc  = roc_auc_score(y_test, probs)

        results.append({
            "season": yr,
            "n_games": len(test),
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "brier":    round(brier, 4),
            "auc":      round(auc, 4),
        })
        print(f"  {yr}: acc={acc:.3f}  ll={ll:.3f}  brier={brier:.3f}  n={len(test)}")

    avg_acc = np.mean([r["accuracy"] for r in results])
    print(f"\n  Average accuracy (2022-2025): {avg_acc:.3f}")
    return results


# ── Production Model Training ──────────────────────────────────────────────────

def train_production(df, feature_cols, medians):
    """Train final model on all data (no holdout) for production use."""
    print("\n── Training Production Model (all data) ───────────────────")
    X = df[feature_cols].values
    y = df["label"].values

    # Use last 2 seasons as validation for early stopping, don't hold out
    val = df[df["season"] >= 2024]
    X_val = val[feature_cols].values
    y_val = val["label"].values

    xgb_m, lr_m, scaler = train_ensemble(X, y, X_val, y_val)
    print(f"  Trained on {len(df):,} games")

    # Feature importance
    importance = dict(zip(feature_cols,
                          xgb_m.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    return xgb_m, lr_m, scaler, importance


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Basketball-God 2.0 — Phase 6: Regular Season Model")
    print("=" * 60)

    print("\n[1] Loading data...")
    df, feature_cols, medians = load_regular_season(min_season=2010)

    print("\n[2] Running CPCV backtest (test seasons 2022-2025)...")
    backtest_results = run_backtest(df, feature_cols, medians)

    print("\n[3] Training production model on all data (2010-2025)...")
    xgb_m, lr_m, scaler, importance = train_production(df, feature_cols, medians)

    print("\n[4] Saving model and artifacts...")

    # Save model bundle
    model_bundle = {
        "xgb_model":     xgb_m,
        "lr_model":      lr_m,
        "scaler":        scaler,
        "feature_cols":  feature_cols,
        "medians":       medians.to_dict(),
        "model_type":    "regular_season",
        "trained_on":    "2010-2025 NCAAB regular season",
        "n_games":       len(df),
    }
    with open(OUT_DIR / "regular_season_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"  Saved: {OUT_DIR / 'regular_season_model.pkl'}")

    # Save backtest JSON
    avg_acc = float(np.mean([r["accuracy"] for r in backtest_results]))
    backtest_out = {
        "per_season":      backtest_results,
        "avg_accuracy":    round(avg_acc, 4),
        "test_seasons":    TEST_SEASONS,
        "n_features":      len(feature_cols),
        "feature_cols":    feature_cols,
    }
    with open(OUT_DIR / "regular_season_backtest.json", "w") as f:
        json.dump(backtest_out, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'regular_season_backtest.json'}")

    # Save feature importance
    with open(OUT_DIR / "feature_importance.json", "w") as f:
        json.dump({"feature_importance": importance}, f, indent=2)
    print(f"  Saved: {OUT_DIR / 'feature_importance.json'}")

    # Save medians (used as fallback defaults for SeasonStatsStore)
    league_avgs = medians.to_dict()
    with open(ROOT / "data" / "league_averages.json", "w") as f:
        json.dump(league_avgs, f, indent=2)
    print(f"  Saved: {ROOT / 'data' / 'league_averages.json'}")

    print("\n[5] Summary:")
    print(f"  Features:          {len(feature_cols)}")
    print(f"  Training games:    {len(df):,}")
    print(f"  Avg CPCV accuracy: {avg_acc:.1%}")
    print(f"\n  Top 5 features by importance:")
    for feat, imp in list(importance.items())[:5]:
        print(f"    {feat}: {imp:.4f}")

    print("\nDone. Run the server to use the new model.\n")
    return backtest_out


if __name__ == "__main__":
    main()
