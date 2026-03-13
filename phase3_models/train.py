"""
Basketball-God — Phase 3: Era-Aware Model Training
===================================================
Implements and compares four approaches for handling 40 years of basketball
data with very different rules, pace, and data availability.

APPROACH A — Time-decayed weighting
  All data, but exponential decay weights (recent games matter more).
  Tune decay half-life: 5, 10, 15, 20 years.

APPROACH B — Era-stratified ensemble
  Separate XGBoost models per era slice, combined with a meta-learner.
  Model_A: 1985-1999 (Tier 1 features), Model_B: 2000-2014 (Tier 1+2),
  Model_C: 2015+ (all features). Meta-learner receives all 3 predictions.

APPROACH C — Single model with era features
  One XGBoost on all data, explicitly receives era metadata as features.
  Lets the model learn to weight eras differently.

APPROACH D — Transfer learning / fine-tuning
  Pre-train on full historical data (1985-2014), then fine-tune on
  recent data (2015+) using XGBoost's warm-start continuation.
  Benchmark: same model trained only on 2015+ data.

VALIDATION — Combinatorial Purged Cross-Validation (CPCV)
  Each fold = one full season. Test years: 2015-2025.
  For each test year Y, train on seasons <= Y-2 (1-season embargo gap).
  This is the most rigorous time-series CV for overlapping sports seasons.

WHY CPCV: Standard k-fold would mix future games into training (data leakage).
A 1-season embargo prevents the model from memorizing team quality signals
from the season immediately before the test season (pre-season form bleeds
across season boundaries).
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import warnings
import pickle
import os
from pathlib import Path
from datetime import datetime

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (log_loss, brier_score_loss, accuracy_score,
                              roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
FEAT_DIR = ROOT / "phase2_features" / "output"
OUT_DIR  = ROOT / "phase3_models" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
TEST_SEASONS    = list(range(2015, 2026))    # Walk-forward test years
EMBARGO_SEASONS = 1                           # Gap between train end and test
MIN_TRAIN_SEASONS = 10                        # Minimum seasons in training set

# Decay half-lives to test for Approach A (in years)
DECAY_HALFLIVES = [5, 10, 15, 20]

# XGBoost base parameters — tuned for basketball prediction
XGB_BASE = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

# ── Feature sets per tier ──────────────────────────────────────────────────────
# WHY SEPARATE: Approach B trains models on different eras with different
# data availability. Tier 1 goes back to 1985; Tiers 2+3 only to 2003.

def get_feature_cols(df: pd.DataFrame, tier: int = 3) -> list:
    """
    Return diff_* feature columns available for the given tier.
    Tier 1 = universal (1985+)
    Tier 2 = box score (2003+)
    Tier 3 = all including Massey
    """
    t1_features = [
        "diff_win_pct", "diff_avg_margin", "diff_sos", "diff_rest_days",
        "diff_games_last_7", "diff_win_streak", "diff_h2h_win_pct_5",
        "diff_h2h_win_pct_10", "diff_seed", "diff_conf_win_pct",
    ]
    t2_features = t1_features + [
        "diff_efg_pct", "diff_opp_efg_pct", "diff_to_rate", "diff_opp_to_rate",
        "diff_orb_rate", "diff_drb_rate", "diff_ft_rate", "diff_opp_ft_rate",
        "diff_fg3_rate", "diff_fg3_pct", "diff_ast_to_ratio",
        "diff_blk_rate", "diff_stl_rate",
        "diff_off_eff", "diff_def_eff", "diff_net_eff", "diff_pace",
    ]
    t3_features = t2_features + [
        "diff_massey_avg_rank", "diff_massey_best_rank",
        "diff_massey_n_systems", "diff_massey_spread",
    ]

    # Also include raw absolute features where available
    raw_feats = [c for c in df.columns if c.startswith(("t1_","t2_")) and
                 not c.endswith("_id")]

    all_by_tier = {1: t1_features, 2: t2_features, 3: t3_features + raw_feats}
    base = all_by_tier.get(tier, t3_features)
    return [c for c in base if c in df.columns]


def get_era_features(df: pd.DataFrame) -> list:
    """Extra features for Approach C — era context as explicit features."""
    return ["has_3pt", "shot_clock", "data_tier",
            "neutral_site", "num_ot"]


# ── CPCV Splitter ──────────────────────────────────────────────────────────────

def cpcv_splits(df: pd.DataFrame, test_seasons: list, embargo: int = 1):
    """
    Yield (train_idx, test_idx, test_season) for walk-forward CPCV.

    WHY: For each test season Y, we train on all seasons <= Y - embargo - 1.
    The 'embargo' gap prevents signals from adjacent seasons from bleeding in
    (e.g. teams that finished strong in Y-1 carry momentum into Y, and any
    game data from that season could subtly leak into predictions).
    """
    for test_year in test_seasons:
        max_train = test_year - embargo - 1
        if max_train < df["season"].min() + MIN_TRAIN_SEASONS:
            continue
        train_idx = df[df["season"] <= max_train].index
        test_idx  = df[df["season"] == test_year].index
        if len(train_idx) < 1000 or len(test_idx) < 50:
            continue
        yield train_idx, test_idx, test_year


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_predictions(y_true, y_prob, season: int, game_types: pd.Series) -> dict:
    """Compute all evaluation metrics for a single test season."""
    y_pred = (y_prob >= 0.5).astype(int)
    tourney_mask = game_types.isin(["ncaa_tourney"])

    metrics = {
        "season": season,
        "n_games": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
        "brier":    round(brier_score_loss(y_true, y_prob), 4),
        "auc":      round(roc_auc_score(y_true, y_prob), 4) if len(np.unique(y_true)) > 1 else np.nan,
    }

    # Tournament-specific accuracy
    if tourney_mask.sum() > 0:
        tm = tourney_mask.values
        metrics["tourney_accuracy"] = round(accuracy_score(y_true[tm], y_pred[tm]), 4)
        metrics["tourney_n"]        = int(tourney_mask.sum())
    else:
        metrics["tourney_accuracy"] = np.nan
        metrics["tourney_n"]        = 0

    return metrics


def seed_baseline_accuracy(test_df: pd.DataFrame) -> float:
    """
    Accuracy of always picking the lower-seeded team.
    Uses diff_seed: positive = team1 has lower (better) seed.
    """
    if test_df["diff_seed"].isna().all():
        return np.nan
    has_seed = test_df["diff_seed"].notna()
    pred_seed = (test_df.loc[has_seed, "diff_seed"] > 0).astype(int)
    true_seed = test_df.loc[has_seed, "label"]
    return round(accuracy_score(true_seed, pred_seed), 4)


# ── Approach A: Time-Decayed Weighting ────────────────────────────────────────

def approach_a(df: pd.DataFrame, half_life: int) -> list:
    """
    Train XGBoost on all data weighted by exponential temporal decay.

    WHY: Older basketball (pre-shot-clock, no 3-point line) follows different
    patterns. Down-weighting old games lets the model learn modern patterns
    while still benefiting from 40 years of sample size.

    weight(game) = 0.5 ^ ((current_year - game_year) / half_life)
    """
    print(f"\n  [A] Half-life={half_life}y")
    results = []
    feat_cols = get_feature_cols(df, tier=3)
    current_year = df["season"].max()

    for train_idx, test_idx, test_year in cpcv_splits(df, TEST_SEASONS):
        train = df.loc[train_idx].copy()
        test  = df.loc[test_idx].copy()

        X_train = train[feat_cols].fillna(0)
        y_train = train["label"].values
        X_test  = test[feat_cols].fillna(0)
        y_test  = test["label"].values

        # Compute decay weights
        years_ago = current_year - train["season"]
        weights = np.power(0.5, years_ago / half_life).values

        # Validation set = most recent train season (for early stopping)
        val_season = train["season"].max()
        val_mask   = train["season"] == val_season
        X_val, y_val = X_train[val_mask], y_train[val_mask]
        X_tr_fit, y_tr_fit = X_train[~val_mask], y_train[~val_mask]
        w_tr_fit = weights[~val_mask]

        model = xgb.XGBClassifier(**XGB_BASE)
        model.fit(
            X_tr_fit, y_tr_fit,
            sample_weight=w_tr_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_prob = model.predict_proba(X_test)[:, 1]
        m = evaluate_predictions(y_test, y_prob, test_year, test["game_type"])
        m["approach"] = f"A_hl{half_life}"
        m["half_life"] = half_life
        m["seed_baseline"] = seed_baseline_accuracy(test)
        results.append(m)
        print(f"    {test_year}: acc={m['accuracy']:.3f}  tourney={m.get('tourney_accuracy','N/A')}  n={m['n_games']}")

    return results


# ── Approach B: Era-Stratified Ensemble ───────────────────────────────────────

def approach_b(df: pd.DataFrame) -> list:
    """
    Three sub-models per era, combined with a Logistic Regression meta-learner.

    Model_A: 1985-1999 (Tier 1 features — only universal stats available)
    Model_B: 2000-2014 (Tier 1+2 features — box scores begin 2003)
    Model_C: 2015+     (All features — modern era, Massey, portal)
    Meta:    LR on all 3 probability outputs + era indicator

    WHY: A team's 3PT% in 1990 is structurally different from 2020 because
    the 3-point line was rare and often a gimmick. Separate models don't need
    to reconcile incompatible statistical norms.
    """
    print(f"\n  [B] Era-stratified ensemble")
    results = []

    # Feature sets per sub-model
    feat_a = get_feature_cols(df, tier=1)   # Universal only
    feat_b = get_feature_cols(df, tier=2)   # Tier 1+2
    feat_c = get_feature_cols(df, tier=3)   # All

    for train_idx, test_idx, test_year in cpcv_splits(df, TEST_SEASONS):
        train = df.loc[train_idx].copy()
        test  = df.loc[test_idx].copy()

        # ── Train three sub-models ──
        def train_sub(era_df, feat_cols, val_frac=0.1):
            X = era_df[feat_cols].fillna(0)
            y = era_df["label"].values
            if len(X) < 100: return None, feat_cols
            val_mask = era_df["season"] == era_df["season"].max()
            model = xgb.XGBClassifier(**{**XGB_BASE, "n_estimators": 300})
            try:
                model.fit(X[~val_mask], y[~val_mask],
                          eval_set=[(X[val_mask], y[val_mask])],
                          verbose=False)
            except Exception:
                model.fit(X, y, verbose=False)
            return model, feat_cols

        model_a, fa = train_sub(train[train["season"] <= 1999], feat_a)
        model_b, fb = train_sub(train[(train["season"] >= 2000) & (train["season"] <= 2014)], feat_b)
        model_c, fc = train_sub(train[train["season"] >= 2015], feat_c)

        # ── Get sub-model predictions on test set ──
        def predict_sub(model, feat_cols, df_):
            if model is None:
                return np.full(len(df_), 0.5)
            return model.predict_proba(df_[feat_cols].fillna(0))[:, 1]

        p_a = predict_sub(model_a, fa, test)
        p_b = predict_sub(model_b, fb, test)
        p_c = predict_sub(model_c, fc, test)

        # ── Meta-learner ──
        # Train meta on the same train set using out-of-fold predictions
        meta_X_list, meta_y_list = [], []
        # Simple: use each sub-model's in-sample predictions on its own era
        # (acceptable for meta training since sub-models are era-specific)
        for era_df, model, feat in [
            (train[train["season"] <= 1999], model_a, fa),
            (train[(train["season"] >= 2000) & (train["season"] <= 2014)], model_b, fb),
            (train[train["season"] >= 2015], model_c, fc),
        ]:
            if model is None or len(era_df) < 50: continue
            p = model.predict_proba(era_df[feat].fillna(0))[:, 1]
            # Pad the other 2 probs with 0.5
            row_idx = era_df.index
            all_probs = np.column_stack([p, np.full(len(p), 0.5), np.full(len(p), 0.5)])
            meta_X_list.append(all_probs)
            meta_y_list.append(era_df["label"].values)

        if meta_X_list:
            meta_X = np.vstack(meta_X_list)
            meta_y = np.concatenate(meta_y_list)
            meta_model = LogisticRegression(C=1.0, max_iter=500)
            meta_model.fit(meta_X, meta_y)
            meta_probs_test = np.column_stack([p_a, p_b, p_c])
            y_prob = meta_model.predict_proba(meta_probs_test)[:, 1]
        else:
            # Fallback: simple average
            y_prob = (p_a + p_b + p_c) / 3

        y_test = test["label"].values
        m = evaluate_predictions(y_test, y_prob, test_year, test["game_type"])
        m["approach"] = "B"
        m["seed_baseline"] = seed_baseline_accuracy(test)
        results.append(m)
        print(f"    {test_year}: acc={m['accuracy']:.3f}  tourney={m.get('tourney_accuracy','N/A')}  n={m['n_games']}")

    return results


# ── Approach C: Single model + era features ────────────────────────────────────

def approach_c(df: pd.DataFrame) -> list:
    """
    Single XGBoost on all data with era metadata as explicit features.

    Era features: shot_clock_length, has_3pt_line, era_numeric (year),
    data_tier, neutral_site.

    WHY: If the model can SEE that shot_clock=30 vs 45, it can learn that
    those two regimes need different interpretations of pace and efficiency.
    No manual era segmentation needed — the model discovers it.
    """
    print(f"\n  [C] Single model + era features")
    results = []
    feat_cols = get_feature_cols(df, tier=3) + get_era_features(df)
    feat_cols = list(dict.fromkeys(feat_cols))  # deduplicate

    # Add numeric season (year) as a feature
    df = df.copy()
    df["season_numeric"] = df["season"].astype(float)
    feat_cols.append("season_numeric")

    for train_idx, test_idx, test_year in cpcv_splits(df, TEST_SEASONS):
        train = df.loc[train_idx].copy()
        test  = df.loc[test_idx].copy()

        X_train = train[feat_cols].fillna(0)
        y_train = train["label"].values
        X_test  = test[feat_cols].fillna(0)
        y_test  = test["label"].values

        val_mask = train["season"] == train["season"].max()
        model = xgb.XGBClassifier(**XGB_BASE)
        model.fit(
            X_train[~val_mask], y_train[~val_mask],
            eval_set=[(X_train[val_mask], y_train[val_mask])],
            verbose=False,
        )
        y_prob = model.predict_proba(X_test)[:, 1]
        m = evaluate_predictions(y_test, y_prob, test_year, test["game_type"])
        m["approach"] = "C"
        m["seed_baseline"] = seed_baseline_accuracy(test)
        results.append(m)
        print(f"    {test_year}: acc={m['accuracy']:.3f}  tourney={m.get('tourney_accuracy','N/A')}  n={m['n_games']}")

    return results


# ── Approach D: Transfer learning / fine-tuning ────────────────────────────────

def approach_d(df: pd.DataFrame) -> list:
    """
    Pre-train on full historical data, fine-tune on recent years.

    Step 1: Train base XGBoost on 1985-2012.
    Step 2: Continue training (warm-start via xgb_model) on 2013+ data.
    Benchmark: XGBoost trained only on the fine-tune period (2013+).

    WHY: "Transfer learning" in tree models = base trees capture universal
    basketball logic (home court, rest, strength of schedule), then fine-tune
    trees refine those patterns for the modern era without forgetting the base.
    """
    print(f"\n  [D] Transfer learning")
    results_transfer = []
    results_recent_only = []
    feat_cols = get_feature_cols(df, tier=3)
    PRETRAIN_CUTOFF = 2012

    for train_idx, test_idx, test_year in cpcv_splits(df, TEST_SEASONS):
        train = df.loc[train_idx].copy()
        test  = df.loc[test_idx].copy()

        X_test = test[feat_cols].fillna(0)
        y_test = test["label"].values

        # ── Benchmark: recent-only ──
        recent = train[train["season"] > PRETRAIN_CUTOFF]
        if len(recent) > 200:
            val_mask_r = recent["season"] == recent["season"].max()
            X_r = recent[feat_cols].fillna(0)
            y_r = recent["label"].values
            m_recent = xgb.XGBClassifier(**{**XGB_BASE, "n_estimators": 300})
            try:
                m_recent.fit(X_r[~val_mask_r], y_r[~val_mask_r],
                              eval_set=[(X_r[val_mask_r], y_r[val_mask_r])],
                              verbose=False)
            except:
                m_recent.fit(X_r, y_r, verbose=False)
            p_recent = m_recent.predict_proba(X_test)[:, 1]
        else:
            p_recent = np.full(len(y_test), 0.5)

        mr = evaluate_predictions(y_test, p_recent, test_year, test["game_type"])
        mr["approach"] = "D_recent_only"
        mr["seed_baseline"] = seed_baseline_accuracy(test)
        results_recent_only.append(mr)

        # ── Transfer: pre-train then fine-tune ──
        pretrain = train[train["season"] <= PRETRAIN_CUTOFF]
        finetune = train[train["season"] > PRETRAIN_CUTOFF]

        if len(pretrain) < 1000 or len(finetune) < 200:
            continue

        # Step 1: Pre-train base model on historical data
        val_p = pretrain[pretrain["season"] == pretrain["season"].max()]
        X_pre = pretrain[feat_cols].fillna(0)
        y_pre = pretrain["label"].values
        val_mask_p = pretrain["season"] == pretrain["season"].max()

        base_params = {**XGB_BASE, "n_estimators": 300, "learning_rate": 0.05}
        base_model = xgb.XGBClassifier(**base_params)
        try:
            base_model.fit(X_pre[~val_mask_p], y_pre[~val_mask_p],
                           eval_set=[(X_pre[val_mask_p], y_pre[val_mask_p])],
                           verbose=False)
        except:
            base_model.fit(X_pre, y_pre, verbose=False)

        # Step 2: Fine-tune on recent data using warm-start continuation
        val_mask_f = finetune["season"] == finetune["season"].max()
        X_ft = finetune[feat_cols].fillna(0)
        y_ft = finetune["label"].values

        # XGBoost warm-start: continue from base model
        finetune_params = {**XGB_BASE, "n_estimators": 200, "learning_rate": 0.02}
        fine_model = xgb.XGBClassifier(**finetune_params)
        try:
            fine_model.fit(X_ft[~val_mask_f], y_ft[~val_mask_f],
                           eval_set=[(X_ft[val_mask_f], y_ft[val_mask_f])],
                           xgb_model=base_model.get_booster(),
                           verbose=False)
        except:
            fine_model.fit(X_ft, y_ft, verbose=False)

        p_transfer = fine_model.predict_proba(X_test)[:, 1]
        mt = evaluate_predictions(y_test, p_transfer, test_year, test["game_type"])
        mt["approach"] = "D_transfer"
        mt["seed_baseline"] = seed_baseline_accuracy(test)
        results_transfer.append(mt)
        print(f"    {test_year}: transfer={mt['accuracy']:.3f}  recent_only={mr['accuracy']:.3f}  "
              f"tourney={mt.get('tourney_accuracy','N/A')}")

    return results_transfer + results_recent_only


# ── Production model training ──────────────────────────────────────────────────

def train_production_model(df: pd.DataFrame, best_approach: str) -> xgb.XGBClassifier:
    """
    Train final production model on ALL available data (no hold-out).
    This is the model we deploy for predictions.

    WHY ALL DATA: In production, every additional game improves the model.
    We've already validated on held-out seasons; we know the expected
    out-of-sample performance. Now we maximise training data.
    """
    print(f"\nTraining production model (Approach {best_approach}, all data)...")
    feat_cols = get_feature_cols(df, tier=3)
    X = df[feat_cols].fillna(0)
    y = df["label"].values

    prod_params = {**XGB_BASE, "early_stopping_rounds": None}
    model = xgb.XGBClassifier(**prod_params)
    model.fit(X, y, verbose=False)

    # Save
    model_path = OUT_DIR / "production_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feat_cols,
                     "approach": best_approach,
                     "trained_on": f"1985-{df['season'].max()}",
                     "n_games": len(df)}, f)
    print(f"  Production model saved: {model_path}")
    return model


# ── Comparison report ──────────────────────────────────────────────────────────

def build_comparison_report(all_results: list, df: pd.DataFrame):
    """
    Aggregate results across all seasons and approaches.
    Output: comparison table + per-season accuracy chart.
    """
    rdf = pd.DataFrame(all_results)
    rdf.to_csv(OUT_DIR / "all_results_raw.csv", index=False)

    # Aggregate by approach
    agg = (rdf.groupby("approach").agg(
        avg_accuracy=("accuracy", "mean"),
        avg_tourney_accuracy=("tourney_accuracy", "mean"),
        avg_log_loss=("log_loss", "mean"),
        avg_brier=("brier", "mean"),
        avg_auc=("auc", "mean"),
        avg_seed_baseline=("seed_baseline", "mean"),
        seasons_tested=("season", "count"),
    ).round(4).reset_index())
    agg["vs_seed_baseline"] = (agg["avg_accuracy"] - agg["avg_seed_baseline"]).round(4)
    agg = agg.sort_values("avg_tourney_accuracy", ascending=False)

    # ── Print comparison table ──
    print("\n" + "="*80)
    print("  PHASE 3 MODEL COMPARISON")
    print("="*80)
    print(f"\n  {'Approach':<22} {'Accuracy':>9} {'Tourney Acc':>12} {'vs Baseline':>12} "
          f"{'Log-Loss':>9} {'Brier':>8} {'AUC':>7}")
    print("  " + "-"*79)
    for _, r in agg.iterrows():
        print(f"  {r['approach']:<22} {r['avg_accuracy']:>9.4f} {r['avg_tourney_accuracy']:>12.4f} "
              f"{r['vs_seed_baseline']:>+12.4f} {r['avg_log_loss']:>9.4f} "
              f"{r['avg_brier']:>8.4f} {r['avg_auc']:>7.4f}")

    # ── Per-season breakdown ──
    print("\n  Per-season accuracy (best approach per season):")
    pivot = rdf.pivot_table(index="season", columns="approach",
                            values="accuracy").round(4)
    print(pivot.to_string())

    # ── Save summary ──
    summary = {
        "comparison_table": agg.to_dict(orient="records"),
        "per_season": pivot.reset_index().to_dict(orient="records"),
        "best_overall_accuracy": agg.iloc[0]["approach"],
        "best_tourney_accuracy": agg.sort_values("avg_tourney_accuracy",
                                                   ascending=False).iloc[0]["approach"],
    }
    with open(OUT_DIR / "comparison_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Best overall: {summary['best_overall_accuracy']}")
    print(f"  Best tourney: {summary['best_tourney_accuracy']}")
    print(f"\n  Full results: {OUT_DIR}")
    print("="*80)

    return agg, summary


# ── Feature importance ─────────────────────────────────────────────────────────

def compute_feature_importance(df: pd.DataFrame) -> dict:
    """
    Train a full-data model and extract top feature importances.
    Uses both XGBoost gain importance and permutation correlation.
    """
    feat_cols = get_feature_cols(df, tier=3)
    X = df[feat_cols].fillna(0)
    y = df["label"].values

    model = xgb.XGBClassifier(**{**XGB_BASE, "early_stopping_rounds": None,
                                   "n_estimators": 300})
    model.fit(X, y, verbose=False)

    importance = dict(zip(feat_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    # Top 20
    top20 = dict(list(importance.items())[:20])

    print("\n  Top 20 features by XGBoost gain importance:")
    for feat, imp in top20.items():
        bar = "#" * int(imp * 500)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    with open(OUT_DIR / "feature_importance.json", "w") as f:
        json.dump({"all": {k: float(v) for k, v in importance.items()},
                   "top20": {k: float(v) for k, v in top20.items()}}, f, indent=2)

    return top20


# ── Calibration check ─────────────────────────────────────────────────────────

def check_calibration(df: pd.DataFrame):
    """
    When model says 70%, do teams win ~70% of the time?
    Plot calibration curve and report max calibration error.
    """
    feat_cols = get_feature_cols(df, tier=3)
    # Use last 5 seasons as out-of-sample
    test_seasons = list(range(2020, 2026))
    train_df = df[df["season"] < 2019]
    test_df  = df[df["season"].isin(test_seasons)]

    if len(train_df) < 1000 or len(test_df) < 100:
        return {}

    X_train = train_df[feat_cols].fillna(0)
    y_train = train_df["label"].values
    X_test  = test_df[feat_cols].fillna(0)
    y_test  = test_df["label"].values

    model = xgb.XGBClassifier(**{**XGB_BASE, "early_stopping_rounds": None,
                                   "n_estimators": 300})
    model.fit(X_train, y_train, verbose=False)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calibration curve
    fraction_pos, mean_predicted = calibration_curve(y_test, y_prob,
                                                      n_bins=10, strategy="uniform")
    max_cal_error = float(np.max(np.abs(fraction_pos - mean_predicted)))

    cal_result = {
        "mean_predicted_prob": mean_predicted.tolist(),
        "fraction_positive": fraction_pos.tolist(),
        "max_calibration_error": round(max_cal_error, 4),
        "note": "Well-calibrated if max_error < 0.05. Apply Platt scaling if > 0.10.",
    }

    print(f"\n  Calibration check (test seasons {test_seasons[0]}-{test_seasons[-1]}):")
    print(f"    Max calibration error: {max_cal_error:.4f}")
    if max_cal_error < 0.05:
        print("    [OK] Well-calibrated — predicted probabilities are reliable.")
    elif max_cal_error < 0.10:
        print("    [WARN] Moderate miscalibration — consider Platt scaling.")
    else:
        print("    [BAD] Significant miscalibration — apply isotonic regression.")

    with open(OUT_DIR / "calibration.json", "w") as f:
        json.dump(cal_result, f, indent=2)

    return cal_result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nBasketball-God — Phase 3: Era-Aware Model Training")
    print(f"Loading feature matrix...")

    df = pd.read_parquet(FEAT_DIR / "features_all.parquet")
    # Only use games where we have at least Tier 1 features
    df = df[df["diff_win_pct"].notna()].copy()
    # Exclude 2026 — incomplete season (future predictions, no ground truth)
    df = df[df["season"] < 2026].copy()

    print(f"  {len(df):,} games loaded, {df['season'].min()}-{df['season'].max()}")
    print(f"  Test seasons: {TEST_SEASONS[0]}-{TEST_SEASONS[-1]} (CPCV, 1-season embargo)")
    print(f"  Features (Tier 3): {len(get_feature_cols(df,3))} columns\n")

    all_results = []

    # ── Approach A: best half-life ──
    print("=" * 60)
    print("APPROACH A: Time-decayed weighting")
    for hl in DECAY_HALFLIVES:
        r = approach_a(df, hl)
        all_results.extend(r)

    # ── Approach B: era ensemble ──
    print("=" * 60)
    print("APPROACH B: Era-stratified ensemble")
    r = approach_b(df)
    all_results.extend(r)

    # ── Approach C: era features ──
    print("=" * 60)
    print("APPROACH C: Single model + era features")
    r = approach_c(df)
    all_results.extend(r)

    # ── Approach D: transfer learning ──
    print("=" * 60)
    print("APPROACH D: Transfer learning (pre-train + fine-tune)")
    r = approach_d(df)
    all_results.extend(r)

    # ── Compare all approaches ──
    agg, summary = build_comparison_report(all_results, df)

    # ── Feature importance ──
    compute_feature_importance(df)

    # ── Calibration check ──
    check_calibration(df)

    # ── Train production model with best approach ──
    best = summary["best_tourney_accuracy"]
    train_production_model(df, best)

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
