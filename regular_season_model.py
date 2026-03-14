"""
regular_season_model.py — Regular season prediction model for Basketball-God 2.0

Wraps the trained XGBoost + LR ensemble from Phase 6.
Accepts a feature dict (as produced by SeasonStatsStore) and returns
win probabilities + confidence.

Train the model first:
    python phase6_regular_season/train.py
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT    = Path(__file__).parent
MODEL_PATH = ROOT / "phase6_regular_season" / "output" / "regular_season_model.pkl"


class RegularSeasonModel:
    """
    Thin wrapper around the trained regular season ensemble.
    Designed to pair with SeasonStatsStore for daily predictions.
    """

    def __init__(self):
        self.xgb_model    = None
        self.lr_model     = None
        self.scaler       = None
        self.feature_cols: list[str] = []
        self.medians: dict = {}
        self.ready = False

    def load(self) -> "RegularSeasonModel":
        if not MODEL_PATH.exists():
            print(
                f"[RegularSeasonModel] Model not found at {MODEL_PATH}.\n"
                f"  Run: python phase6_regular_season/train.py",
                file=sys.stderr,
            )
            return self
        try:
            with open(MODEL_PATH, "rb") as f:
                bundle = pickle.load(f)
            self.xgb_model    = bundle["xgb_model"]
            self.lr_model     = bundle["lr_model"]
            self.scaler       = bundle["scaler"]
            self.feature_cols = bundle["feature_cols"]
            self.medians      = bundle.get("medians", {})
            self.ready = True
            print(f"[RegularSeasonModel] Loaded ({len(self.feature_cols)} features, "
                  f"{bundle.get('n_games', '?'):,} training games)")
        except Exception as e:
            print(f"[RegularSeasonModel] Load failed: {e}", file=sys.stderr)
        return self

    def predict(self, features: dict, home_name: str = "", away_name: str = "") -> dict:
        """
        Predict win probability from a feature dict (diff_* keys).

        features: dict produced by SeasonStatsStore.get_matchup_features()
        Returns standard prediction dict compatible with daily_predictor output.
        """
        base = {
            "home_team": home_name, "away_team": away_name,
            "home_display": home_name, "away_display": away_name,
            "prob_home_wins": 0.50, "prob_away_wins": 0.50,
            "model_spread": 0.0, "predicted_total": 145.0,
            "confidence": "low", "model_data_available": False,
            "model_type": "regular_season",
        }
        if not self.ready:
            return base

        try:
            import pandas as pd
            from daily_predictor import prob_to_spread

            # Build feature row — fill any missing with stored medians
            row = {}
            for col in self.feature_cols:
                val = features.get(col)
                if val is None:
                    val = self.medians.get(col, 0.0)
                row[col] = float(val) if val is not None else 0.0

            X = pd.DataFrame([row])[self.feature_cols]
            Xs = self.scaler.transform(X.values)

            p_xgb = float(self.xgb_model.predict_proba(Xs)[0, 1])
            p_lr  = float(self.lr_model.predict_proba(Xs)[0, 1])
            h_prob = 0.65 * p_xgb + 0.35 * p_lr
            a_prob = 1.0 - h_prob

            conf = ("high"   if max(h_prob, a_prob) >= 0.65 else
                    "medium" if max(h_prob, a_prob) >= 0.55 else "low")

            n_home = features.get("home_n_games", 0)
            n_away = features.get("away_n_games", 0)
            data_ok = (n_home >= 3 and n_away >= 3)

            return {
                "home_team":    home_name,
                "away_team":    away_name,
                "home_display": home_name,
                "away_display": away_name,
                "home_id":      features.get("home_id"),
                "away_id":      features.get("away_id"),
                "prob_home_wins":  round(h_prob, 3),
                "prob_away_wins":  round(a_prob, 3),
                "model_spread":    prob_to_spread(h_prob),
                "predicted_total": 145.0,
                "confidence":      conf,
                "model_data_available": data_ok,
                "model_type":     "regular_season",
                "home_n_games":   n_home,
                "away_n_games":   n_away,
            }

        except Exception as e:
            print(f"[RegularSeasonModel] predict error ({home_name} vs {away_name}): {e}",
                  file=sys.stderr)
            return base
