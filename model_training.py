"""
Model Training — XGBoost + Logistic Regression ensemble for any D1 game.

Walk-forward validation:
  - Train on seasons before the target season
  - Never use future data to predict past games
  - Ensemble: 65% XGBoost + 35% Logistic Regression
"""
import json
import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from xgboost import XGBClassifier
import joblib

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from feature_engineering import get_feature_matrix, normalize_features


class NCAAModel:
    """Ensemble model for NCAA D1 basketball game predictions."""

    def __init__(self):
        self.xgb_model = None
        self.lr_model = None
        self.scaler = None
        self.feature_names = None
        self.feature_importances = None
        self.training_metrics = {}

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Train ensemble on matchup features."""
        X_train, y_train = get_feature_matrix(train_df)
        self.feature_names = list(X_train.columns)

        # Normalize
        if val_df is not None and len(val_df) > 0:
            X_val, y_val = get_feature_matrix(val_df)
            X_train_s, self.scaler, X_val_s = normalize_features(X_train, X_val)
        else:
            X_train_s, self.scaler = normalize_features(X_train)
            X_val_s, y_val = None, None

        # XGBoost
        self.xgb_model = XGBClassifier(**config.XGB_PARAMS)
        self.xgb_model.fit(X_train_s, y_train, verbose=False)

        # Logistic Regression
        self.lr_model = LogisticRegression(**config.LOGISTIC_PARAMS)
        self.lr_model.fit(X_train_s, y_train)

        # Feature importances
        self.feature_importances = dict(
            zip(self.feature_names, self.xgb_model.feature_importances_)
        )

        # Compute training metrics
        train_proba = self.predict_proba(X_train)
        train_preds = (train_proba > 0.5).astype(int)
        self.training_metrics["train_accuracy"] = float(accuracy_score(y_train, train_preds))
        self.training_metrics["train_games"] = len(y_train)

        if X_val_s is not None:
            val_proba = self.predict_proba(X_val)
            val_preds = (val_proba > 0.5).astype(int)
            self.training_metrics["val_accuracy"] = float(accuracy_score(y_val, val_preds))
            self.training_metrics["val_log_loss"] = float(log_loss(y_val, val_proba))
            self.training_metrics["val_brier"] = float(brier_score_loss(y_val, val_proba))
            self.training_metrics["val_games"] = len(y_val)

        print(f"  Train accuracy: {self.training_metrics['train_accuracy']:.1%} "
              f"({self.training_metrics['train_games']} games)")
        if "val_accuracy" in self.training_metrics:
            print(f"  Val accuracy:   {self.training_metrics['val_accuracy']:.1%} "
                  f"({self.training_metrics['val_games']} games)")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of home team winning (ensemble)."""
        if isinstance(X, pd.DataFrame):
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index,
            )
        else:
            X_scaled = self.scaler.transform(X)

        xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
        lr_proba = self.lr_model.predict_proba(X_scaled)[:, 1]

        ensemble = (
            config.ENSEMBLE_WEIGHT_XGB * xgb_proba +
            config.ENSEMBLE_WEIGHT_LR * lr_proba
        )
        return ensemble

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict winner (1 = home, 0 = away)."""
        return (self.predict_proba(X) > 0.5).astype(int)

    def save(self, path=None):
        """Save model to disk."""
        if path is None:
            path = config.MODELS_DIR
        path = config.MODELS_DIR
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.xgb_model, path / "xgb_model.pkl")
        joblib.dump(self.lr_model, path / "lr_model.pkl")
        joblib.dump(self.scaler, path / "scaler.pkl")

        meta = {
            "feature_names": self.feature_names,
            "feature_importances": {k: round(float(v), 6) for k, v in self.feature_importances.items()},
            "training_metrics": self.training_metrics,
        }
        with open(path / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[Model] Saved to {path}")

    def load(self, path=None):
        """Load model from disk."""
        if path is None:
            path = config.MODELS_DIR

        self.xgb_model = joblib.load(path / "xgb_model.pkl")
        self.lr_model = joblib.load(path / "lr_model.pkl")
        self.scaler = joblib.load(path / "scaler.pkl")

        with open(path / "model_meta.json") as f:
            meta = json.load(f)
        self.feature_names = meta["feature_names"]
        self.feature_importances = meta.get("feature_importances", {})
        self.training_metrics = meta.get("training_metrics", {})

        print(f"[Model] Loaded from {path}")


def train_walk_forward(matchups_df: pd.DataFrame) -> dict:
    """
    Walk-forward validation across seasons.
    For each season S, train on all seasons < S, predict S.
    """
    print("\n" + "=" * 60)
    print("  WALK-FORWARD TRAINING")
    print("=" * 60)

    seasons = sorted(matchups_df["season"].unique())
    min_train = 3  # Need at least 3 seasons to start testing

    results = []

    for i, test_season in enumerate(seasons):
        if i < min_train:
            continue

        train_seasons = seasons[:i]
        train_data = matchups_df[matchups_df["season"].isin(train_seasons)]
        test_data = matchups_df[matchups_df["season"] == test_season]

        if len(test_data) == 0:
            continue

        print(f"\n--- Season {test_season} (train: {train_seasons[0]}-{train_seasons[-1]}, "
              f"{len(train_data)} games) ---")

        model = NCAAModel()
        model.train(train_data)

        # Evaluate on test season
        X_test, y_test = get_feature_matrix(test_data)
        proba = model.predict_proba(X_test)
        preds = (proba > 0.5).astype(int)

        acc = float(accuracy_score(y_test, preds))
        ll = float(log_loss(y_test, proba))
        brier = float(brier_score_loss(y_test, proba))

        # Home-win baseline
        home_baseline = float(y_test.mean())

        results.append({
            "season": int(test_season),
            "train_seasons": f"{train_seasons[0]}-{train_seasons[-1]}",
            "n_train": len(train_data),
            "n_test": len(test_data),
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4),
            "home_win_baseline": round(home_baseline, 4),
            "improvement": round(acc - home_baseline, 4),
        })

        print(f"  Accuracy: {acc:.1%} (home baseline: {home_baseline:.1%}, "
              f"improvement: {acc - home_baseline:+.1%})")
        print(f"  Log loss: {ll:.4f}, Brier: {brier:.4f}")

    # Summary
    if results:
        avg_acc = np.mean([r["accuracy"] for r in results])
        avg_baseline = np.mean([r["home_win_baseline"] for r in results])
        avg_improvement = np.mean([r["improvement"] for r in results])

        print("\n" + "=" * 60)
        print("  WALK-FORWARD SUMMARY")
        print("=" * 60)
        print(f"  Seasons tested: {len(results)}")
        print(f"  Avg accuracy: {avg_acc:.1%}")
        print(f"  Avg home baseline: {avg_baseline:.1%}")
        print(f"  Avg improvement: {avg_improvement:+.1%}")

    return {
        "seasons": results,
        "summary": {
            "n_seasons": len(results),
            "avg_accuracy": round(float(avg_acc), 4) if results else 0,
            "avg_baseline": round(float(avg_baseline), 4) if results else 0,
            "avg_improvement": round(float(avg_improvement), 4) if results else 0,
        },
    }


def train_production_model(matchups_df: pd.DataFrame) -> NCAAModel:
    """
    Train the production model on ALL available data.
    Use the most recent season as validation.
    """
    print("\n[Model] Training production model on all data...")

    seasons = sorted(matchups_df["season"].unique())
    val_season = seasons[-1]
    train_seasons = seasons[:-1]

    train_data = matchups_df[matchups_df["season"].isin(train_seasons)]
    val_data = matchups_df[matchups_df["season"] == val_season]

    model = NCAAModel()
    model.train(train_data, val_data)
    model.save()

    return model


if __name__ == "__main__":
    print("Model training module — use via pipeline or predict.py")
