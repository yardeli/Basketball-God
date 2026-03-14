"""
phase7_v2/clv_tracker.py — Closing Line Value (CLV) Tracker
=============================================================
Logs model predictions alongside opening market lines.
After games close, records the closing line and computes CLV.

CLV = model_implied_prob - closing_line_implied_prob
  > 0 : Model was on the right side of line movement (positive EV)
  < 0 : Line moved against the model's prediction

NOTE ON HISTORICAL BACKTESTING
The Odds API does not provide historical closing line data, and this system
did not log predictions + lines historically. Therefore:
  - CLV tracking begins from the date this module is deployed
  - Historical backtest CLV is estimated as:
      estimated_CLV = model_prob - elo_baseline_prob
    where elo_baseline represents a naive market (Elo-implied probability).
    This measures how much value the full model adds over a simple Elo market.
  - True CLV tracking accumulates over time as this system logs predictions.

Usage:
    tracker = CLVTracker()
    tracker.log_prediction("game_id_123", "Duke", "UNC", model_prob=0.68, market_odds=-200)
    # Later, when line closes:
    tracker.log_closing_line("game_id_123", closing_odds=-240)
    tracker.save()
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
CLV_LOG_PATH = ROOT / "data" / "cache" / "clv_log.json"


def american_to_prob(american: int) -> float:
    """Convert American odds to implied probability."""
    if american > 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def remove_vig(home_prob: float, away_prob: float) -> tuple[float, float]:
    """Remove bookmaker vig (overround) from implied probabilities."""
    total = home_prob + away_prob
    return home_prob / total, away_prob / total


def prob_to_american(p: float) -> int:
    """Convert probability to American odds."""
    p = max(0.001, min(0.999, p))
    if p >= 0.5:
        return int(-(p / (1 - p)) * 100)
    return int(((1 - p) / p) * 100)


class CLVTracker:
    """
    Logs and tracks Closing Line Value for every prediction.

    Log format per entry:
    {
        "game_id":          str,
        "home_team":        str,
        "away_team":        str,
        "game_date":        str (ISO),
        "logged_at":        str (ISO),
        "model_home_prob":  float,
        "model_away_prob":  float,
        "model_home_odds":  int (American),
        "opening_home_odds": int or null,
        "opening_away_odds": int or null,
        "opening_home_implied": float or null,
        "closing_home_odds": int or null,
        "closing_away_odds": int or null,
        "closing_home_implied": float or null,
        "clv_home":         float or null,   # model_prob - closing_implied (no-vig)
        "result_home_won":  bool or null,    # filled after game
        "beat_closing":     bool or null,    # was model on right side of movement?
    }
    """

    def __init__(self):
        self.log: list[dict] = []
        self._load()

    def _load(self):
        if CLV_LOG_PATH.exists():
            try:
                self.log = json.loads(CLV_LOG_PATH.read_text(encoding="utf-8"))
            except Exception:
                self.log = []

    def save(self):
        CLV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CLV_LOG_PATH.write_text(json.dumps(self.log, indent=2), encoding="utf-8")

    def log_prediction(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        model_home_prob: float,
        opening_home_odds: int | None = None,
        opening_away_odds: int | None = None,
        game_date: str | None = None,
    ) -> dict:
        """
        Record a prediction at prediction time (before game).
        Call this when generating daily predictions.
        """
        # Remove vig if both sides available
        opening_home_implied = None
        if opening_home_odds is not None and opening_away_odds is not None:
            raw_h = american_to_prob(opening_home_odds)
            raw_a = american_to_prob(opening_away_odds)
            opening_home_implied, _ = remove_vig(raw_h, raw_a)

        entry = {
            "game_id":              game_id,
            "home_team":            home_team,
            "away_team":            away_team,
            "game_date":            game_date or datetime.now(timezone.utc).date().isoformat(),
            "logged_at":            datetime.now(timezone.utc).isoformat(),
            "model_home_prob":      round(model_home_prob, 4),
            "model_away_prob":      round(1 - model_home_prob, 4),
            "model_home_odds":      prob_to_american(model_home_prob),
            "opening_home_odds":    opening_home_odds,
            "opening_away_odds":    opening_away_odds,
            "opening_home_implied": round(opening_home_implied, 4) if opening_home_implied else None,
            "closing_home_odds":    None,
            "closing_away_odds":    None,
            "closing_home_implied": None,
            "clv_home":             None,
            "result_home_won":      None,
            "beat_closing":         None,
            "model_version":        "v2.0",
        }
        # Avoid duplicate entries
        existing = [e for e in self.log if e["game_id"] == game_id]
        if not existing:
            self.log.append(entry)
        return entry

    def log_closing_line(
        self,
        game_id: str,
        closing_home_odds: int,
        closing_away_odds: int,
    ) -> dict | None:
        """
        Record the closing market line (call just before tip-off).
        Computes CLV = model_prob - closing_implied_prob (no-vig).
        """
        entry = next((e for e in self.log if e["game_id"] == game_id), None)
        if not entry:
            return None

        raw_h = american_to_prob(closing_home_odds)
        raw_a = american_to_prob(closing_away_odds)
        closing_home_implied, _ = remove_vig(raw_h, raw_a)

        entry["closing_home_odds"]    = closing_home_odds
        entry["closing_away_odds"]    = closing_away_odds
        entry["closing_home_implied"] = round(closing_home_implied, 4)
        entry["clv_home"]             = round(entry["model_home_prob"] - closing_home_implied, 4)

        # Beat closing: model was on the right side of line movement
        if entry["opening_home_implied"] is not None:
            model_liked_home = entry["model_home_prob"] > entry["opening_home_implied"]
            line_moved_home  = closing_home_implied > entry["opening_home_implied"]
            entry["beat_closing"] = (model_liked_home == line_moved_home)

        return entry

    def log_result(self, game_id: str, home_won: bool) -> dict | None:
        """Record the game result after it finishes."""
        entry = next((e for e in self.log if e["game_id"] == game_id), None)
        if entry:
            entry["result_home_won"] = home_won
        return entry

    def get_summary(self, min_games: int = 10) -> dict:
        """Compute CLV summary statistics."""
        with_clv    = [e for e in self.log if e["clv_home"] is not None]
        with_result = [e for e in self.log if e["result_home_won"] is not None]
        beat_close  = [e for e in with_clv if e.get("beat_closing") is True]

        avg_clv = (sum(e["clv_home"] for e in with_clv) / len(with_clv)) if with_clv else None

        # Accuracy
        correct = [e for e in with_result
                   if (e["model_home_prob"] > 0.5) == bool(e["result_home_won"])]
        accuracy = len(correct) / len(with_result) if with_result else None

        return {
            "n_predictions":    len(self.log),
            "n_with_clv":       len(with_clv),
            "n_with_result":    len(with_result),
            "avg_clv":          round(avg_clv, 4) if avg_clv is not None else None,
            "avg_clv_pct":      f"{avg_clv*100:+.2f}%" if avg_clv is not None else "N/A",
            "beat_close_rate":  round(len(beat_close) / len(with_clv), 3) if with_clv else None,
            "prediction_acc":   round(accuracy, 3) if accuracy is not None else None,
            "note":             (
                "CLV tracking active from deployment date. "
                "Historical CLV estimated as model_prob - elo_baseline_prob in A/B backtest."
            ),
        }

    def estimate_retroactive_clv(
        self,
        model_prob: float,
        elo_diff: float,
        sigma: float = 400.0,
    ) -> float:
        """
        Retroactive CLV estimate: model vs naive Elo-only market.
        Positive = model adds value over a market priced purely on Elo.

        elo_diff: home_elo - away_elo (from EloSystem)
        sigma: Elo spread parameter (400 = standard)
        """
        elo_prob = 1.0 / (1.0 + math.pow(10, -elo_diff / sigma))
        return round(model_prob - elo_prob, 4)


# ── Convenience function for server.py integration ────────────────────────────

_tracker_instance: CLVTracker | None = None


def get_tracker() -> CLVTracker:
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CLVTracker()
    return _tracker_instance


def log_daily_prediction(
    game: dict,
    prediction: dict,
    market_odds: dict | None = None,
) -> None:
    """
    Convenience wrapper — call from daily_predictor.py after each prediction.
    game: from The Odds API (has game_id, home_team, away_team, commence_time)
    prediction: from DailyPredictor.predict_game()
    market_odds: h2h dict with home/away prices
    """
    tracker = get_tracker()
    h2h = market_odds or {}
    home_odds = h2h.get("home", {}).get("price") if h2h else None
    away_odds = h2h.get("away", {}).get("price") if h2h else None

    tracker.log_prediction(
        game_id=str(game.get("id", "")),
        home_team=game.get("home_team", ""),
        away_team=game.get("away_team", ""),
        model_home_prob=prediction.get("prob_home_wins", 0.5),
        opening_home_odds=int(home_odds) if home_odds else None,
        opening_away_odds=int(away_odds) if away_odds else None,
        game_date=game.get("commence_time", "")[:10],
    )
    tracker.save()
