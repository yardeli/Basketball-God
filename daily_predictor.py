"""
daily_predictor.py — Real-time game predictions and betting edge calculations.

Integrates the trained NCAAModel with live market odds to find value bets.
"""

import math
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── Probability / odds utilities ──────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def prob_to_american(p: float) -> str:
    if p <= 0.001:
        return "+9999"
    if p >= 0.999:
        return "-9999"
    if p >= 0.5:
        return str(int(-(p / (1 - p)) * 100))
    return f"+{int(((1 - p) / p) * 100)}"


def american_to_implied(american: int) -> float:
    if american > 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def fmt_american(price: int) -> str:
    return f"+{price}" if price > 0 else str(price)


def prob_to_spread(h_prob: float, sigma: float = 11.0) -> float:
    """Convert home win probability to expected margin (+ = home favored)."""
    h_prob = max(0.001, min(0.999, h_prob))
    p = h_prob if h_prob >= 0.5 else 1 - h_prob
    t = math.sqrt(-2 * math.log(p))
    z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
        1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
    )
    return round(sigma * (z if h_prob >= 0.5 else -z), 1)


def cover_prob(model_spread: float, market_line: float, sigma: float = 10.0) -> float:
    """Prob home covers market_line given model's predicted spread (+ = home favored)."""
    return _normal_cdf((model_spread - market_line) / sigma)


def total_over_prob(model_total: float, market_line: float, sigma: float = 15.0) -> float:
    return 1 - _normal_cdf((market_line - model_total) / sigma)


def _signal(edge: float) -> str:
    if edge >= 0.15:
        return "STRONG BET"
    if edge >= 0.02:
        return "MODEL HIGHER"
    if edge <= -0.02:
        return "MARKET HIGHER"
    return "AGREE"


# ── DailyPredictor ────────────────────────────────────────────────────────────

class DailyPredictor:
    """Wraps the trained NCAAModel for real-time game win-probability predictions."""

    def __init__(self):
        self.model = None
        self.elo   = None
        self.ready = False
        self._team_index: dict[str, int] = {}   # normalized name → team_id
        self._id_to_name: dict[int, str] = {}

    def load(self) -> "DailyPredictor":
        try:
            from model_training import NCAAModel
            from elo import EloSystem, build_elo_ratings
            from config import PROCESSED_DIR
            import pandas as pd

            self.model = NCAAModel()
            self.model.load()

            all_games_path = PROCESSED_DIR / "all_games.csv"
            if all_games_path.exists():
                games = pd.read_csv(all_games_path)
                self.elo, _ = build_elo_ratings(games)
                # Build team-name lookup from games data
                for col_id, col_name in [("home_id", "home_team"), ("away_id", "away_team")]:
                    if col_id in games.columns and col_name in games.columns:
                        pairs = games[[col_id, col_name]].drop_duplicates()
                        for _, row in pairs.iterrows():
                            tid  = int(row[col_id])
                            name = str(row[col_name])
                            self._id_to_name[tid] = name
                            self._team_index[name.lower()] = tid
            else:
                from elo import EloSystem
                self.elo = EloSystem()

            self.ready = True
        except Exception as e:
            print(f"[DailyPredictor] load failed: {e}", file=sys.stderr)
            self.ready = False
        return self

    def _find_team_id(self, name: str) -> Optional[int]:
        key = name.lower().strip()
        if key in self._team_index:
            return self._team_index[key]
        # Partial / fuzzy match
        for k, tid in self._team_index.items():
            if key in k or k in key:
                return tid
        return None

    def predict_game(self, home: str, away: str) -> dict:
        base = {
            "home_team": home, "away_team": away,
            "home_display": home, "away_display": away,
            "home_id": None,  "away_id": None,
            "prob_home_wins": 0.50, "prob_away_wins": 0.50,
            "model_spread": 0.0,   "predicted_total": 145.0,
            "confidence": "low",   "model_data_available": False,
        }
        if not self.ready:
            return base

        try:
            import pandas as pd
            from config import MATCHUP_FEATURES

            home_id = self._find_team_id(home)
            away_id = self._find_team_id(away)
            home_elo = self.elo.get_rating(home_id) if home_id else 1500.0
            away_elo = self.elo.get_rating(away_id) if away_id else 1500.0

            feature_row = {f: 0 for f in MATCHUP_FEATURES}
            feature_row["elo_diff"]  = home_elo - away_elo
            feature_row["home_away"] = 1

            X = pd.DataFrame([feature_row])[self.model.feature_names]
            h_prob = float(self.model.predict_proba(X)[0])
            a_prob = 1.0 - h_prob

            conf = ("high"   if max(h_prob, a_prob) >= 0.65 else
                    "medium" if max(h_prob, a_prob) >= 0.55 else "low")

            home_display = self._id_to_name.get(home_id, home) if home_id else home
            away_display = self._id_to_name.get(away_id, away) if away_id else away

            return {
                "home_team": home, "away_team": away,
                "home_display": home_display, "away_display": away_display,
                "home_id": home_id, "away_id": away_id,
                "prob_home_wins": round(h_prob, 3),
                "prob_away_wins": round(a_prob, 3),
                "model_spread":   prob_to_spread(h_prob),
                "predicted_total": 145.0,
                "confidence": conf,
                "model_data_available": True,
            }
        except Exception as e:
            print(f"[DailyPredictor] predict_game error ({home} vs {away}): {e}",
                  file=sys.stderr)
            return base


# ── Edge calculations ─────────────────────────────────────────────────────────

def calculate_game_edges(prediction: dict, game: dict) -> list[dict]:
    """
    Compute betting edges for all available markets.
    Returns list of bet dicts sorted by edge descending.
    """
    bets = []
    h_prob       = prediction.get("prob_home_wins", 0.50)
    a_prob       = prediction.get("prob_away_wins", 0.50)
    model_spread = prediction.get("model_spread", 0.0)
    pred_total   = prediction.get("predicted_total") or 145.0
    home         = game.get("home_team", "")
    away         = game.get("away_team", "")

    # ── Moneyline ─────────────────────────────────────────────────────────────
    h2h = game.get("h2h", {})
    for team, model_prob, side in [(home, h_prob, "home"), (away, a_prob, "away")]:
        mkt = h2h.get(side)
        if not mkt:
            continue
        price       = mkt["price"]
        raw_implied = american_to_implied(price)
        edge        = model_prob - raw_implied
        bets.append({
            "bet_type":       "Moneyline",
            "bet":            f"{team} to win",
            "pick_team":      team,
            "line":           None,
            "model_odds":     f"{prob_to_american(model_prob)} ({model_prob*100:.1f}%)",
            "market_odds":    fmt_american(price),
            "market_odds_raw": price,
            "market_implied": round(raw_implied, 4),
            "edge":           round(edge, 4),
            "signal":         _signal(edge),
            "best_book":      mkt.get("best_book", ""),
        })

    # ── Spread ────────────────────────────────────────────────────────────────
    spread_mkt = game.get("spread", {})
    for team, is_home in [(home, True), (away, False)]:
        side = "home" if is_home else "away"
        mkt  = spread_mkt.get(side)
        if not mkt or mkt.get("point") is None:
            continue
        market_line  = mkt["point"]   # negative = fav must cover; positive = dog gets points
        price        = mkt["price"]
        raw_implied  = american_to_implied(price)
        adj_spread   = model_spread if is_home else -model_spread
        cov_prob     = cover_prob(adj_spread, -market_line)
        edge         = cov_prob - raw_implied
        line_str     = fmt_american(int(market_line)) if market_line == int(market_line) else (
                       f"+{market_line}" if market_line > 0 else str(market_line))
        mdl_str      = fmt_american(int(adj_spread)) if adj_spread == int(adj_spread) else (
                       f"+{adj_spread}" if adj_spread > 0 else str(adj_spread))
        bets.append({
            "bet_type":       "Spread",
            "bet":            f"{team} {line_str} (mdl {mdl_str})",
            "pick_team":      team,
            "line":           market_line,
            "model_odds":     f"{prob_to_american(cov_prob)} ({cov_prob*100:.1f}%)",
            "market_odds":    fmt_american(price),
            "market_odds_raw": price,
            "market_implied": round(raw_implied, 4),
            "edge":           round(edge, 4),
            "signal":         _signal(edge),
            "best_book":      mkt.get("best_book", ""),
        })

    # ── Totals ────────────────────────────────────────────────────────────────
    total_mkt = game.get("total", {})
    over_mkt  = total_mkt.get("over")
    under_mkt = total_mkt.get("under")
    if over_mkt and over_mkt.get("point") is not None:
        market_line  = over_mkt["point"]
        over_price   = over_mkt["price"]
        under_price  = under_mkt["price"] if under_mkt else over_price
        over_implied  = american_to_implied(over_price)
        under_implied = american_to_implied(under_price)
        over_prob    = total_over_prob(pred_total, market_line)
        under_prob   = 1.0 - over_prob
        bets.append({
            "bet_type":       "Total Over",
            "bet":            f"Over {market_line} (mdl {pred_total:.0f})",
            "pick_team":      f"Over {market_line}",
            "line":           market_line,
            "model_odds":     f"{prob_to_american(over_prob)} ({over_prob*100:.1f}%)",
            "market_odds":    fmt_american(over_price),
            "market_odds_raw": over_price,
            "market_implied": round(over_implied, 4),
            "edge":           round(over_prob - over_implied, 4),
            "signal":         _signal(over_prob - over_implied),
            "best_book":      over_mkt.get("best_book", ""),
        })
        bets.append({
            "bet_type":       "Total Under",
            "bet":            f"Under {market_line} (mdl {pred_total:.0f})",
            "pick_team":      f"Under {market_line}",
            "line":           market_line,
            "model_odds":     f"{prob_to_american(under_prob)} ({under_prob*100:.1f}%)",
            "market_odds":    fmt_american(under_price),
            "market_odds_raw": under_price,
            "market_implied": round(under_implied, 4),
            "edge":           round(under_prob - under_implied, 4),
            "signal":         _signal(under_prob - under_implied),
            "best_book":      under_mkt.get("best_book", "") if under_mkt else "",
        })

    bets.sort(key=lambda b: -(b.get("edge") or 0))
    return bets
