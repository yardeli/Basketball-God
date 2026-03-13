"""
Feature Engineering — Build matchup features from game logs, Elo, and text signals.

For any D1 game, we compute:
  1. Rolling team stats (last N games)
  2. Elo ratings (historical strength)
  3. Context features (home/away, rest days, conference game)
  4. Text signals (injury impact, sentiment)
  5. Matchup differentials (team1 - team2 for all features)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def compute_rolling_stats(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling per-team statistics from game logs.
    Each game gets the team's stats BEFORE that game (no lookahead).
    """
    print("[Features] Computing rolling team stats...")

    games = games_df.sort_values("date").copy()

    # We'll build team stat histories from both sides of each game
    team_games: dict[int, list[dict]] = {}

    rolling_rows = []

    for idx, game in games.iterrows():
        home_id = int(game["home_id"])
        away_id = int(game["away_id"])

        # Get rolling stats BEFORE this game
        home_stats = _get_team_rolling(team_games.get(home_id, []))
        away_stats = _get_team_rolling(team_games.get(away_id, []))

        rolling_rows.append({
            "idx": idx,
            # Home team rolling stats
            "home_points_pg": home_stats["points_pg"],
            "home_opp_points_pg": home_stats["opp_points_pg"],
            "home_margin_pg": home_stats["margin_pg"],
            "home_win_pct": home_stats["win_pct"],
            "home_momentum": home_stats["momentum"],
            "home_games_played": home_stats["games_played"],
            # Away team rolling stats
            "away_points_pg": away_stats["points_pg"],
            "away_opp_points_pg": away_stats["opp_points_pg"],
            "away_margin_pg": away_stats["margin_pg"],
            "away_win_pct": away_stats["win_pct"],
            "away_momentum": away_stats["momentum"],
            "away_games_played": away_stats["games_played"],
        })

        # Record this game for both teams
        home_game = {
            "points": int(game["home_score"]),
            "opp_points": int(game["away_score"]),
            "win": int(game["home_score"] > game["away_score"]),
        }
        away_game = {
            "points": int(game["away_score"]),
            "opp_points": int(game["home_score"]),
            "win": int(game["away_score"] > game["home_score"]),
        }

        if home_id not in team_games:
            team_games[home_id] = []
        team_games[home_id].append(home_game)

        if away_id not in team_games:
            team_games[away_id] = []
        team_games[away_id].append(away_game)

    rolling_df = pd.DataFrame(rolling_rows).set_index("idx")
    result = games.join(rolling_df)

    print(f"  Computed rolling stats for {len(result)} games")
    return result


def _get_team_rolling(game_history: list[dict]) -> dict:
    """Compute rolling averages from a team's game history."""
    if not game_history:
        return {
            "points_pg": 70.0,  # D1 average defaults
            "opp_points_pg": 70.0,
            "margin_pg": 0.0,
            "win_pct": 0.5,
            "momentum": 0.0,
            "games_played": 0,
        }

    window = config.ROLLING_WINDOW
    recent = game_history[-window:]
    momentum_games = game_history[-config.MOMENTUM_WINDOW:]

    points = [g["points"] for g in recent]
    opp_points = [g["opp_points"] for g in recent]
    wins = [g["win"] for g in recent]
    momentum_wins = [g["win"] for g in momentum_games]

    return {
        "points_pg": np.mean(points),
        "opp_points_pg": np.mean(opp_points),
        "margin_pg": np.mean([p - o for p, o in zip(points, opp_points)]),
        "win_pct": np.mean([g["win"] for g in game_history]),  # Full season win pct
        "momentum": np.mean(momentum_wins) - np.mean(wins) if len(game_history) > config.MOMENTUM_WINDOW else 0.0,
        "games_played": len(game_history),
    }


def create_matchup_features(games_df: pd.DataFrame,
                             text_signals: dict = None) -> pd.DataFrame:
    """
    Create matchup differential features for each game.
    Every feature is team1 - team2 (home - away by default).
    """
    print("[Features] Creating matchup features...")

    games = games_df.copy()

    # Injury and sentiment signals (default to 0 if not available)
    injuries = text_signals.get("injuries", {}) if text_signals else {}
    sentiments = text_signals.get("sentiment", {}) if text_signals else {}

    features = []
    for _, game in games.iterrows():
        home_id = int(game.get("home_id", 0))
        away_id = int(game.get("away_id", 0))
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        matchup = {
            # Identifiers (not features)
            "game_id": game.get("game_id", ""),
            "season": game.get("season", 0),
            "date": game.get("date", ""),
            "home_id": home_id,
            "away_id": away_id,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": game.get("home_score", 0),
            "away_score": game.get("away_score", 0),

            # Target
            "home_win": int(game.get("home_win", game.get("home_score", 0) > game.get("away_score", 0))),

            # ── Matchup differentials ──

            # Elo
            "elo_diff": game.get("home_elo", 1500) - game.get("away_elo", 1500),

            # Net rating (points margin per game)
            "net_rating_diff": (
                game.get("home_margin_pg", 0) - game.get("away_margin_pg", 0)
            ),

            # Win percentage
            "win_pct_diff": game.get("home_win_pct", 0.5) - game.get("away_win_pct", 0.5),

            # Offensive output
            "points_pg_diff": game.get("home_points_pg", 70) - game.get("away_points_pg", 70),

            # Defensive strength (lower opp points = better)
            "opp_points_pg_diff": game.get("home_opp_points_pg", 70) - game.get("away_opp_points_pg", 70),

            # Momentum (recent form vs season average)
            "momentum_diff": game.get("home_momentum", 0) - game.get("away_momentum", 0),

            # Context
            "home_away": 0 if game.get("neutral_site", 0) else 1,
            "rest_days_diff": 0,  # Would need schedule data to compute
            "conf_game": int(game.get("conference_game", 0)),

            # Conference power
            "power_conf_diff": 0,  # Computed below if conference data available

            # Text signals
            "injury_impact_diff": injuries.get(home_team, 0) - injuries.get(away_team, 0),
            "sentiment_diff": sentiments.get(home_team, 0) - sentiments.get(away_team, 0),

            # Games played (experience this season)
            "home_games_played": game.get("home_games_played", 0),
            "away_games_played": game.get("away_games_played", 0),
        }

        features.append(matchup)

    result = pd.DataFrame(features)
    print(f"  Created {len(result)} matchup feature rows")
    return result


def get_feature_matrix(df: pd.DataFrame, features: list[str] = None) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target y."""
    if features is None:
        features = config.MATCHUP_FEATURES

    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing features: {missing}")

    X = df[available].copy()
    y = df["home_win"].copy()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


def normalize_features(X_train: pd.DataFrame,
                        X_other: pd.DataFrame = None) -> tuple:
    """Normalize features. Fit on train only."""
    scaler = StandardScaler()
    cols = X_train.columns

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=cols, index=X_train.index
    )

    if X_other is not None:
        X_other_scaled = pd.DataFrame(
            scaler.transform(X_other), columns=cols, index=X_other.index
        )
        return X_train_scaled, scaler, X_other_scaled

    return X_train_scaled, scaler


if __name__ == "__main__":
    print("Feature engineering module — use via pipeline or predict.py")
