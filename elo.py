"""
Elo Rating System — Track team strength across an entire season.

Each team starts at 1500. After each game, ratings adjust based on:
  - Win/loss outcome
  - Margin of victory (capped to prevent blowout distortion)
  - Home court advantage (~65 Elo points)
  - Between seasons, ratings revert 33% toward the mean

This gives us a real-time strength metric that captures:
  - Quality of wins (beating a 1700 team matters more than beating a 1300 team)
  - Recent form (ratings update after every game)
  - Historical context (carryover between seasons)
"""
import math
import pandas as pd
import config


class EloSystem:
    """Maintains Elo ratings for all D1 teams across seasons."""

    def __init__(self):
        self.ratings: dict[int, float] = {}  # team_id -> rating
        self.history: list[dict] = []  # game-by-game rating changes

    def get_rating(self, team_id: int) -> float:
        """Get current rating for a team (default: initial rating)."""
        return self.ratings.get(team_id, config.ELO_INITIAL)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected probability of team A winning."""
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def margin_of_victory_multiplier(self, margin: int, elo_diff: float) -> float:
        """
        Scale K-factor by margin of victory.
        Uses log to dampen blowouts. Also adjusts for elo_diff to prevent
        autocorrelation (strong teams that win big shouldn't get extra credit).
        """
        abs_margin = abs(margin)
        mov = math.log(max(abs_margin, 1) + 1)
        # Reduce multiplier when favorite wins big (expected outcome)
        if elo_diff > 0 and margin > 0:
            mov *= 2.2 / (2.2 + 0.001 * elo_diff)
        elif elo_diff < 0 and margin < 0:
            mov *= 2.2 / (2.2 + 0.001 * abs(elo_diff))
        return mov

    def update(self, home_id: int, away_id: int, home_score: int, away_score: int,
               neutral_site: bool = False) -> dict:
        """
        Update ratings after a game. Returns rating changes.
        """
        home_rating = self.get_rating(home_id)
        away_rating = self.get_rating(away_id)

        # Home court advantage (not on neutral sites)
        hca = 0 if neutral_site else config.ELO_HOME_ADVANTAGE
        adj_home = home_rating + hca

        # Expected scores
        exp_home = self.expected_score(adj_home, away_rating)
        exp_away = 1.0 - exp_home

        # Actual outcome
        if home_score > away_score:
            actual_home, actual_away = 1.0, 0.0
        elif away_score > home_score:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        # Margin of victory multiplier
        margin = home_score - away_score
        mov = self.margin_of_victory_multiplier(margin, adj_home - away_rating)

        # Rating changes
        k = config.ELO_K * mov
        home_change = k * (actual_home - exp_home)
        away_change = k * (actual_away - exp_away)

        # Apply
        self.ratings[home_id] = home_rating + home_change
        self.ratings[away_id] = away_rating + away_change

        record = {
            "home_id": home_id,
            "away_id": away_id,
            "home_pre": home_rating,
            "away_pre": away_rating,
            "home_post": self.ratings[home_id],
            "away_post": self.ratings[away_id],
            "home_change": home_change,
            "away_change": away_change,
            "home_expected": round(exp_home, 4),
            "margin": margin,
        }
        self.history.append(record)
        return record

    def new_season(self):
        """
        Revert all ratings toward the mean between seasons.
        This prevents stale ratings and accounts for roster turnover.
        """
        mean_rating = sum(self.ratings.values()) / max(len(self.ratings), 1)
        revert = config.ELO_SEASON_REVERT

        for team_id in self.ratings:
            self.ratings[team_id] = (
                self.ratings[team_id] * (1 - revert) + mean_rating * revert
            )

    def process_season_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all games in a season dataframe, updating Elo after each game.
        Games must be sorted by date. Returns the dataframe with elo columns added.
        """
        games = games_df.sort_values("date").copy()

        home_elos = []
        away_elos = []

        for _, game in games.iterrows():
            home_id = int(game["home_id"])
            away_id = int(game["away_id"])

            # Record pre-game Elo
            home_elos.append(self.get_rating(home_id))
            away_elos.append(self.get_rating(away_id))

            # Update after game
            neutral = bool(game.get("neutral_site", 0))
            self.update(home_id, away_id,
                        int(game["home_score"]), int(game["away_score"]),
                        neutral_site=neutral)

        games["home_elo"] = home_elos
        games["away_elo"] = away_elos
        games["elo_diff"] = games["home_elo"] - games["away_elo"]

        return games

    def get_all_ratings(self) -> dict[int, float]:
        """Return copy of all current ratings."""
        return dict(self.ratings)

    def get_top_teams(self, n: int = 25) -> list[tuple[int, float]]:
        """Return top N teams by Elo rating."""
        sorted_teams = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_teams[:n]


def build_elo_ratings(all_games: pd.DataFrame) -> tuple[EloSystem, pd.DataFrame]:
    """
    Build Elo ratings from scratch across all seasons.
    Returns the EloSystem and games dataframe with elo columns.
    """
    print("[Elo] Building ratings from scratch...")

    elo = EloSystem()
    seasons = sorted(all_games["season"].unique())
    all_with_elo = []

    for season in seasons:
        season_games = all_games[all_games["season"] == season].copy()
        season_with_elo = elo.process_season_games(season_games)
        all_with_elo.append(season_with_elo)

        top = elo.get_top_teams(5)
        print(f"  Season {season}: {len(season_games)} games processed, "
              f"top Elo: {top[0][1]:.0f}" if top else f"  Season {season}: no games")

        # Revert ratings between seasons
        elo.new_season()

    combined = pd.concat(all_with_elo, ignore_index=True)
    print(f"[Elo] Done — {len(combined)} total games with Elo ratings")

    return elo, combined
