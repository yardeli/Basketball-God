"""
NCAA-generator v2.0 — Configuration
General-purpose NCAA D1 Men's Basketball prediction system.
Predicts ANY game, not just March Madness.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
VIZ_DIR = BASE_DIR / "visualizations"

# ─── ESPN API ────────────────────────────────────────────────────────────────
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_TEAMS = f"{ESPN_BASE}/teams"
ESPN_SCOREBOARD = f"{ESPN_BASE}/scoreboard"
ESPN_STANDINGS = f"{ESPN_BASE}/standings"
ESPN_RANKINGS = f"{ESPN_BASE}/rankings"
ESPN_SUMMARY = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
ESPN_GROUPS_D1 = 50  # Group ID for Division I

# ─── News/Injury RSS Feeds ──────────────────────────────────────────────────
RSS_FEEDS = {
    "espn_cbb": "https://www.espn.com/espn/rss/ncb/news",
    "cbs_cbb": "https://www.cbssports.com/rss/headlines/college-basketball/",
    "rotowire_cbb": "https://www.rotowire.com/rss/cbasketball-news.xml",
}
ROTOWIRE_INJURY_URL = "https://www.rotowire.com/cbasketball/injury-report.php"

# ─── Seasons ─────────────────────────────────────────────────────────────────
# NCAA season labeled by ending year (e.g., 2025 = 2024-25 season)
HISTORICAL_SEASONS = list(range(2010, 2027))  # 2009-10 through 2025-26
CURRENT_SEASON = 2026  # 2025-26 season
MIN_SEASON = 2010  # Earliest season to scrape

# ─── Power Conferences (2025-26 alignment) ───────────────────────────────────
POWER_CONFERENCES = {
    "SEC", "Big Ten", "Big 12", "ACC", "Big East",
}
MID_MAJOR_STRONG = {
    "WCC", "American Athletic", "Mountain West", "Atlantic 10", "Missouri Valley",
}

# ─── Features ────────────────────────────────────────────────────────────────
# Team-level rolling stats (computed from game logs — scoreboard data)
TEAM_ROLLING_FEATURES = [
    "points_pg", "opp_points_pg", "margin_pg", "win_pct", "momentum",
]

# Matchup differential features (home - away)
# These are the features actually available from ESPN scoreboard data.
# Box-score-level stats (FG%, 3PT%, rebounds, etc.) would require
# per-game summary API calls — a future enhancement.
MATCHUP_FEATURES = [
    # Core strength
    "elo_diff", "net_rating_diff", "win_pct_diff",
    # Scoring
    "points_pg_diff", "opp_points_pg_diff",
    # Context
    "home_away", "rest_days_diff", "conf_game",
    "power_conf_diff", "momentum_diff",
    # Text-derived
    "injury_impact_diff", "sentiment_diff",
]

# ─── Elo System ──────────────────────────────────────────────────────────────
ELO_INITIAL = 1500
ELO_K = 20  # K-factor
ELO_HOME_ADVANTAGE = 65  # ~65 Elo points home court advantage
ELO_SEASON_REVERT = 0.33  # Revert 33% to mean between seasons

# ─── Model ───────────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 5,
    "n_estimators": 800,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

LOGISTIC_PARAMS = {
    "C": 0.5,
    "max_iter": 2000,
    "random_state": 42,
}

ENSEMBLE_WEIGHT_XGB = 0.65
ENSEMBLE_WEIGHT_LR = 0.35

# ─── Rolling Window ─────────────────────────────────────────────────────────
ROLLING_WINDOW = 10  # Games for rolling averages
MOMENTUM_WINDOW = 5  # Recent games for momentum signal

# ─── Request Settings ────────────────────────────────────────────────────────
REQUEST_DELAY = 1.0  # Seconds between API requests (be respectful)
REQUEST_TIMEOUT = 15
USER_AGENT = "NCAA-generator/2.0 (research project)"
