"""
Phase 1: Unified Data Schema — Single canonical schema for all NCAA D1 game data.

Every game from every source gets normalized into this schema.
Sources may provide different levels of detail, tracked by `data_completeness_tier`:
  - Tier 1: Full box score (FG/FGA, 3P, FT, rebounds, assists, turnovers, etc.)
  - Tier 2: Basic stats (score, some shooting stats, maybe rebounds)
  - Tier 3: Score only (date, teams, final score — that's it)

Era metadata captures rule changes that affect stat interpretation:
  - pre_3pt: Before 1986-87 (no 3-point line)
  - early_3pt: 1986-87 to 2007-08 (3pt line at 19'9")
  - extended_3pt: 2008-09 to 2018-19 (3pt moved to 20'9")
  - modern_3pt: 2019-20+ (3pt at current distance)
  - pre_shot_clock_change: Before 2015-16 (35-second shot clock)
  - post_shot_clock_change: 2015-16+ (30-second shot clock)
  - covid_year: 2020-21 (shortened season, empty arenas, opt-outs)
"""
import sqlite3
from pathlib import Path

SCHEMA_VERSION = "1.0"

# ── Game types ──
GAME_TYPES = {
    "regular": "Regular Season",
    "conf_tourney": "Conference Tournament",
    "ncaa_tourney": "NCAA Tournament (March Madness)",
    "nit": "National Invitation Tournament",
    "cbi": "College Basketball Invitational",
    "cit": "CollegeInsider.com Tournament",
    "other_postseason": "Other Postseason",
    "exhibition": "Exhibition",
}

# ── Era definitions ──
def get_era(season: int) -> str:
    """
    Return the era tag for a given season (ending year).
    E.g., season=1987 means the 1986-87 season.
    """
    if season <= 1986:
        return "pre_3pt"
    elif season <= 2008:
        return "early_3pt"
    elif season <= 2019:
        return "extended_3pt"
    else:
        return "modern_3pt"


def get_era_flags(season: int) -> dict:
    """Return all era-related boolean flags for a season."""
    return {
        "era": get_era(season),
        "has_3pt_line": season >= 1987,
        "shot_clock_30": season >= 2016,  # Changed from 35s to 30s
        "covid_year": season == 2021,
        "transfer_portal_era": season >= 2019,  # Portal became major factor
        "nil_era": season >= 2022,  # NIL deals began
        "3pt_distance_ft": 19.75 if season < 2008 else (20.75 if season < 2020 else 22.15),
    }


# ── SQLite Schema ──
CREATE_TABLES_SQL = """
-- Teams master table with canonical names and aliases
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL UNIQUE,
    espn_id INTEGER,
    kaggle_id INTEGER,
    sports_ref_id TEXT,
    abbreviation TEXT,
    conference TEXT,
    conference_2024 TEXT,  -- Current conference (post-realignment)
    is_d1 INTEGER DEFAULT 1,
    first_season INTEGER,
    last_season INTEGER
);

-- Team name aliases for deduplication
CREATE TABLE IF NOT EXISTS team_aliases (
    alias TEXT PRIMARY KEY,
    team_id INTEGER NOT NULL,
    source TEXT,  -- which data source uses this name
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- Conference history (teams change conferences)
CREATE TABLE IF NOT EXISTS conference_history (
    team_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    conference TEXT NOT NULL,
    PRIMARY KEY (team_id, season),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- Unified games table
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,  -- source_type:source_id (e.g., "espn:401234567")
    season INTEGER NOT NULL,
    date TEXT NOT NULL,  -- YYYY-MM-DD
    game_type TEXT NOT NULL DEFAULT 'regular',  -- regular/conf_tourney/ncaa_tourney/nit/other

    -- Teams
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_team_name TEXT NOT NULL,
    away_team_name TEXT NOT NULL,
    neutral_site INTEGER DEFAULT 0,

    -- Score
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    home_win INTEGER NOT NULL,
    num_overtimes INTEGER DEFAULT 0,

    -- Tournament context
    ncaa_seed_home INTEGER,
    ncaa_seed_away INTEGER,
    tournament_round TEXT,  -- R64/R32/S16/E8/F4/NCG

    -- Conference
    home_conference TEXT,
    away_conference TEXT,
    conference_game INTEGER DEFAULT 0,

    -- Rankings (AP poll at game time, if available)
    home_ap_rank INTEGER,
    away_ap_rank INTEGER,

    -- Basic box score (Tier 2+)
    home_fgm INTEGER, home_fga INTEGER,
    away_fgm INTEGER, away_fga INTEGER,
    home_fg3m INTEGER, home_fg3a INTEGER,
    away_fg3m INTEGER, away_fg3a INTEGER,
    home_ftm INTEGER, home_fta INTEGER,
    away_ftm INTEGER, away_fta INTEGER,
    home_or INTEGER, home_dr INTEGER,  -- offensive/defensive rebounds
    away_or INTEGER, away_dr INTEGER,
    home_ast INTEGER, home_to INTEGER, home_stl INTEGER, home_blk INTEGER, home_pf INTEGER,
    away_ast INTEGER, away_to INTEGER, away_stl INTEGER, away_blk INTEGER, away_pf INTEGER,

    -- Computed
    margin INTEGER,  -- home_score - away_score
    total_points INTEGER,

    -- Metadata
    era TEXT NOT NULL,
    data_completeness_tier INTEGER NOT NULL DEFAULT 3,  -- 1=full box, 2=basic, 3=score only
    data_source TEXT NOT NULL,  -- espn/kaggle/sports_ref/manual
    source_game_id TEXT,  -- Original ID from source
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(date);
CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_games_type ON games(game_type);
CREATE INDEX IF NOT EXISTS idx_games_source ON games(data_source);

-- Team season stats (pre-computed per-season aggregates)
CREATE TABLE IF NOT EXISTS team_season_stats (
    team_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    games_played INTEGER,
    wins INTEGER,
    losses INTEGER,
    win_pct REAL,
    points_per_game REAL,
    opp_points_per_game REAL,
    scoring_margin REAL,
    -- Advanced (when available)
    adj_off_eff REAL,  -- KenPom-style adjusted offensive efficiency
    adj_def_eff REAL,
    adj_tempo REAL,
    sos_rank INTEGER,  -- Strength of schedule
    rpi REAL,  -- RPI (pre-2019) or NET (2019+)
    -- Tournament
    ncaa_seed INTEGER,
    PRIMARY KEY (team_id, season),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- Data source tracking
CREATE TABLE IF NOT EXISTS data_sources (
    source_name TEXT PRIMARY KEY,
    games_loaded INTEGER DEFAULT 0,
    seasons_covered TEXT,
    last_updated TEXT,
    notes TEXT
);

-- Experiment log
CREATE TABLE IF NOT EXISTS experiment_log (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    phase TEXT,
    description TEXT,
    parameters TEXT,  -- JSON
    metrics TEXT,  -- JSON
    notes TEXT
);
"""


def init_database(db_path: str = None) -> sqlite3.Connection:
    """Initialize the SQLite database with the unified schema."""
    if db_path is None:
        db_path = str(Path(__file__).parent / "output" / "basketball_god.db")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(CREATE_TABLES_SQL)
    conn.commit()

    print(f"[Schema] Database initialized: {db_path}")
    return conn


def compute_completeness_tier(game: dict) -> int:
    """
    Determine data completeness tier for a game record.
    Tier 1: Full box score (FG/FGA, 3P, FT, rebounds, assists, turnovers, etc.)
    Tier 2: Basic stats (some shooting stats or rebounds available)
    Tier 3: Score only
    """
    # Check for full box score fields
    box_fields = ["home_fgm", "home_fga", "home_ast", "home_to", "home_or", "home_dr"]
    has_box = sum(1 for f in box_fields if game.get(f) is not None) >= 5

    if has_box:
        return 1

    # Check for basic stats
    basic_fields = ["home_fgm", "home_fga", "home_ftm", "home_fta"]
    has_basic = sum(1 for f in basic_fields if game.get(f) is not None) >= 2

    if has_basic:
        return 2

    return 3


if __name__ == "__main__":
    conn = init_database()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Tables created: {tables}")
    conn.close()
