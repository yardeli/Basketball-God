"""
Basketball-God — Phase 1: Data Acquisition & Schema Design
==========================================================
Ingests all Kaggle March Machine Learning Mania data into a unified SQLite
database with clean schema, era metadata, team normalization, and data
completeness tiers.

WHY THIS MATTERS:
  The original model trained only on ~1,100 NCAA tournament games (2007–2024).
  This pipeline unlocks ~200,000 games going back to 1985, covering regular
  season, conference tournaments, NCAA tournament, NIT, CBI, and CIT.
  More data + era awareness = a model that actually understands basketball
  patterns rather than just memorizing recent tournament results.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
KAGGLE_DIR = ROOT / "phase1_data" / "sources" / "kaggle"
DB_PATH    = ROOT / "phase1_data" / "output" / "basketball_god.db"
REPORT_DIR = ROOT / "phase1_data" / "output"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Era definitions ──────────────────────────────────────────────────────────
# These are RULE CHANGES that structurally altered the game. Features computed
# across eras must account for these shifts or comparisons are meaningless.
ERA_DEFINITIONS = {
    # Kaggle "Season" = year of March/April (e.g. Season 1985 = 1984-85)
    "pre_3pt":          (1985, 1986),   # No 3-point line before 1986-87
    "3pt_early":        (1987, 1993),   # 3pt line exists, 45-sec shot clock
    "modern_35sec":     (1994, 2015),   # 35-second shot clock era
    "modern_30sec":     (2016, 2019),   # 30-second shot clock era
    "covid":            (2021, 2021),   # Bubble / reduced schedule
    "portal_era":       (2022, 9999),   # Transfer portal fully open
}

def get_era(season: int) -> str:
    for name, (start, end) in ERA_DEFINITIONS.items():
        if start <= season <= end:
            return name
    return "modern_30sec"

# Shot clock length by era (used as a feature later)
def get_shot_clock(season: int) -> int:
    if season < 1987:  return 45   # No shot clock before 1985-86; used loosely
    if season < 1994:  return 45
    if season < 2016:  return 35
    return 30

# 3-point line flag
def has_3pt_line(season: int) -> bool:
    return season >= 1987

def daynum_to_date(season: int, day_num: int, seasons_df: pd.DataFrame) -> str:
    """Convert Kaggle DayNum to actual calendar date using MSeasons.csv."""
    row = seasons_df[seasons_df["Season"] == season]
    if row.empty:
        return None
    day_zero = pd.to_datetime(row.iloc[0]["DayZero"])
    actual_date = day_zero + timedelta(days=day_num)
    return actual_date.strftime("%Y-%m-%d")


# ── Database setup ────────────────────────────────────────────────────────────

def create_schema(conn: sqlite3.Connection):
    """
    Creates all tables. Schema is designed for time-series integrity:
    every feature that will be joined at game-time is stored here so we
    can guarantee no future data leaks into a game's pre-game context.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")

    statements = [
        """CREATE TABLE IF NOT EXISTS teams (
            team_id         INTEGER PRIMARY KEY,
            team_name       TEXT NOT NULL,
            first_d1_season INTEGER,
            last_d1_season  INTEGER
        )""",
        """CREATE TABLE IF NOT EXISTS team_spellings (
            spelling        TEXT PRIMARY KEY,
            team_id         INTEGER
        )""",
        """CREATE TABLE IF NOT EXISTS team_conferences (
            season          INTEGER,
            team_id         INTEGER,
            conference      TEXT,
            PRIMARY KEY (season, team_id)
        )""",
        """CREATE TABLE IF NOT EXISTS seasons (
            season          INTEGER PRIMARY KEY,
            day_zero        TEXT,
            region_w        TEXT,
            region_x        TEXT,
            region_y        TEXT,
            region_z        TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS tourney_seeds (
            season          INTEGER,
            team_id         INTEGER,
            seed_str        TEXT,
            region          TEXT,
            seed_num        INTEGER,
            play_in         INTEGER,
            PRIMARY KEY (season, team_id)
        )""",
        """CREATE TABLE IF NOT EXISTS team_coaches (
            season          INTEGER,
            team_id         INTEGER,
            first_day_num   INTEGER,
            last_day_num    INTEGER,
            coach_name      TEXT,
            PRIMARY KEY (season, team_id, first_day_num)
        )""",
        """CREATE TABLE IF NOT EXISTS games (
            game_id         TEXT PRIMARY KEY,
            season          INTEGER NOT NULL,
            day_num         INTEGER,
            game_date       TEXT,
            game_type       TEXT NOT NULL,
            neutral_site    INTEGER,
            w_team_id       INTEGER,
            l_team_id       INTEGER,
            w_score         INTEGER,
            l_score         INTEGER,
            score_diff      INTEGER,
            num_ot          INTEGER DEFAULT 0,
            w_loc           TEXT,
            era             TEXT,
            shot_clock      INTEGER,
            has_3pt         INTEGER,
            data_tier       INTEGER,
            source          TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS game_stats (
            game_id         TEXT,
            team_id         INTEGER,
            opponent_id     INTEGER,
            is_winner       INTEGER,
            score           INTEGER,
            opp_score       INTEGER,
            fgm             INTEGER,
            fga             INTEGER,
            fg_pct          REAL,
            fgm3            INTEGER,
            fga3            INTEGER,
            fg3_pct         REAL,
            ftm             INTEGER,
            fta             INTEGER,
            ft_pct          REAL,
            off_reb         INTEGER,
            def_reb         INTEGER,
            tot_reb         INTEGER,
            ast             INTEGER,
            turnovers       INTEGER,
            stl             INTEGER,
            blk             INTEGER,
            pf              INTEGER,
            poss_est        REAL,
            off_eff         REAL,
            def_eff         REAL,
            efg_pct         REAL,
            to_rate         REAL,
            orb_rate        REAL,
            ft_rate         REAL,
            PRIMARY KEY (game_id, team_id)
        )""",
        """CREATE TABLE IF NOT EXISTS massey_ordinals (
            season          INTEGER,
            ranking_day     INTEGER,
            system_name     TEXT,
            team_id         INTEGER,
            ordinal_rank    INTEGER,
            PRIMARY KEY (season, ranking_day, system_name, team_id)
        )""",
        """CREATE TABLE IF NOT EXISTS team_rankings_snapshot (
            season          INTEGER,
            day_num         INTEGER,
            team_id         INTEGER,
            n_systems       INTEGER,
            avg_rank        REAL,
            best_rank       INTEGER,
            worst_rank      INTEGER,
            rank_spread     REAL,
            PRIMARY KEY (season, day_num, team_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_games_season      ON games(season)",
        "CREATE INDEX IF NOT EXISTS idx_games_w_team      ON games(w_team_id)",
        "CREATE INDEX IF NOT EXISTS idx_games_l_team      ON games(l_team_id)",
        "CREATE INDEX IF NOT EXISTS idx_games_date        ON games(game_date)",
        "CREATE INDEX IF NOT EXISTS idx_game_stats_team   ON game_stats(team_id)",
        "CREATE INDEX IF NOT EXISTS idx_massey_season_day ON massey_ordinals(season, ranking_day)",
        "CREATE INDEX IF NOT EXISTS idx_massey_team       ON massey_ordinals(team_id)",
    ]

    for stmt in statements:
        cur.execute(stmt)
    conn.commit()
    print("  Schema created.")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_teams(conn: sqlite3.Connection):
    print("Loading teams...")
    df = pd.read_csv(KAGGLE_DIR / "MTeams.csv")
    df.columns = ["team_id", "team_name", "first_d1_season", "last_d1_season"]
    df.to_sql("teams", conn, if_exists="replace", index=False)

    # Team spellings for external source normalization
    sp = pd.read_csv(KAGGLE_DIR / "MTeamSpellings.csv")
    sp.columns = ["spelling", "team_id"]
    sp.to_sql("team_spellings", conn, if_exists="replace", index=False)

    print(f"  {len(df)} teams, {len(sp)} name spellings loaded.")


def load_seasons(conn: sqlite3.Connection) -> pd.DataFrame:
    print("Loading seasons...")
    df = pd.read_csv(KAGGLE_DIR / "MSeasons.csv")
    df.to_sql("seasons", conn, if_exists="replace", index=False)
    print(f"  {len(df)} seasons ({df['Season'].min()}–{df['Season'].max()}).")
    return df


def load_conferences(conn: sqlite3.Connection):
    print("Loading conference memberships...")
    df = pd.read_csv(KAGGLE_DIR / "MTeamConferences.csv")
    df.columns = ["season", "team_id", "conference"]
    df.to_sql("team_conferences", conn, if_exists="replace", index=False)
    print(f"  {len(df)} team-season conference records.")


def load_seeds(conn: sqlite3.Connection):
    print("Loading tournament seeds...")
    df = pd.read_csv(KAGGLE_DIR / "MNCAATourneySeeds.csv")
    df.columns = ["season", "seed_str", "team_id"]

    # Parse seed string: "W01", "X16b" etc.
    df["region"]   = df["seed_str"].str[0]
    df["seed_num"] = df["seed_str"].str[1:3].astype(int)
    df["play_in"]  = df["seed_str"].str[3:].str.len().gt(0).astype(int)

    df.to_sql("tourney_seeds", conn, if_exists="replace", index=False)
    print(f"  {len(df)} seed records ({df['season'].min()}–{df['season'].max()}).")


def load_coaches(conn: sqlite3.Connection):
    print("Loading coaches...")
    df = pd.read_csv(KAGGLE_DIR / "MTeamCoaches.csv")
    df.to_sql("team_coaches", conn, if_exists="replace", index=False)
    print(f"  {len(df)} coach-season records.")


def _compact_to_games(df: pd.DataFrame, game_type: str,
                      seasons_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a compact results DataFrame (W/L format) to unified games rows.
    Compact = score only, Tier 3 for pre-2003, Tier 2 otherwise.
    """
    rows = []
    for _, r in df.iterrows():
        s  = int(r["Season"])
        dn = int(r["DayNum"])
        wid = int(r["WTeamID"])
        lid = int(r["LTeamID"])
        gid = f"{s}_{dn}_{wid}_{lid}"

        wloc = str(r.get("WLoc", "N"))
        neutral = 1 if wloc == "N" else 0

        rows.append({
            "game_id":      gid,
            "season":       s,
            "day_num":      dn,
            "game_date":    daynum_to_date(s, dn, seasons_df),
            "game_type":    game_type,
            "neutral_site": neutral,
            "w_team_id":    wid,
            "l_team_id":    lid,
            "w_score":      int(r["WScore"]),
            "l_score":      int(r["LScore"]),
            "score_diff":   int(r["WScore"]) - int(r["LScore"]),
            "num_ot":       int(r.get("NumOT", 0)),
            "w_loc":        wloc,
            "era":          get_era(s),
            "shot_clock":   get_shot_clock(s),
            "has_3pt":      int(has_3pt_line(s)),
            "data_tier":    1 if s >= 2003 else (2 if s >= 1993 else 3),
            "source":       "kaggle_compact",
        })
    return pd.DataFrame(rows)


def _detailed_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode detailed results into per-team-per-game stats rows.
    Computes derived efficiency metrics at ingest time.

    WHY COMPUTE HERE: pre-computing eFG%, pace, efficiency at ingest avoids
    recomputing in every training run and ensures consistency.
    """
    rows = []
    for _, r in df.iterrows():
        s   = int(r["Season"])
        dn  = int(r["DayNum"])
        wid = int(r["WTeamID"])
        lid = int(r["LTeamID"])
        gid = f"{s}_{dn}_{wid}_{lid}"

        for prefix, team_id, opp_id, is_win, pfx_opp in [
            ("W", wid, lid, 1, "L"),
            ("L", lid, wid, 0, "W"),
        ]:
            fgm  = r[f"{prefix}FGM"]
            fga  = r[f"{prefix}FGA"]
            fgm3 = r[f"{prefix}FGM3"]
            fga3 = r[f"{prefix}FGA3"]
            ftm  = r[f"{prefix}FTM"]
            fta  = r[f"{prefix}FTA"]
            orb  = r[f"{prefix}OR"]
            drb  = r[f"{prefix}DR"]
            ast  = r[f"{prefix}Ast"]
            to   = r[f"{prefix}TO"]
            stl  = r[f"{prefix}Stl"]
            blk  = r[f"{prefix}Blk"]
            pf   = r[f"{prefix}PF"]
            pts  = r[f"{prefix}Score"]
            opp_pts = r[f"{pfx_opp}Score"]
            opp_orb = r[f"{pfx_opp}OR"]
            opp_fga = r[f"{pfx_opp}FGA"]
            opp_fta = r[f"{pfx_opp}FTA"]

            # Estimated possessions (Dean Oliver formula)
            poss = (fga + 0.44 * fta - orb + to +
                    opp_fga + 0.44 * opp_fta - opp_orb + r[f"{pfx_opp}TO"]) / 2
            poss = max(poss, 1)  # guard against zero

            # Efficiency (points per 100 possessions)
            off_eff = (pts / poss) * 100
            def_eff = (opp_pts / poss) * 100

            # Effective FG% = (FGM + 0.5 * FGM3) / FGA
            efg = (fgm + 0.5 * fgm3) / fga if fga > 0 else np.nan

            # Turnover rate = TO / possessions
            to_rate = to / poss

            # Offensive rebound rate = OR / (OR + opponent DR)
            opp_dr = r[f"{pfx_opp}DR"]
            orb_rate = orb / (orb + opp_dr) if (orb + opp_dr) > 0 else np.nan

            # FT rate = FTA / FGA (how often you get to the line)
            ft_rate = fta / fga if fga > 0 else np.nan

            rows.append({
                "game_id":    gid,
                "team_id":    team_id,
                "opponent_id": opp_id,
                "is_winner":  is_win,
                "score":      int(pts),
                "opp_score":  int(opp_pts),
                "fgm":        int(fgm),
                "fga":        int(fga),
                "fg_pct":     round(fgm / fga, 4) if fga > 0 else None,
                "fgm3":       int(fgm3),
                "fga3":       int(fga3),
                "fg3_pct":    round(fgm3 / fga3, 4) if fga3 > 0 else None,
                "ftm":        int(ftm),
                "fta":        int(fta),
                "ft_pct":     round(ftm / fta, 4) if fta > 0 else None,
                "off_reb":    int(orb),
                "def_reb":    int(drb),
                "tot_reb":    int(orb + drb),
                "ast":        int(ast),
                "turnovers":  int(to),
                "stl":        int(stl),
                "blk":        int(blk),
                "pf":         int(pf),
                "poss_est":   round(poss, 2),
                "off_eff":    round(off_eff, 2),
                "def_eff":    round(def_eff, 2),
                "efg_pct":    round(efg, 4) if not np.isnan(efg) else None,
                "to_rate":    round(to_rate, 4),
                "orb_rate":   round(orb_rate, 4) if orb_rate is not None and not np.isnan(orb_rate) else None,
                "ft_rate":    round(ft_rate, 4) if ft_rate is not None and not np.isnan(ft_rate) else None,
            })
    return pd.DataFrame(rows)


def load_games(conn: sqlite3.Connection, seasons_df: pd.DataFrame):
    print("Loading games...")
    all_games  = []
    all_stats  = []

    # ── Regular season compact (1985+, all games, score only for pre-2003) ──
    print("  Regular season compact...")
    df = pd.read_csv(KAGGLE_DIR / "MRegularSeasonCompactResults.csv")
    all_games.append(_compact_to_games(df, "regular", seasons_df))

    # ── Regular season detailed (2003+, full box scores) ──
    print("  Regular season detailed (box scores)...")
    df_det = pd.read_csv(KAGGLE_DIR / "MRegularSeasonDetailedResults.csv")
    all_stats.append(_detailed_stats(df_det))
    # Upgrade data_tier to 1 for seasons with detailed data
    detailed_gids = set(
        f"{int(r.Season)}_{int(r.DayNum)}_{int(r.WTeamID)}_{int(r.LTeamID)}"
        for _, r in df_det.iterrows()
    )

    # ── NCAA tournament compact ──
    print("  NCAA tournament (compact)...")
    df = pd.read_csv(KAGGLE_DIR / "MNCAATourneyCompactResults.csv")
    all_games.append(_compact_to_games(df, "ncaa_tourney", seasons_df))

    # ── NCAA tournament detailed ──
    print("  NCAA tournament (box scores)...")
    df_det = pd.read_csv(KAGGLE_DIR / "MNCAATourneyDetailedResults.csv")
    all_stats.append(_detailed_stats(df_det))
    detailed_gids |= set(
        f"{int(r.Season)}_{int(r.DayNum)}_{int(r.WTeamID)}_{int(r.LTeamID)}"
        for _, r in df_det.iterrows()
    )

    # ── Conference tournament games (uses compact results, filtered by game IDs) ──
    print("  Conference tournament games...")
    conf_df = pd.read_csv(KAGGLE_DIR / "MConferenceTourneyGames.csv")
    # These are game ID references — mark matching games as conf_tourney type
    conf_keys = set(
        f"{int(r.Season)}_{int(r.DayNum)}_{int(r.WTeamID)}_{int(r.LTeamID)}"
        for _, r in conf_df.iterrows()
        if "WTeamID" in conf_df.columns
    )
    # Fallback: file may have different format
    if "WTeamID" not in conf_df.columns:
        print(f"    conf_tourney columns: {list(conf_df.columns)}")

    # ── Secondary tournaments (NIT, CBI, CIT, etc.) ──
    print("  Secondary tournaments (NIT/CBI/CIT)...")
    df = pd.read_csv(KAGGLE_DIR / "MSecondaryTourneyCompactResults.csv")
    # Map tourney type
    def map_secondary(t):
        t = str(t).strip().upper()
        if "NIT" in t: return "nit"
        if "CBI" in t: return "cbi"
        if "CIT" in t: return "cit"
        return "other_tourney"
    df["game_type_mapped"] = df["SecondaryTourney"].apply(map_secondary)
    for gtype, grp in df.groupby("game_type_mapped"):
        all_games.append(_compact_to_games(grp, gtype, seasons_df))

    # ── Combine all games ──
    print("  Merging and deduplicating...")
    games_df = pd.concat(all_games, ignore_index=True)

    # Deduplicate: same game_id may appear in compact + (implicitly via detailed)
    # Keep one row per game_id, preferring the more specific type
    type_priority = {
        "ncaa_tourney": 0, "nit": 1, "cbi": 2, "cit": 3,
        "conf_tourney": 4, "other_tourney": 5, "regular": 6
    }
    games_df["_pri"] = games_df["game_type"].map(type_priority).fillna(9)
    games_df = (games_df.sort_values("_pri")
                        .drop_duplicates(subset="game_id", keep="first")
                        .drop(columns=["_pri"])
                        .reset_index(drop=True))

    # Mark conference tourney games by cross-referencing conf_tourney key set
    if conf_keys:
        mask = games_df["game_id"].isin(conf_keys)
        games_df.loc[mask & (games_df["game_type"] == "regular"), "game_type"] = "conf_tourney"

    # Upgrade data_tier where detailed stats exist
    games_df.loc[games_df["game_id"].isin(detailed_gids), "data_tier"] = 1

    games_df.to_sql("games", conn, if_exists="replace", index=False)
    print(f"  {len(games_df):,} total games written to DB.")

    # ── Write game stats ──
    print("  Writing box score stats...")
    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df = stats_df.drop_duplicates(subset=["game_id", "team_id"], keep="first")
    stats_df.to_sql("game_stats", conn, if_exists="replace", index=False)
    print(f"  {len(stats_df):,} team-game stat rows written.")

    return games_df, stats_df


def load_massey_ordinals(conn: sqlite3.Connection):
    """
    Load Massey Ordinals — multiple computer ranking systems per week.
    WHY: This gives us a KenPom-substitute for every team going back to 2003.
    We'll aggregate these into a consensus ranking snapshot per game day.
    """
    print("Loading Massey Ordinals (computer rankings)...")
    # 5.8M rows — use chunked loading
    chunks = []
    for chunk in pd.read_csv(KAGGLE_DIR / "MMasseyOrdinals.csv", chunksize=500_000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df.columns = ["season", "ranking_day", "system_name", "team_id", "ordinal_rank"]

    df.to_sql("massey_ordinals", conn, if_exists="replace", index=False,
              chunksize=5_000)
    print(f"  {len(df):,} ranking records ({df['season'].min()}–{df['season'].max()}).")
    print(f"  {df['system_name'].nunique()} distinct rating systems.")

    # ── Build consensus snapshots ──
    # For each (season, ranking_day, team_id) aggregate across all systems.
    # WHY: Individual systems have biases. Consensus is more stable and
    # correlates better with actual strength.
    print("  Building consensus ranking snapshots (this takes ~30s)...")
    snap = (df.groupby(["season", "ranking_day", "team_id"])["ordinal_rank"]
              .agg(n_systems="count",
                   avg_rank="mean",
                   best_rank="min",
                   worst_rank="max",
                   rank_spread="std")
              .reset_index())
    snap.columns = ["season", "day_num", "team_id",
                    "n_systems", "avg_rank", "best_rank", "worst_rank", "rank_spread"]
    snap["avg_rank"]   = snap["avg_rank"].round(1)
    snap["rank_spread"] = snap["rank_spread"].round(2)
    snap.to_sql("team_rankings_snapshot", conn, if_exists="replace", index=False,
                chunksize=5_000)
    print(f"  {len(snap):,} consensus snapshot records.")


# ── Summary report ─────────────────────────────────────────────────────────────

def generate_report(conn: sqlite3.Connection, games_df: pd.DataFrame):
    """
    Generate a human-readable summary of what we ingested.
    Flags gaps, anomalies, and completeness distribution.
    """
    print("\nGenerating summary report...")
    report = {}

    # Total games by type
    report["games_by_type"] = (
        games_df.groupby("game_type").size()
        .sort_values(ascending=False)
        .to_dict()
    )

    # Games by decade
    games_df["decade"] = (games_df["season"] // 10 * 10).astype(str) + "s"
    report["games_by_decade"] = (
        games_df.groupby("decade").size().to_dict()
    )

    # Data completeness distribution
    report["data_tier_counts"] = (
        games_df.groupby("data_tier").size().to_dict()
    )
    report["data_tier_labels"] = {
        1: "Tier 1 — Full box score (FG/3PT/FT/reb/ast/to/stl/blk)",
        2: "Tier 2 — Score only (2003+ seasons, no box score available)",
        3: "Tier 3 — Score only (pre-2003)",
    }

    # Games by era
    report["games_by_era"] = (
        games_df.groupby("era").size()
        .sort_values(ascending=False)
        .to_dict()
    )

    # Season range
    report["season_range"] = {
        "earliest": int(games_df["season"].min()),
        "latest":   int(games_df["season"].max()),
        "total_seasons": int(games_df["season"].nunique()),
    }

    # Team count
    teams_df = pd.read_sql("SELECT COUNT(*) as n FROM teams", conn)
    report["total_teams"] = int(teams_df["n"].iloc[0])

    # Massey systems
    sys_count = pd.read_sql(
        "SELECT COUNT(DISTINCT system_name) as n FROM massey_ordinals", conn
    )
    report["massey_rating_systems"] = int(sys_count["n"].iloc[0])

    # Missing games check: identify seasons with unusually low game counts
    season_counts = games_df.groupby("season").size()
    median_games  = season_counts.median()
    low_seasons   = season_counts[season_counts < median_games * 0.5].index.tolist()
    report["anomalies"] = {
        "low_game_count_seasons": low_seasons,
        "note": "COVID 2021 season expected to be short (~3,000 games vs ~5,500 normal)."
    }

    # Save report
    report_path = REPORT_DIR / "phase1_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Pretty print
    print("\n" + "="*60)
    print("  PHASE 1 INGESTION SUMMARY")
    print("="*60)
    print(f"\n  Seasons: {report['season_range']['earliest']}–{report['season_range']['latest']}"
          f" ({report['season_range']['total_seasons']} seasons)")
    print(f"  Teams:   {report['total_teams']}")

    print("\n  Games by type:")
    for gtype, cnt in sorted(report["games_by_type"].items(), key=lambda x: -x[1]):
        print(f"    {gtype:<20} {cnt:>7,}")
    total_games = sum(report["games_by_type"].values())
    print(f"    {'TOTAL':<20} {total_games:>7,}")

    print("\n  Games by decade:")
    for dec, cnt in sorted(report["games_by_decade"].items()):
        print(f"    {dec:<12} {cnt:>7,}")

    print("\n  Data completeness:")
    for tier, cnt in sorted(report["data_tier_counts"].items()):
        label = report["data_tier_labels"].get(tier, "Unknown tier")
        pct   = cnt / total_games * 100
        print(f"    Tier {tier}: {cnt:>7,} games ({pct:.1f}%)")
        print(f"           {label}")

    print("\n  Massey rating systems loaded:", report["massey_rating_systems"])

    if low_seasons:
        print(f"\n  [!] Low game count seasons: {low_seasons}")
    else:
        print("\n  [OK] No anomalous season gaps detected.")

    print(f"\n  Full report: {report_path}")
    print("="*60)

    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nBasketball-God — Phase 1: Data Ingestion")
    print(f"DB path: {DB_PATH}")
    print(f"Kaggle data: {KAGGLE_DIR}\n")

    if not KAGGLE_DIR.exists():
        print(f"ERROR: Kaggle data directory not found: {KAGGLE_DIR}")
        sys.exit(1)

    # Check required files
    required = [
        "MTeams.csv", "MSeasons.csv", "MTeamSpellings.csv",
        "MRegularSeasonCompactResults.csv", "MRegularSeasonDetailedResults.csv",
        "MNCAATourneyCompactResults.csv", "MNCAATourneyDetailedResults.csv",
        "MNCAATourneySeeds.csv", "MTeamConferences.csv", "MConferenceTourneyGames.csv",
        "MSecondaryTourneyCompactResults.csv", "MMasseyOrdinals.csv", "MTeamCoaches.csv",
    ]
    missing = [f for f in required if not (KAGGLE_DIR / f).exists()]
    if missing:
        print(f"ERROR: Missing files: {missing}")
        sys.exit(1)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    try:
        create_schema(conn)
        load_teams(conn)
        seasons_df = load_seasons(conn)
        load_conferences(conn)
        load_seeds(conn)
        load_coaches(conn)
        games_df, stats_df = load_games(conn, seasons_df)
        load_massey_ordinals(conn)
        report = generate_report(conn, games_df)
        conn.commit()
        print("\nPhase 1 complete. Database ready.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
