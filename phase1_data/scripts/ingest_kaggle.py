"""
Phase 1 — Kaggle March Machine Learning Mania Data Ingestion

The Kaggle NCAA tournament dataset is one of the best free sources for historical
NCAA basketball data. It covers 1985-present with:
  - Regular season game results (scores)
  - Tournament game results with seeds and slots
  - Team names and IDs
  - Conference assignments
  - Some seasons have detailed box scores (2003+)

To use this ingester:
  1. Download the dataset from: https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data
     (or any recent year's competition — the historical data is the same)
  2. Place the CSV files in: Basketball-God/phase1_data/sources/kaggle/
  3. Run: python ingest_kaggle.py

Expected files:
  - MRegularSeasonCompactResults.csv (1985+, all regular season games, score only)
  - MRegularSeasonDetailedResults.csv (2003+, games with box score stats)
  - MNCAATourneyCompactResults.csv (1985+, tournament games, score only)
  - MNCAATourneyDetailedResults.csv (2003+, tournament box scores)
  - MNCAATourneySeeds.csv (seeds for every team in every tournament)
  - MTeams.csv (team ID ↔ name mapping)
  - MTeamConferences.csv (conference assignments by year)
  - MConferences.csv (conference name lookup)
"""
import sys
import os
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from pathlib import Path

from phase1_data.schema import init_database, get_era, compute_completeness_tier
from phase1_data.team_normalization import TeamNormalizer

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

KAGGLE_DIR = Path(__file__).parent.parent / "sources" / "kaggle"


def _check_kaggle_files() -> dict[str, Path]:
    """Check which Kaggle CSV files are available."""
    expected = {
        "teams": "MTeams.csv",
        "reg_compact": "MRegularSeasonCompactResults.csv",
        "reg_detailed": "MRegularSeasonDetailedResults.csv",
        "tourney_compact": "MNCAATourneyCompactResults.csv",
        "tourney_detailed": "MNCAATourneyDetailedResults.csv",
        "seeds": "MNCAATourneySeeds.csv",
        "conferences": "MTeamConferences.csv",
        "conf_names": "MConferences.csv",
    }

    found = {}
    missing = []
    for key, filename in expected.items():
        path = KAGGLE_DIR / filename
        if path.exists():
            found[key] = path
        else:
            missing.append(filename)

    if missing:
        print(f"[Kaggle] Missing files: {missing}")
        print(f"  Download from: https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data")
        print(f"  Place in: {KAGGLE_DIR}")
    else:
        print(f"[Kaggle] All {len(found)} expected files found")

    return found


def _load_team_names(files: dict) -> dict[int, str]:
    """Load Kaggle team_id → team_name mapping."""
    if "teams" not in files:
        return {}
    df = pd.read_csv(files["teams"])
    return dict(zip(df["TeamID"], df["TeamName"]))


def _load_seeds(files: dict) -> dict[tuple[int, int], int]:
    """Load tournament seeds: (season, team_id) → seed number."""
    if "seeds" not in files:
        return {}
    df = pd.read_csv(files["seeds"])

    seeds = {}
    for _, row in df.iterrows():
        seed_str = row["Seed"]
        # Seed format: "W01" (region + seed number)
        seed_num = int(seed_str[1:3])
        seeds[(int(row["Season"]), int(row["TeamID"]))] = seed_num

    return seeds


def _load_conferences(files: dict) -> dict[tuple[int, int], str]:
    """Load conference assignments: (season, team_id) → conference."""
    if "conferences" not in files:
        return {}

    df = pd.read_csv(files["conferences"])
    conf_names = {}
    if "conf_names" in files:
        cn = pd.read_csv(files["conf_names"])
        conf_names = dict(zip(cn["ConfAbbrev"], cn["Description"]))

    result = {}
    for _, row in df.iterrows():
        abbrev = row["ConfAbbrev"]
        full_name = conf_names.get(abbrev, abbrev)
        result[(int(row["Season"]), int(row["TeamID"]))] = full_name

    return result


def ingest_kaggle_compact(conn: sqlite3.Connection, normalizer: TeamNormalizer,
                           files: dict, team_names: dict, seeds: dict,
                           conferences: dict, game_type: str = "regular"):
    """
    Ingest compact (score-only) game results.
    These go back to 1985 and are Tier 3 data.
    """
    if game_type == "regular":
        key = "reg_compact"
        label = "Regular Season"
    else:
        key = "tourney_compact"
        label = "NCAA Tournament"

    if key not in files:
        print(f"[Kaggle] No {label} compact results file found")
        return 0

    print(f"[Kaggle] Ingesting {label} compact results...")
    df = pd.read_csv(files[key])

    inserted = 0
    for _, row in df.iterrows():
        season = int(row["Season"])
        day_num = int(row["DayNum"])

        w_id = int(row["WTeamID"])
        l_id = int(row["LTeamID"])
        w_score = int(row["WScore"])
        l_score = int(row["LScore"])

        w_name = team_names.get(w_id, f"Team_{w_id}")
        l_name = team_names.get(l_id, f"Team_{l_id}")

        w_canonical, w_db_id = normalizer.resolve(w_name, "kaggle")
        l_canonical, l_db_id = normalizer.resolve(l_name, "kaggle")

        # Kaggle doesn't explicitly mark home/away in compact results
        # WLoc column: H = winner was home, A = winner was away, N = neutral
        w_loc = str(row.get("WLoc", "N"))
        if w_loc == "H":
            home_name, away_name = w_canonical, l_canonical
            home_id, away_id = w_db_id, l_db_id
            home_score, away_score = w_score, l_score
            neutral = 0
        elif w_loc == "A":
            home_name, away_name = l_canonical, w_canonical
            home_id, away_id = l_db_id, w_db_id
            home_score, away_score = l_score, w_score
            neutral = 0
        else:
            # Neutral or unknown — arbitrarily assign winner as "home"
            home_name, away_name = w_canonical, l_canonical
            home_id, away_id = w_db_id, l_db_id
            home_score, away_score = w_score, l_score
            neutral = 1

        # Tournament seeds
        home_seed = seeds.get((season, w_id if w_loc != "A" else l_id))
        away_seed = seeds.get((season, l_id if w_loc != "A" else w_id))

        # Conferences
        w_conf = conferences.get((season, w_id), "")
        l_conf = conferences.get((season, l_id), "")
        home_conf = w_conf if w_loc != "A" else l_conf
        away_conf = l_conf if w_loc != "A" else w_conf

        # Determine number of overtimes (if available)
        n_ot = int(row.get("NumOT", 0)) if "NumOT" in row.index else 0

        game_record = {
            "game_id": f"kaggle:{game_type}:{season}:{day_num}:{w_id}:{l_id}",
            "season": season,
            "date": f"{season}-{day_num:03d}",  # Day number within season
            "game_type": "ncaa_tourney" if game_type == "tourney" else "regular",
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team_name": home_name,
            "away_team_name": away_name,
            "neutral_site": neutral,
            "home_score": home_score,
            "away_score": away_score,
            "home_win": int(home_score > away_score),
            "num_overtimes": n_ot,
            "ncaa_seed_home": home_seed,
            "ncaa_seed_away": away_seed,
            "home_conference": home_conf,
            "away_conference": away_conf,
            "conference_game": int(home_conf == away_conf and home_conf != ""),
            "margin": home_score - away_score,
            "total_points": home_score + away_score,
            "era": get_era(season),
            "data_completeness_tier": 3,
            "data_source": "kaggle",
            "source_game_id": f"{season}_{day_num}_{w_id}_{l_id}",
        }

        try:
            conn.execute("""
                INSERT OR IGNORE INTO games (
                    game_id, season, date, game_type,
                    home_team_id, away_team_id, home_team_name, away_team_name,
                    neutral_site, home_score, away_score, home_win, num_overtimes,
                    ncaa_seed_home, ncaa_seed_away,
                    home_conference, away_conference, conference_game,
                    margin, total_points,
                    era, data_completeness_tier, data_source, source_game_id
                ) VALUES (
                    :game_id, :season, :date, :game_type,
                    :home_team_id, :away_team_id, :home_team_name, :away_team_name,
                    :neutral_site, :home_score, :away_score, :home_win, :num_overtimes,
                    :ncaa_seed_home, :ncaa_seed_away,
                    :home_conference, :away_conference, :conference_game,
                    :margin, :total_points,
                    :era, :data_completeness_tier, :data_source, :source_game_id
                )
            """, game_record)
            inserted += 1
        except Exception:
            pass

    conn.commit()
    print(f"  {inserted} {label.lower()} games inserted (Tier 3)")
    return inserted


def ingest_kaggle_detailed(conn: sqlite3.Connection, normalizer: TeamNormalizer,
                            files: dict, team_names: dict, seeds: dict,
                            conferences: dict, game_type: str = "regular"):
    """
    Ingest detailed (box score) game results.
    Available from 2003+ — these upgrade existing Tier 3 records to Tier 1.
    """
    if game_type == "regular":
        key = "reg_detailed"
        label = "Regular Season"
    else:
        key = "tourney_detailed"
        label = "NCAA Tournament"

    if key not in files:
        print(f"[Kaggle] No {label} detailed results file found")
        return 0

    print(f"[Kaggle] Ingesting {label} detailed results (box scores)...")
    df = pd.read_csv(files[key])

    updated = 0
    for _, row in df.iterrows():
        season = int(row["Season"])
        day_num = int(row["DayNum"])
        w_id = int(row["WTeamID"])
        l_id = int(row["LTeamID"])

        # Build game_id to match compact records
        game_id = f"kaggle:{game_type}:{season}:{day_num}:{w_id}:{l_id}"

        w_loc = str(row.get("WLoc", "N"))

        # Map winner/loser stats to home/away
        if w_loc == "A":
            # Winner was away, loser was home
            h_prefix, a_prefix = "L", "W"
        else:
            # Winner was home (or neutral)
            h_prefix, a_prefix = "W", "L"

        # Extract box score stats
        box_stats = {
            "home_fgm": _safe_int(row, f"{h_prefix}FGM"),
            "home_fga": _safe_int(row, f"{h_prefix}FGA"),
            "away_fgm": _safe_int(row, f"{a_prefix}FGM"),
            "away_fga": _safe_int(row, f"{a_prefix}FGA"),
            "home_fg3m": _safe_int(row, f"{h_prefix}FGM3"),
            "home_fg3a": _safe_int(row, f"{h_prefix}FGA3"),
            "away_fg3m": _safe_int(row, f"{a_prefix}FGM3"),
            "away_fg3a": _safe_int(row, f"{a_prefix}FGA3"),
            "home_ftm": _safe_int(row, f"{h_prefix}FTM"),
            "home_fta": _safe_int(row, f"{h_prefix}FTA"),
            "away_ftm": _safe_int(row, f"{a_prefix}FTM"),
            "away_fta": _safe_int(row, f"{a_prefix}FTA"),
            "home_or": _safe_int(row, f"{h_prefix}OR"),
            "home_dr": _safe_int(row, f"{h_prefix}DR"),
            "away_or": _safe_int(row, f"{a_prefix}OR"),
            "away_dr": _safe_int(row, f"{a_prefix}DR"),
            "home_ast": _safe_int(row, f"{h_prefix}Ast"),
            "home_to": _safe_int(row, f"{h_prefix}TO"),
            "home_stl": _safe_int(row, f"{h_prefix}Stl"),
            "home_blk": _safe_int(row, f"{h_prefix}Blk"),
            "home_pf": _safe_int(row, f"{h_prefix}PF"),
            "away_ast": _safe_int(row, f"{a_prefix}Ast"),
            "away_to": _safe_int(row, f"{a_prefix}TO"),
            "away_stl": _safe_int(row, f"{a_prefix}Stl"),
            "away_blk": _safe_int(row, f"{a_prefix}Blk"),
            "away_pf": _safe_int(row, f"{a_prefix}PF"),
        }

        # Determine tier
        has_full = all(v is not None for v in [
            box_stats["home_fgm"], box_stats["home_ast"],
            box_stats["home_to"], box_stats["home_or"]
        ])
        tier = 1 if has_full else 2

        # Update existing record with box score data
        set_clauses = []
        values = []
        for col, val in box_stats.items():
            if val is not None:
                set_clauses.append(f"{col} = ?")
                values.append(val)

        set_clauses.append("data_completeness_tier = ?")
        values.append(tier)
        set_clauses.append("updated_at = datetime('now')")
        values.append(game_id)

        conn.execute(
            f"UPDATE games SET {', '.join(set_clauses)} WHERE game_id = ?",
            values
        )
        updated += 1

    conn.commit()
    print(f"  {updated} {label.lower()} games upgraded to Tier {1}")
    return updated


def _safe_int(row, col):
    """Safely extract an integer from a DataFrame row."""
    if col not in row.index:
        return None
    val = row[col]
    if pd.isna(val):
        return None
    return int(val)


def ingest_all_kaggle(conn: sqlite3.Connection, normalizer: TeamNormalizer):
    """Run the full Kaggle ingestion pipeline."""
    print("\n" + "=" * 60)
    print("  KAGGLE DATA INGESTION")
    print("=" * 60)

    files = _check_kaggle_files()
    if not files:
        print("[Kaggle] No files found. Download the dataset first.")
        return 0

    team_names = _load_team_names(files)
    seeds = _load_seeds(files)
    conferences = _load_conferences(files)

    print(f"  Teams: {len(team_names)}")
    print(f"  Seeds: {len(seeds)} team-season entries")
    print(f"  Conferences: {len(conferences)} team-season entries")

    total = 0

    # Compact results (1985+, Tier 3)
    total += ingest_kaggle_compact(conn, normalizer, files, team_names, seeds, conferences, "regular")
    total += ingest_kaggle_compact(conn, normalizer, files, team_names, seeds, conferences, "tourney")

    # Detailed results (2003+, upgrades to Tier 1)
    ingest_kaggle_detailed(conn, normalizer, files, team_names, seeds, conferences, "regular")
    ingest_kaggle_detailed(conn, normalizer, files, team_names, seeds, conferences, "tourney")

    # Track source
    conn.execute("""
        INSERT INTO data_sources (source_name, games_loaded, seasons_covered, last_updated, notes)
        VALUES ('kaggle', ?, '1985-2024', datetime('now'), 'March Machine Learning Mania dataset')
        ON CONFLICT(source_name) DO UPDATE SET
            games_loaded = ?,
            last_updated = datetime('now')
    """, (total, total))
    conn.commit()

    print(f"\n[Kaggle] Total: {total} games ingested")
    return total


if __name__ == "__main__":
    db_path = str(Path(__file__).parent.parent / "output" / "basketball_god.db")
    conn = init_database(db_path)
    normalizer = TeamNormalizer()

    ingest_all_kaggle(conn, normalizer)

    print(f"\n{normalizer.get_unresolved_report()}")
    normalizer.save(str(Path(__file__).parent.parent / "output" / "team_names.json"))
    conn.close()
