"""
Phase 1 — ESPN Data Ingestion

Ingest game data from ESPN's public API into the unified schema.
ESPN covers ~2002 to present with score data.
Some seasons back to ~2000 are available with limited data.

This builds on the existing data_scraper.py but writes to SQLite
and adds data completeness tracking.
"""
import sys
import os
import sqlite3

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from pathlib import Path

from phase1_data.schema import init_database, get_era, compute_completeness_tier
from phase1_data.team_normalization import TeamNormalizer
from data_scraper import scrape_all_teams, scrape_season_games

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def ingest_espn_teams(conn: sqlite3.Connection, normalizer: TeamNormalizer):
    """Scrape ESPN teams and insert into database."""
    print("[ESPN] Ingesting teams...")

    teams_df = scrape_all_teams()

    for _, row in teams_df.iterrows():
        name = row["name"]
        canonical, team_id = normalizer.resolve(name, source="espn")

        # Upsert team
        conn.execute("""
            INSERT INTO teams (team_id, canonical_name, espn_id, abbreviation, conference, is_d1)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(canonical_name) DO UPDATE SET
                espn_id = excluded.espn_id,
                abbreviation = excluded.abbreviation,
                conference = excluded.conference
        """, (team_id, canonical, int(row["espn_id"]), row.get("abbreviation", ""),
              row.get("conference", "")))

        # Add aliases
        for alias in [name, row.get("short_name", ""), row.get("abbreviation", "")]:
            if alias:
                conn.execute("""
                    INSERT OR IGNORE INTO team_aliases (alias, team_id, source)
                    VALUES (?, ?, 'espn')
                """, (alias, team_id))

    conn.commit()
    print(f"  {len(teams_df)} ESPN teams ingested")


def ingest_espn_season(conn: sqlite3.Connection, season: int,
                        normalizer: TeamNormalizer, force: bool = False):
    """
    Scrape one season of ESPN games and insert into database.
    Skips seasons already loaded unless force=True.
    """
    # Check if already loaded
    cursor = conn.execute(
        "SELECT COUNT(*) FROM games WHERE season = ? AND data_source = 'espn'",
        (season,)
    )
    existing = cursor.fetchone()[0]
    if existing > 0 and not force:
        print(f"[ESPN] Season {season}: {existing} games already loaded, skipping")
        return existing

    # Check for cached CSV first
    csv_path = Path(__file__).parent.parent.parent / "data" / "raw" / f"games_{season}.csv"
    if csv_path.exists():
        print(f"[ESPN] Loading season {season} from cache...")
        games_df = pd.read_csv(csv_path)
    else:
        games_df = scrape_season_games(season)

    if len(games_df) == 0:
        print(f"[ESPN] Season {season}: no games found")
        return 0

    era = get_era(season)
    inserted = 0

    for _, game in games_df.iterrows():
        home_name = str(game.get("home_team", ""))
        away_name = str(game.get("away_team", ""))

        home_canonical, home_id = normalizer.resolve(home_name, "espn")
        away_canonical, away_id = normalizer.resolve(away_name, "espn")

        home_score = int(game.get("home_score", 0))
        away_score = int(game.get("away_score", 0))

        game_record = {
            "game_id": f"espn:{game.get('game_id', '')}",
            "season": season,
            "date": str(game.get("date", "")),
            "game_type": "regular",  # ESPN scoreboard is mostly regular season
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team_name": home_canonical,
            "away_team_name": away_canonical,
            "neutral_site": int(game.get("neutral_site", 0)),
            "home_score": home_score,
            "away_score": away_score,
            "home_win": int(home_score > away_score),
            "margin": home_score - away_score,
            "total_points": home_score + away_score,
            "conference_game": int(game.get("conference_game", 0)),
            "era": era,
            "data_completeness_tier": 3,  # ESPN scoreboard = score only
            "data_source": "espn",
            "source_game_id": str(game.get("game_id", "")),
        }

        try:
            conn.execute("""
                INSERT OR IGNORE INTO games (
                    game_id, season, date, game_type,
                    home_team_id, away_team_id, home_team_name, away_team_name,
                    neutral_site, home_score, away_score, home_win,
                    margin, total_points, conference_game,
                    era, data_completeness_tier, data_source, source_game_id
                ) VALUES (
                    :game_id, :season, :date, :game_type,
                    :home_team_id, :away_team_id, :home_team_name, :away_team_name,
                    :neutral_site, :home_score, :away_score, :home_win,
                    :margin, :total_points, :conference_game,
                    :era, :data_completeness_tier, :data_source, :source_game_id
                )
            """, game_record)
            inserted += 1
        except Exception as e:
            pass  # Duplicate game IDs are expected

    conn.commit()
    print(f"[ESPN] Season {season}: {inserted} games inserted")

    # Update data source tracking
    conn.execute("""
        INSERT INTO data_sources (source_name, games_loaded, seasons_covered, last_updated)
        VALUES ('espn', ?, ?, datetime('now'))
        ON CONFLICT(source_name) DO UPDATE SET
            games_loaded = games_loaded + ?,
            last_updated = datetime('now')
    """, (inserted, str(season), inserted))
    conn.commit()

    return inserted


def ingest_all_espn(conn: sqlite3.Connection, normalizer: TeamNormalizer,
                    seasons: list[int] = None, force: bool = False):
    """Ingest all ESPN seasons."""
    if seasons is None:
        # ESPN has decent data from ~2002 onward
        seasons = list(range(2003, 2027))

    print(f"\n[ESPN] Ingesting {len(seasons)} seasons ({seasons[0]}-{seasons[-1]})...")

    # Teams first
    ingest_espn_teams(conn, normalizer)

    total = 0
    for season in seasons:
        count = ingest_espn_season(conn, season, normalizer, force=force)
        total += count

    print(f"\n[ESPN] Total: {total} games ingested across {len(seasons)} seasons")
    return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=str, default="2003-2026",
                        help="Season range (e.g., 2003-2026)")
    parser.add_argument("--force", action="store_true", help="Re-ingest existing seasons")
    args = parser.parse_args()

    start, end = map(int, args.seasons.split("-"))
    seasons = list(range(start, end + 1))

    db_path = str(Path(__file__).parent.parent / "output" / "basketball_god.db")
    conn = init_database(db_path)
    normalizer = TeamNormalizer()

    # Load ESPN teams for name resolution
    teams_csv = Path(__file__).parent.parent.parent / "data" / "raw" / "teams.csv"
    if teams_csv.exists():
        normalizer.load_espn_teams(str(teams_csv))

    ingest_all_espn(conn, normalizer, seasons, force=args.force)

    # Save unresolved names
    print(f"\n{normalizer.get_unresolved_report()}")
    normalizer.save(str(Path(__file__).parent.parent / "output" / "team_names.json"))

    conn.close()
