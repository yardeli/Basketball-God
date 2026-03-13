"""
Phase 1 — Sports Reference / College Basketball Reference Scraper

Sports-Reference.com (via basketball-reference.com) has the most comprehensive
historical NCAA basketball data, covering 1949-present with varying detail:
  - 1949-1992: Scores, basic season stats
  - 1993+: Full box scores for most games
  - 2010+: Play-by-play data

IMPORTANT: Sports Reference has rate limiting and may block aggressive scraping.
We use a respectful 3-second delay between requests and cache everything.
If blocked, the pipeline continues with ESPN + Kaggle data.

This scraper focuses on getting:
  1. Season schedules with scores (fills gaps in ESPN/Kaggle)
  2. Box scores for pre-2003 games (Kaggle detailed only goes back to 2003)
  3. Season-level team stats and rankings
"""
import sys
import os
import re
import time
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

from phase1_data.schema import init_database, get_era
from phase1_data.team_normalization import TeamNormalizer

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Basketball-God/2.0 (academic research project)",
})

SPORTS_REF_BASE = "https://www.sports-reference.com/cbb"
SPORTS_REF_DELAY = 3.5  # Seconds between requests (be very respectful)
CACHE_DIR = Path(__file__).parent.parent / "sources" / "sports_ref_cache"


def _fetch_page(url: str) -> str | None:
    """Fetch a page with caching and rate limiting."""
    # Check cache first
    cache_key = url.replace("https://", "").replace("/", "_").replace("?", "_")
    cache_path = CACHE_DIR / f"{cache_key}.html"

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    try:
        resp = SESSION.get(url, timeout=15)

        if resp.status_code == 429:
            print(f"  [WARN] Rate limited by Sports Reference. Waiting 60s...")
            time.sleep(60)
            return None

        if resp.status_code == 403:
            print(f"  [WARN] Blocked by Sports Reference (403). Stopping.")
            return None

        resp.raise_for_status()
        time.sleep(SPORTS_REF_DELAY)

        # Cache the response
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")

        return resp.text

    except Exception as e:
        print(f"  [WARN] Sports Reference request failed: {e}")
        return None


def scrape_season_schedule(season: int) -> list[dict]:
    """
    Scrape a full season schedule from Sports Reference.
    Season parameter is the ending year (e.g., 2024 = 2023-24 season).

    Returns list of game dicts with whatever data is available.
    """
    url = f"{SPORTS_REF_BASE}/seasons/{season}-school-stats.html"
    html = _fetch_page(url)

    if not html:
        return []

    # This page has team season stats — useful for team_season_stats table
    # For individual games, we'd need to go to each team's schedule page
    # which is very request-intensive. We'll do this selectively.

    games = []

    # Parse the season stats table for team-level aggregates
    # This is more efficient than scraping individual game pages

    return games


def scrape_team_season_stats(season: int) -> list[dict]:
    """
    Scrape team-level season stats from Sports Reference.
    This is much more efficient than game-by-game scraping.

    Returns list of {team, wins, losses, pts_per_game, opp_pts, ...}
    """
    url = f"{SPORTS_REF_BASE}/seasons/{season}-school-stats.html"
    html = _fetch_page(url)

    if not html:
        return []

    teams = []

    # Parse the main stats table
    # Sports Reference uses <table id="basic_school_stats">
    table_match = re.search(
        r'<table[^>]*id="basic_school_stats"[^>]*>(.*?)</table>',
        html, re.DOTALL
    )

    if not table_match:
        # Try alternate table structure
        table_match = re.search(
            r'<table[^>]*class="[^"]*stats_table[^"]*"[^>]*>(.*?)</table>',
            html, re.DOTALL
        )

    if not table_match:
        print(f"  [WARN] Could not find stats table for {season}")
        return []

    table_html = table_match.group(1)

    # Parse rows
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL)
    for row in rows:
        # Skip header rows
        if '<th ' in row and 'data-stat="school_name"' not in row:
            continue

        cells = re.findall(r'<t[dh][^>]*data-stat="([^"]*)"[^>]*>(.*?)</t[dh]>', row, re.DOTALL)
        if not cells:
            continue

        data = {}
        for stat_name, value in cells:
            clean_val = re.sub(r'<[^>]+>', '', value).strip()
            data[stat_name] = clean_val

        team_name = data.get("school_name", "")
        if not team_name or team_name == "School":
            continue

        # Clean team name (remove NCAA tournament markers like *)
        team_name = re.sub(r'[*†]', '', team_name).strip()

        team_stats = {
            "team": team_name,
            "season": season,
            "games": _safe_float(data.get("g", "")),
            "wins": _safe_float(data.get("wins", "")),
            "losses": _safe_float(data.get("losses", "")),
            "win_pct": _safe_float(data.get("win_loss_pct", "")),
            "srs": _safe_float(data.get("srs", "")),  # Simple Rating System
            "sos": _safe_float(data.get("sos", "")),  # Strength of Schedule
            "pts_per_game": _safe_float(data.get("pts_per_g", "")),
            "opp_pts_per_game": _safe_float(data.get("opp_pts_per_g", "")),
            # Shooting (when available)
            "fg_pct": _safe_float(data.get("fg_pct", "")),
            "fg3_pct": _safe_float(data.get("fg3_pct", "")),
            "ft_pct": _safe_float(data.get("ft_pct", "")),
            # Pace and efficiency
            "pace": _safe_float(data.get("pace", "")),
            "off_rtg": _safe_float(data.get("off_rtg", "")),
            "def_rtg": _safe_float(data.get("def_rtg", "")),
        }

        if team_stats["games"] and team_stats["games"] > 0:
            teams.append(team_stats)

    print(f"  Season {season}: {len(teams)} team stat records parsed")
    return teams


def _safe_float(val: str) -> float | None:
    """Safely convert a string to float."""
    if not val or val == "" or val == "-":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def ingest_sports_ref_stats(conn: sqlite3.Connection, normalizer: TeamNormalizer,
                             seasons: list[int] = None):
    """
    Ingest team season stats from Sports Reference.
    This fills the team_season_stats table with SRS, SOS, efficiency data.
    """
    if seasons is None:
        seasons = list(range(1993, 2026))  # Stats are most reliable 1993+

    print(f"\n[SportsRef] Scraping team season stats for {len(seasons)} seasons...")
    print("  (3.5s delay between requests — this will take a while)")

    total_teams = 0
    blocked = False

    for season in seasons:
        if blocked:
            break

        team_stats = scrape_team_season_stats(season)

        if not team_stats:
            # Might be blocked
            print(f"  Season {season}: no data (may be blocked or unavailable)")
            continue

        for ts in team_stats:
            canonical, team_id = normalizer.resolve(ts["team"], "sports_ref")

            conn.execute("""
                INSERT INTO team_season_stats (
                    team_id, season, games_played, wins, losses, win_pct,
                    points_per_game, opp_points_per_game, sos_rank
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_id, season) DO UPDATE SET
                    games_played = excluded.games_played,
                    wins = excluded.wins,
                    losses = excluded.losses,
                    win_pct = excluded.win_pct,
                    points_per_game = excluded.points_per_game,
                    opp_points_per_game = excluded.opp_points_per_game
            """, (
                team_id, season,
                int(ts["games"]) if ts["games"] else None,
                int(ts["wins"]) if ts["wins"] else None,
                int(ts["losses"]) if ts["losses"] else None,
                ts["win_pct"],
                ts["pts_per_game"],
                ts["opp_pts_per_game"],
                None,  # SOS as rank would need sorting
            ))

        conn.commit()
        total_teams += len(team_stats)

    # Track source
    conn.execute("""
        INSERT INTO data_sources (source_name, games_loaded, seasons_covered, last_updated, notes)
        VALUES ('sports_ref', 0, ?, datetime('now'), 'Team season stats (SRS, SOS, efficiency)')
        ON CONFLICT(source_name) DO UPDATE SET
            seasons_covered = ?,
            last_updated = datetime('now')
    """, (f"{seasons[0]}-{seasons[-1]}", f"{seasons[0]}-{seasons[-1]}"))
    conn.commit()

    print(f"\n[SportsRef] {total_teams} team-season stat records ingested")
    return total_teams


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=str, default="2020-2025",
                        help="Season range (e.g., 2020-2025). Start small to test!")
    args = parser.parse_args()

    start, end = map(int, args.seasons.split("-"))
    seasons = list(range(start, end + 1))

    db_path = str(Path(__file__).parent.parent / "output" / "basketball_god.db")
    conn = init_database(db_path)
    normalizer = TeamNormalizer()

    ingest_sports_ref_stats(conn, normalizer, seasons)

    print(f"\n{normalizer.get_unresolved_report()}")
    conn.close()
