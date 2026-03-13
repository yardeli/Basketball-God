"""
Data Scraper — Pull real NCAA D1 men's basketball data from ESPN's public API.

Collects:
  - All D1 team info (363 teams, conferences, IDs)
  - Game-by-game scores for full seasons
  - Team season stats and standings

ESPN's API is free, no authentication required.
We rate-limit requests to be respectful.
"""
import json
import time
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": config.USER_AGENT})


def _get(url: str, params: dict = None) -> dict | None:
    """Make a rate-limited GET request to ESPN API."""
    try:
        resp = SESSION.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        time.sleep(config.REQUEST_DELAY)
        return resp.json()
    except Exception as e:
        print(f"  [WARN] Request failed: {url} — {e}")
        time.sleep(2)
        return None


# ─── TEAMS ───────────────────────────────────────────────────────────────────

def scrape_all_teams() -> pd.DataFrame:
    """Fetch all D1 men's basketball teams from ESPN."""
    print("[Scraper] Fetching all D1 teams...")

    all_teams = []
    page = 1
    while True:
        data = _get(config.ESPN_TEAMS, params={"limit": 100, "page": page, "groups": config.ESPN_GROUPS_D1})
        if not data:
            break

        teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        if not teams:
            break

        for entry in teams:
            team = entry.get("team", {})
            all_teams.append({
                "espn_id": int(team.get("id", 0)),
                "name": team.get("displayName", ""),
                "abbreviation": team.get("abbreviation", ""),
                "short_name": team.get("shortDisplayName", ""),
                "color": team.get("color", ""),
                "logo": team.get("logos", [{}])[0].get("href", "") if team.get("logos") else "",
                "conference": team.get("groups", {}).get("parent", {}).get("shortName", "")
                              if team.get("groups") else "",
                "conference_id": team.get("groups", {}).get("parent", {}).get("id", "")
                                if team.get("groups") else "",
            })

        if len(teams) < 100:
            break
        page += 1

    df = pd.DataFrame(all_teams)
    print(f"  Found {len(df)} teams")

    # Save
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DIR / "teams.csv", index=False)
    return df


# ─── SCOREBOARD / GAMES ─────────────────────────────────────────────────────

def scrape_season_games(season: int, teams_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Scrape all D1 games for a season by iterating day-by-day through the scoreboard.
    Season parameter is the ending year (e.g., 2026 = 2025-26 season).
    """
    print(f"[Scraper] Scraping games for {season-1}-{str(season)[2:]} season...")

    # Season typically runs Nov 1 → April 10
    start_date = datetime(season - 1, 11, 1)
    end_date = datetime(season, 4, 10)

    # For current season, stop at today
    today = datetime.now()
    if end_date > today:
        end_date = today - timedelta(days=1)

    all_games = []
    current = start_date
    days_scraped = 0
    total_days = (end_date - start_date).days

    while current <= end_date:
        date_str = current.strftime("%Y%m%d")

        data = _get(config.ESPN_SCOREBOARD, params={
            "dates": date_str,
            "groups": config.ESPN_GROUPS_D1,
            "limit": 200,
        })

        if data:
            events = data.get("events", [])
            for event in events:
                game = _parse_game_event(event, season)
                if game:
                    all_games.append(game)

        days_scraped += 1
        if days_scraped % 14 == 0:
            pct = days_scraped / total_days * 100
            print(f"  {current.strftime('%Y-%m-%d')} — {len(all_games)} games so far ({pct:.0f}%)")

        current += timedelta(days=1)

    df = pd.DataFrame(all_games)
    if len(df) > 0:
        df = df.drop_duplicates(subset=["game_id"])
        df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

    print(f"  Season {season}: {len(df)} games scraped")

    # Save
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DIR / f"games_{season}.csv", index=False)
    return df


def _parse_game_event(event: dict, season: int) -> dict | None:
    """Parse an ESPN event JSON into a flat game record."""
    try:
        game_id = event.get("id", "")
        status = event.get("status", {}).get("type", {}).get("name", "")

        # Only include completed games
        if status != "STATUS_FINAL":
            return None

        date_str = event.get("date", "")[:10]
        competitions = event.get("competitions", [{}])
        if not competitions:
            return None

        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            return None

        # ESPN lists home team first typically
        teams = {}
        for c in competitors:
            ha = c.get("homeAway", "")
            team_info = c.get("team", {})
            teams[ha] = {
                "id": int(team_info.get("id", 0)),
                "name": team_info.get("displayName", ""),
                "abbreviation": team_info.get("abbreviation", ""),
                "score": int(c.get("score", 0)),
                "winner": c.get("winner", False),
                "conference": team_info.get("conferenceId", ""),
            }

        home = teams.get("home", {})
        away = teams.get("away", {})

        if not home or not away:
            return None

        # Determine if neutral site
        neutral = comp.get("neutralSite", False)

        # Conference game?
        conf_game = comp.get("conferenceCompetition", False)

        return {
            "game_id": game_id,
            "season": season,
            "date": date_str,
            "home_id": home["id"],
            "home_team": home["name"],
            "home_abbrev": home.get("abbreviation", ""),
            "home_score": home["score"],
            "away_id": away["id"],
            "away_team": away["name"],
            "away_abbrev": away.get("abbreviation", ""),
            "away_score": away["score"],
            "home_win": int(home["score"] > away["score"]),
            "margin": home["score"] - away["score"],
            "total_points": home["score"] + away["score"],
            "neutral_site": int(neutral),
            "conference_game": int(conf_game),
        }

    except Exception as e:
        return None


# ─── MULTI-SEASON ────────────────────────────────────────────────────────────

def scrape_all_seasons(seasons: list[int] = None) -> pd.DataFrame:
    """Scrape games for multiple seasons and combine."""
    if seasons is None:
        seasons = config.HISTORICAL_SEASONS

    teams_df = scrape_all_teams()

    all_dfs = []
    for season in seasons:
        cache_path = config.RAW_DIR / f"games_{season}.csv"

        # Use cache if exists and season is complete
        if cache_path.exists() and season < config.CURRENT_SEASON:
            print(f"[Scraper] Loading cached {season} season...")
            df = pd.read_csv(cache_path)
        else:
            df = scrape_season_games(season, teams_df)

        if len(df) > 0:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(config.PROCESSED_DIR / "all_games.csv", index=False)
        print(f"\n[Scraper] Total: {len(combined)} games across {len(seasons)} seasons")
        return combined

    return pd.DataFrame()


# ─── QUICK FETCH (for predictions) ──────────────────────────────────────────

def fetch_today_games() -> list[dict]:
    """Fetch today's scheduled games."""
    today = datetime.now().strftime("%Y%m%d")
    data = _get(config.ESPN_SCOREBOARD, params={
        "dates": today,
        "groups": config.ESPN_GROUPS_D1,
        "limit": 200,
    })

    if not data:
        return []

    games = []
    for event in data.get("events", []):
        competitions = event.get("competitions", [{}])
        if not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        teams = {}
        for c in competitors:
            ha = c.get("homeAway", "")
            team_info = c.get("team", {})
            teams[ha] = {
                "id": int(team_info.get("id", 0)),
                "name": team_info.get("displayName", ""),
                "score": int(c.get("score", 0)) if c.get("score") else 0,
            }

        status = event.get("status", {}).get("type", {}).get("name", "")
        games.append({
            "game_id": event.get("id", ""),
            "date": event.get("date", "")[:10],
            "home_team": teams.get("home", {}).get("name", ""),
            "home_id": teams.get("home", {}).get("id", 0),
            "away_team": teams.get("away", {}).get("name", ""),
            "away_id": teams.get("away", {}).get("id", 0),
            "status": status,
            "neutral_site": int(comp.get("neutralSite", False)),
        })

    return games


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", action="store_true", help="Scrape teams only")
    parser.add_argument("--season", type=int, help="Scrape a specific season")
    parser.add_argument("--all", action="store_true", help="Scrape all seasons")
    parser.add_argument("--today", action="store_true", help="Show today's games")
    args = parser.parse_args()

    if args.teams:
        scrape_all_teams()
    elif args.season:
        scrape_all_teams()
        scrape_season_games(args.season)
    elif args.today:
        games = fetch_today_games()
        for g in games:
            print(f"  {g['away_team']:30s} @ {g['home_team']:30s}  [{g['status']}]")
        print(f"\n  {len(games)} games today")
    elif args.all:
        scrape_all_seasons()
    else:
        print("Usage: python data_scraper.py --teams | --season 2026 | --all | --today")
