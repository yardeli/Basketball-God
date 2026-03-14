"""
Basketball-God — Live Odds Fetcher
===================================
Pulls NCAAB championship futures from The Odds API (free tier).
Converts American odds to implied probability and compares against
Basketball-God model predictions.
"""

import os
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ODDS_API_BASE    = "https://api.the-odds-api.com/v4"
SPORT            = "basketball_ncaab"
CACHE_FILE       = Path(__file__).parent / "phase5_deploy" / "output" / "odds_cache.json"
GAMES_CACHE_DIR  = Path(__file__).parent / "phase5_deploy" / "output"
CACHE_TTL        = 3600   # 1 hour for championship futures
GAMES_CACHE_TTL  = 1800   # 30 min for daily game lines


# ── American odds utilities ────────────────────────────────────────────────────

def american_to_implied(american: int) -> float:
    """Convert American odds to implied probability (0-1)."""
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def implied_to_american(prob: float) -> str:
    """Convert implied probability to American odds string."""
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob >= 0.5:
        american = -(prob / (1 - prob)) * 100
        return f"{int(american)}"
    else:
        american = ((1 - prob) / prob) * 100
        return f"+{int(american)}"


def format_american(american: int) -> str:
    return f"+{american}" if american > 0 else str(american)


def remove_vig(probs: list[float]) -> list[float]:
    """Remove bookmaker vig (overround) to get fair implied probabilities."""
    total = sum(probs)
    if total == 0:
        return probs
    return [p / total for p in probs]


# ── API calls ──────────────────────────────────────────────────────────────────

def _get_api_key() -> Optional[str]:
    key = os.environ.get("ODDS_API_KEY") or os.environ.get("BETTING_ODS_API")
    if not key or key == "your_api_key_here":
        return None
    return key


def fetch_championship_odds(force_refresh: bool = False) -> dict:
    """
    Fetch NCAAB championship futures from The Odds API.

    Returns a dict with:
        {
            "teams": [{"name", "best_odds", "best_book", "implied_prob", "fair_prob"}, ...],
            "requests_remaining": int,
            "cached": bool,
            "error": str or None,
            "fetched_at": timestamp,
        }

    Falls back gracefully on any error.
    """
    # Check cache first
    if not force_refresh and CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            age = time.time() - cached.get("fetched_at", 0)
            if age < CACHE_TTL:
                cached["cached"] = True
                return cached
        except Exception:
            pass

    api_key = _get_api_key()
    if not api_key:
        return _error_response("ODDS_API_KEY not set in .env")

    try:
        # Try outrights (championship futures) first
        url = f"{ODDS_API_BASE}/sports/{SPORT}/odds/"
        params = {
            "apiKey":     api_key,
            "regions":    "us",
            "markets":    "outrights",
            "oddsFormat": "american",
        }
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code == 401:
            return _error_response("Invalid API key")
        if resp.status_code == 429:
            return _error_response("Monthly API quota exhausted (500 req/month free tier)")

        if resp.status_code == 422 or (resp.status_code == 200 and not resp.json()):
            # Outrights not available — try alternate sports key for tournament futures
            # The Odds API sometimes lists the tournament separately
            alt_sports = [
                "basketball_ncaab_championship_winner",
                "basketball_ncaab",
            ]
            for alt in alt_sports:
                alt_url = f"{ODDS_API_BASE}/sports/{alt}/odds/"
                alt_params = {**params, "markets": "h2h"}
                alt_resp = requests.get(alt_url, params=alt_params, timeout=10)
                if alt_resp.status_code == 200 and alt_resp.json():
                    resp = alt_resp
                    break
            else:
                return _error_response(
                    "Futures market not live yet — check back closer to Selection Sunday"
                )

        if resp.status_code != 200:
            return _error_response(f"API error {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        remaining = int(resp.headers.get("x-requests-remaining", -1))
        used      = int(resp.headers.get("x-requests-used", -1))

        result = _parse_championship_odds(data, remaining, used)
        result["cached"] = False

        # Save to cache
        try:
            CACHE_FILE.write_text(json.dumps(result, indent=2))
        except Exception:
            pass

        return result

    except requests.exceptions.ConnectionError:
        return _load_cache_or_error("No internet connection")
    except requests.exceptions.Timeout:
        return _load_cache_or_error("API request timed out")
    except Exception as e:
        return _load_cache_or_error(f"Unexpected error: {e}")


def _parse_championship_odds(data: list, remaining: int, used: int) -> dict:
    """Parse raw API response into clean team odds structure."""
    # Aggregate best odds per team across all bookmakers
    team_odds: dict[str, list[int]] = {}
    team_books: dict[str, str] = {}

    for event in data:
        for bookmaker in event.get("bookmakers", []):
            book_name = bookmaker.get("title", "")
            for market in bookmaker.get("markets", []):
                if market.get("key") not in ("outrights", "h2h"):
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    if name and price is not None:
                        try:
                            price = int(price)
                        except (ValueError, TypeError):
                            continue
                        if name not in team_odds:
                            team_odds[name] = []
                        team_odds[name].append((price, book_name))

    # Find best (most favorable) odds per team = highest American odds
    teams = []
    for name, price_list in team_odds.items():
        best_price, best_book = max(price_list, key=lambda x: x[0])
        implied = american_to_implied(best_price)
        teams.append({
            "name":         name,
            "best_odds":    best_price,
            "best_odds_fmt": format_american(best_price),
            "best_book":    best_book,
            "implied_prob": round(implied, 4),
            "all_prices":   [p for p, _ in price_list],
        })

    # Remove vig from implied probs
    raw_probs = [t["implied_prob"] for t in teams]
    fair_probs = remove_vig(raw_probs)
    for t, fp in zip(teams, fair_probs):
        t["fair_prob"] = round(fp, 4)

    # Sort by fair probability descending
    teams.sort(key=lambda x: -x["fair_prob"])

    return {
        "teams":               teams,
        "requests_remaining":  remaining,
        "requests_used":       used,
        "n_events":            len(data),
        "error":               None,
        "fetched_at":          time.time(),
    }


def _error_response(msg: str) -> dict:
    return {
        "teams":              [],
        "requests_remaining": -1,
        "requests_used":      -1,
        "n_events":           0,
        "error":              msg,
        "fetched_at":         time.time(),
        "cached":             False,
    }


def _load_cache_or_error(msg: str) -> dict:
    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            cached["cached"] = True
            cached["error"]  = f"[stale cache] {msg}"
            return cached
        except Exception:
            pass
    return _error_response(msg)


# ── Model-vs-market comparison ─────────────────────────────────────────────────

def compare_model_to_market(model_probs: list[tuple[str, float]],
                             market_data: dict) -> list[dict]:
    """
    Align model champion probabilities with market odds.

    model_probs: [(team_name, probability), ...]
    Returns list of dicts with model, market, and edge columns.
    """
    market_by_name = {}
    for t in market_data.get("teams", []):
        # Normalize name for fuzzy matching
        key = t["name"].lower().replace("state", "st").replace("  ", " ").strip()
        market_by_name[key] = t

    results = []
    for model_name, model_prob in model_probs:
        norm = model_name.lower().replace("state", "st").replace("  ", " ").strip()

        # Try exact match first, then partial
        market_team = market_by_name.get(norm)
        if not market_team:
            for mkey, mval in market_by_name.items():
                if norm in mkey or mkey in norm:
                    market_team = mval
                    break

        if market_team:
            market_prob = market_team["fair_prob"]
            edge = model_prob - market_prob
            market_odds = market_team["best_odds_fmt"]
            agreement = "AGREE" if abs(edge) < 0.03 else (
                "MODEL HIGHER" if edge > 0 else "MARKET HIGHER"
            )
        else:
            market_prob = None
            edge = None
            market_odds = "N/A"
            agreement = "NO MARKET"

        results.append({
            "team":         model_name,
            "model_prob":   round(model_prob, 4),
            "model_odds":   implied_to_american(model_prob),
            "market_odds":  market_odds,
            "market_prob":  market_prob,
            "edge":         round(edge, 4) if edge is not None else None,
            "signal":       agreement,
        })

    return results


# ── Today's game lines ─────────────────────────────────────────────────────────

def fetch_todays_games(force_refresh: bool = False) -> dict:
    """
    Fetch today's NCAAB games with h2h, spread, and totals from The Odds API.

    Returns:
        {
            "games":               [...],
            "date":                "YYYY-MM-DD",
            "n_games":             int,
            "requests_remaining":  int,
            "cached":              bool,
            "error":               str or None,
        }

    Each game dict:
        {
            "id", "home_team", "away_team", "commence_time",
            "h2h":    {"home": {"price": int, "best_book": str},
                       "away": {"price": int, "best_book": str}},
            "spread": {"home": {"price": int, "point": float, "best_book": str},
                       "away": {"price": int, "point": float, "best_book": str}},
            "total":  {"over":  {"price": int, "point": float, "best_book": str},
                       "under": {"price": int, "point": float, "best_book": str}},
        }
    """
    # Use PST (UTC-8) date so cache resets at midnight local time, not midnight UTC
    pst_now    = datetime.now(timezone.utc) - timedelta(hours=8)
    today_str  = pst_now.strftime("%Y-%m-%d")
    cache_path = GAMES_CACHE_DIR / f"games_cache_{today_str}.json"

    if not force_refresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            age    = time.time() - cached.get("fetched_at", 0)
            if age < GAMES_CACHE_TTL:
                cached["cached"] = True
                return cached
        except Exception:
            pass

    api_key = _get_api_key()
    if not api_key:
        return _games_error("ODDS_API_KEY not set in .env")

    try:
        url    = f"{ODDS_API_BASE}/sports/{SPORT}/odds/"
        params = {
            "apiKey":       api_key,
            "regions":      "us",
            "markets":      "h2h,spreads,totals",
            "oddsFormat":   "american",
            "dateFormat":   "iso",
        }
        resp = requests.get(url, params=params, timeout=15)

        if resp.status_code == 401:
            return _games_load_cache_or_error(cache_path, "Invalid API key — showing cached games")
        if resp.status_code == 429:
            return _games_load_cache_or_error(cache_path, "API quota exhausted — showing cached games")
        if resp.status_code != 200:
            return _games_error(f"API error {resp.status_code}")

        events    = resp.json()
        remaining = int(resp.headers.get("x-requests-remaining", -1))

        games = []
        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            game = {
                "id":            event.get("id", ""),
                "home_team":     home,
                "away_team":     away,
                "commence_time": event.get("commence_time", ""),
                "h2h":    {},
                "spread": {},
                "total":  {},
            }
            _parse_event_markets(event, game)
            games.append(game)

        result = {
            "games":              games,
            "date":               today_str,
            "n_games":            len(games),
            "requests_remaining": remaining,
            "cached":             False,
            "error":              None,
            "fetched_at":         time.time(),
        }
        try:
            cache_path.write_text(json.dumps(result, indent=2))
        except Exception:
            pass
        return result

    except requests.exceptions.ConnectionError:
        return _games_load_cache_or_error(cache_path, "No internet connection")
    except requests.exceptions.Timeout:
        return _games_load_cache_or_error(cache_path, "API request timed out")
    except Exception as e:
        return _games_load_cache_or_error(cache_path, f"Unexpected error: {e}")


def _parse_event_markets(event: dict, game: dict) -> None:
    """Aggregate best odds per side/market across all bookmakers."""
    # Accumulators: key → list of (price, book_name)
    h2h_prices:    dict[str, list] = {}   # "home" / "away"
    spread_prices: dict[str, list] = {}   # "home" / "away"
    spread_points: dict[str, list] = {}
    total_prices:  dict[str, list] = {}   # "over" / "under"
    total_points:  dict[str, list] = {}

    home = game["home_team"]
    away = game["away_team"]

    for bookmaker in event.get("bookmakers", []):
        book = bookmaker.get("title", "")
        for market in bookmaker.get("markets", []):
            key = market.get("key")
            for outcome in market.get("outcomes", []):
                name  = outcome.get("name", "")
                price = outcome.get("price")
                point = outcome.get("point")
                if price is None:
                    continue
                try:
                    price = int(price)
                except (ValueError, TypeError):
                    continue

                if key == "h2h":
                    side = "home" if name == home else ("away" if name == away else None)
                    if side:
                        h2h_prices.setdefault(side, []).append((price, book))

                elif key == "spreads":
                    side = "home" if name == home else ("away" if name == away else None)
                    if side and point is not None:
                        spread_prices.setdefault(side, []).append((price, book))
                        spread_points.setdefault(side, []).append(float(point))

                elif key == "totals":
                    side = name.lower()   # "over" or "under"
                    if side in ("over", "under") and point is not None:
                        total_prices.setdefault(side, []).append((price, book))
                        total_points.setdefault(side, []).append(float(point))

    # Best h2h = highest American odds per side
    for side, prices in h2h_prices.items():
        best_price, best_book = max(prices, key=lambda x: x[0])
        game["h2h"][side] = {"price": best_price, "best_book": best_book}

    # Best spread = highest odds at most-common line
    for side, prices in spread_prices.items():
        points = spread_points.get(side, [])
        if not points:
            continue
        consensus = max(set(points), key=points.count)
        candidates = [(pr, bk) for (pr, bk), pt in zip(prices, points) if pt == consensus]
        if candidates:
            best_price, best_book = max(candidates, key=lambda x: x[0])
            game["spread"][side] = {"price": best_price, "point": consensus, "best_book": best_book}

    # Best total = highest odds at most-common line
    for side, prices in total_prices.items():
        points = total_points.get(side, [])
        if not points:
            continue
        consensus = max(set(points), key=points.count)
        candidates = [(pr, bk) for (pr, bk), pt in zip(prices, points) if pt == consensus]
        if candidates:
            best_price, best_book = max(candidates, key=lambda x: x[0])
            game["total"][side] = {"price": best_price, "point": consensus, "best_book": best_book}


def _games_error(msg: str) -> dict:
    return {
        "games": [], "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "n_games": 0, "requests_remaining": -1,
        "cached": False, "error": msg, "fetched_at": time.time(),
    }


def _games_load_cache_or_error(cache_path: Path, msg: str) -> dict:
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            cached["cached"] = True
            cached["error"]  = f"[stale] {msg}"
            return cached
        except Exception:
            pass
    return _games_error(msg)
