"""
season_stats_store.py — Live season rolling stats for Basketball-God 2.0

Fetches completed NCAAB games for the current season from ESPN's public API,
computes rolling team stats (the same feature vector as the trained model),
and serves matchup feature dicts for daily predictions.

Key improvements over zero-fill:
  - Real Tier 1 stats (win%, margin, rest, win streak) from all 6K+ season games
  - Real Tier 2 box score stats (eFG%, TO rate, ORB, pace) from game summaries
  - Elo-based power rankings computed from season results as NET rank proxy
  - League-average defaults (from league_averages.json) replace zero for sparse teams

Usage:
    store = SeasonStatsStore()
    store.refresh()                         # fetch + compute (cached, call once/day)
    features = store.get_matchup_features("Duke", "NC State")
"""

import json
import time
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ESPN_BASE       = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_WEB_BASE   = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_SCOREBOARD = f"{ESPN_BASE}/scoreboard"
ESPN_SUMMARY    = f"{ESPN_WEB_BASE}/summary"

REQUEST_DELAY   = 0.35
REQUEST_TIMEOUT = 12

CURRENT_SEASON  = 2026
SEASON_START    = date(2025, 11, 1)
ROLLING_WINDOW  = 10
SUMMARY_LOOKBACK = 15    # Fetch full box-score summaries for the last N games per team
SUMMARY_MAX_FETCH = 200  # Hard cap on total summary API calls per refresh

# ── League-average fallbacks ──────────────────────────────────────────────────
_DEFAULT_LEAGUE_AVGS: dict = {}


def _load_league_averages() -> dict:
    global _DEFAULT_LEAGUE_AVGS
    p = ROOT / "data" / "league_averages.json"
    if p.exists():
        _DEFAULT_LEAGUE_AVGS = json.loads(p.read_text())
    return _DEFAULT_LEAGUE_AVGS


def _lg(feature: str, fallback: float = 0.0) -> float:
    return _DEFAULT_LEAGUE_AVGS.get(feature, fallback)


# ── ESPN helpers ──────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT,
                         headers={"User-Agent": "Basketball-God/2.0 (research)"})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[SeasonStatsStore] GET failed {url}: {e}", file=sys.stderr)
        return None


def _fetch_scoreboard_day(game_date: date) -> list[dict]:
    """Return completed game dicts for a given date (scoreboard-level stats only)."""
    params = {"dates": game_date.strftime("%Y%m%d"), "groups": 50, "limit": 200}
    data = _get(ESPN_SCOREBOARD, params)
    if not data:
        return []

    games = []
    for evt in data.get("events", []):
        comp = evt.get("competitions", [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed", False):
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        home_score = int(home.get("score", 0) or 0)
        away_score = int(away.get("score", 0) or 0)
        if home_score == 0 and away_score == 0:
            continue

        games.append({
            "game_id":    evt.get("id"),
            "date":       game_date.isoformat(),
            "home_id":    home.get("team", {}).get("id"),
            "home_name":  home.get("team", {}).get("displayName", ""),
            "away_id":    away.get("team", {}).get("id"),
            "away_name":  away.get("team", {}).get("displayName", ""),
            "home_score": home_score,
            "away_score": away_score,
            "home_won":   home_score > away_score,
            # Box score stats populated later by _enrich_with_summaries
            "home_stats": _parse_scoreboard_stats(home),
            "away_stats": _parse_scoreboard_stats(away),
            "has_summary": False,
        })
    return games


def _parse_scoreboard_stats(competitor: dict) -> dict:
    """Parse stats available in the scoreboard endpoint (partial — no TO/ORB/DRB/STL/BLK)."""
    raw = {}
    for s in competitor.get("statistics", []):
        key = s.get("name", "")
        val = s.get("displayValue", "")
        try:
            raw[key] = float(val.replace("%", "").replace(",", ""))
        except (ValueError, AttributeError):
            raw[key] = None
    return raw


def _fetch_game_summary(game_id: str) -> Optional[dict]:
    """Fetch full box score from ESPN summary endpoint. Returns {home_stats, away_stats}."""
    data = _get(ESPN_SUMMARY, {"event": game_id})
    if not data:
        return None

    bx = data.get("boxscore", {})
    teams = bx.get("teams", [])
    if len(teams) < 2:
        return None

    # ESPN summary lists away team first, home team second
    result = {}
    for t in teams:
        home_away = t.get("homeAway", "")
        stats_raw = {}
        for s in t.get("statistics", []):
            name = s.get("name", "")
            val  = s.get("displayValue", "")
            # Handle compound format "made-attempted"
            if "-" in val:
                parts = val.split("-")
                try:
                    stats_raw[name + "Made"]      = float(parts[0])
                    stats_raw[name + "Attempted"] = float(parts[1])
                except ValueError:
                    pass
            else:
                try:
                    stats_raw[name] = float(val)
                except (ValueError, TypeError):
                    pass
        result[home_away] = stats_raw

    return result   # {"home": {...}, "away": {...}}


def _merge_summary_stats(game: dict, summary: dict):
    """Merge full box score stats into game dict."""
    if not summary:
        return
    game["home_stats"].update(summary.get("home", {}))
    game["away_stats"].update(summary.get("away", {}))
    game["has_summary"] = True


# ── Elo power ratings (NET rank proxy) ───────────────────────────────────────

class _SimpleElo:
    """Minimal Elo system for computing power rankings from season game results."""
    K = 20
    HOME_ADV = 65
    DEFAULT = 1500.0

    def __init__(self):
        self.ratings: dict[str, float] = {}

    def get(self, tid: str) -> float:
        return self.ratings.get(tid, self.DEFAULT)

    def update(self, home_id: str, away_id: str, home_won: bool,
               home_score: int, away_score: int):
        h = self.get(home_id)
        a = self.get(away_id)
        exp_h = 1 / (1 + 10 ** ((a - (h + self.HOME_ADV)) / 400))
        margin = abs(home_score - away_score)
        mov = (margin / 20) ** 0.7   # margin-of-victory multiplier
        k = self.K * mov
        if home_won:
            self.ratings[home_id] = h + k * (1 - exp_h)
            self.ratings[away_id] = a + k * (0 - (1 - exp_h))
        else:
            self.ratings[home_id] = h + k * (0 - exp_h)
            self.ratings[away_id] = a + k * (1 - (1 - exp_h))


def _build_elo_rankings(games: list[dict]) -> dict[str, int]:
    """
    Run Elo through all season games, return {team_id: rank} (1 = best).
    Used as proxy for diff_massey_avg_rank.
    """
    elo = _SimpleElo()
    for g in sorted(games, key=lambda x: x["date"]):
        elo.update(g["home_id"], g["away_id"], g["home_won"],
                   g["home_score"], g["away_score"])

    # Rank teams: higher Elo = better = lower rank number
    sorted_teams = sorted(elo.ratings.items(), key=lambda x: -x[1])
    return {tid: rank + 1 for rank, (tid, _) in enumerate(sorted_teams)}


# ── Rolling stats per team ────────────────────────────────────────────────────

def _parse_box(st: dict, opp_st: dict) -> dict:
    """
    Compute per-game efficiency stats from box score dict.
    Handles both scoreboard (partial) and summary (full) stat formats.
    """
    # FG / 3P / FT
    fgm  = st.get("fieldGoalsMade") or st.get("fieldGoalsMade-fieldGoalsAttemptedMade") or 0
    fga  = st.get("fieldGoalsAttempted") or st.get("fieldGoalsMade-fieldGoalsAttemptedAttempted") or 1
    fg3m = st.get("threePointFieldGoalsMade") or st.get("threePointFieldGoalsMade-threePointFieldGoalsAttemptedMade") or 0
    fg3a = st.get("threePointFieldGoalsAttempted") or st.get("threePointFieldGoalsMade-threePointFieldGoalsAttemptedAttempted") or 1
    ftm  = st.get("freeThrowsMade") or st.get("freeThrowsMade-freeThrowsAttemptedMade") or 0
    fta  = st.get("freeThrowsAttempted") or st.get("freeThrowsMade-freeThrowsAttemptedAttempted") or 1

    # Board/misc — only in summary
    orb  = st.get("offensiveRebounds") or 0
    drb  = st.get("defensiveRebounds") or 0
    trb  = st.get("totalRebounds") or st.get("rebounds") or (orb + drb)
    to_  = st.get("totalTurnovers") or st.get("turnovers") or 0
    ast  = st.get("assists") or 0
    blk  = st.get("blocks") or 0
    stl  = st.get("steals") or 0

    # Opponent stats (for def eFG, opp TO rate)
    opp_fgm  = opp_st.get("fieldGoalsMade") or 0
    opp_fg3m = opp_st.get("threePointFieldGoalsMade") or 0
    opp_fga  = opp_st.get("fieldGoalsAttempted") or 1
    opp_fta  = opp_st.get("freeThrowsAttempted") or 1
    opp_orb  = opp_st.get("offensiveRebounds") or 0
    opp_to   = opp_st.get("totalTurnovers") or opp_st.get("turnovers") or 0

    # Possessions (Dean Oliver)
    poss     = max(fga + 0.44 * fta - orb + to_, 1)
    opp_poss = max(opp_fga + 0.44 * opp_fta - opp_orb + opp_to, 1)

    efg      = (fgm + 0.5 * fg3m) / max(fga, 1)
    opp_efg  = (opp_fgm + 0.5 * opp_fg3m) / max(opp_fga, 1)
    off_eff  = 100 * (ftm + 2 * (fgm - fg3m) + 3 * fg3m) / poss
    def_eff  = 100 * (opp_st.get("points") or 0) / opp_poss if opp_st.get("points") else None
    pace     = (poss + opp_poss) / 2

    has_full = bool(orb or drb or to_)   # summary data present?

    return {
        "efg_pct":     efg,
        "opp_efg_pct": opp_efg,
        "to_rate":     to_ / poss    if has_full else None,
        "opp_to_rate": opp_to / opp_poss if has_full else None,
        "orb_rate":    orb / max(orb + drb, 1)     if has_full else None,
        "drb_rate":    drb / max(orb + drb, 1)     if has_full else None,
        "ft_rate":     fta / max(fga, 1),
        "opp_ft_rate": opp_fta / max(opp_fga, 1),
        "fg3_rate":    fg3a / max(fga, 1),
        "fg3_pct":     fg3m / max(fg3a, 1),
        "ast_to_ratio":ast / max(to_, 1) if has_full else None,
        "blk_rate":    blk / max(opp_fga, 1) if has_full else None,
        "stl_rate":    stl / max(opp_poss, 1) if has_full else None,
        "off_eff":     off_eff,
        "def_eff":     def_eff,
        "net_eff":     (off_eff - def_eff) if def_eff else None,
        "pace":        pace,
    }


def _compute_team_stats(games: list[dict], team_id: str, as_of_date: date) -> dict:
    """Compute rolling season stats for a team from completed game list."""
    team_games = [
        g for g in games
        if (g["home_id"] == team_id or g["away_id"] == team_id)
        and date.fromisoformat(g["date"]) < as_of_date
    ]
    team_games.sort(key=lambda g: g["date"])

    n = len(team_games)
    if n == 0:
        return {}

    wins, margins = [], []
    box_stats = {k: [] for k in ["efg_pct","opp_efg_pct","to_rate","opp_to_rate",
                                   "orb_rate","drb_rate","ft_rate","opp_ft_rate",
                                   "fg3_rate","fg3_pct","ast_to_ratio","blk_rate",
                                   "stl_rate","off_eff","def_eff","net_eff","pace"]}
    dates = []

    for g in team_games:
        is_home  = g["home_id"] == team_id
        pts_for  = g["home_score"] if is_home else g["away_score"]
        pts_opp  = g["away_score"] if is_home else g["home_score"]
        won      = g["home_won"]   if is_home else not g["home_won"]

        wins.append(1 if won else 0)
        margins.append(pts_for - pts_opp)
        dates.append(date.fromisoformat(g["date"]))

        h_st = g.get("home_stats", {})
        a_st = g.get("away_stats", {})
        my_st  = h_st if is_home else a_st
        opp_st = a_st if is_home else h_st
        # inject points for def_eff calculation
        if "points" not in opp_st:
            opp_st = dict(opp_st)
            opp_st["points"] = pts_opp

        bx = _parse_box(my_st, opp_st)
        for k in box_stats:
            v = bx.get(k)
            if v is not None and not (k == "ast_to_ratio" and v > 20):
                box_stats[k].append(v)

    w = ROLLING_WINDOW

    def tail_mean(lst, n=w):
        sl = [x for x in lst[-n:] if x is not None]
        return sum(sl) / len(sl) if sl else None

    # Win streak
    streak = 0
    for w_val in reversed(wins):
        expected = wins[-1]
        if w_val == expected:
            streak += 1 if expected == 1 else -1
        else:
            break

    rest = (as_of_date - dates[-1]).days if dates else 7

    stats = {
        "n_games":      n,
        "win_pct":      sum(wins) / n,
        "avg_margin":   sum(margins) / n,
        "rest_days":    min(rest, 30),
        "games_last_7": sum(1 for d in dates if (as_of_date - d).days <= 7),
        "win_streak":   streak,
        "sos":          sum(-m for m in margins[-w:]) / min(len(margins), w),
    }
    for k in box_stats:
        stats[k] = tail_mean(box_stats[k])

    return stats


def _compute_h2h(games: list[dict], home_id: str, away_id: str,
                 as_of_date: date, n: int = 10) -> dict:
    h2h = [
        g for g in games
        if set([g["home_id"], g["away_id"]]) == set([home_id, away_id])
        and date.fromisoformat(g["date"]) < as_of_date
    ]
    h2h.sort(key=lambda g: g["date"])
    recent = h2h[-n:]
    if not recent:
        return {"h2h_win_pct_5": 0.5, "h2h_win_pct_10": 0.5, "h2h_games": 0}

    def home_wins(g):
        return (g["home_id"] == home_id and g["home_won"]) or \
               (g["away_id"] == home_id and not g["home_won"])

    last5 = recent[-5:]
    return {
        "h2h_win_pct_5":  sum(1 for g in last5 if home_wins(g)) / len(last5),
        "h2h_win_pct_10": sum(1 for g in recent if home_wins(g)) / len(recent),
        "h2h_games":      len(recent),
    }


def _safe_diff(h: dict, a: dict, key: str) -> float:
    hv = h.get(key)
    av = a.get(key)
    if hv is None and av is None:
        return _lg(f"diff_{key}", 0.0)
    hv = hv if hv is not None else _lg(key, 0.0)
    av = av if av is not None else _lg(key, 0.0)
    return hv - av


# ── SeasonStatsStore ──────────────────────────────────────────────────────────

class SeasonStatsStore:
    """
    Maintains current-season rolling stats for all D1 teams.
    Backed by ESPN public API + local JSON cache.
    """

    def __init__(self, season: int = CURRENT_SEASON):
        self.season      = season
        self.games: list[dict] = []
        self.elo_ranks: dict[str, int] = {}   # team_id → rank (1=best), NET proxy
        self._name_map: dict[str, str] = {}   # lower_name → team_id
        _load_league_averages()

    def refresh(self, force: bool = False) -> "SeasonStatsStore":
        self._load_or_fetch_games(force)
        self._enrich_with_summaries()
        self._build_elo_rankings()
        self._build_name_map()
        print(f"[SeasonStatsStore] {len(self.games):,} games | "
              f"{len(self.elo_ranks)} teams ranked | "
              f"{len(self._name_map)} name aliases")
        return self

    def get_matchup_features(self, home_name: str, away_name: str,
                              game_date: Optional[date] = None) -> dict:
        if game_date is None:
            game_date = date.today()

        home_id = self._find_id(home_name)
        away_id = self._find_id(away_name)

        h_stats = _compute_team_stats(self.games, home_id, game_date) if home_id else {}
        a_stats = _compute_team_stats(self.games, away_id, game_date) if away_id else {}
        h2h     = _compute_h2h(self.games, home_id, away_id, game_date) if (home_id and away_id) else {}

        # Elo-based power rank (proxy for NET/Massey) — lower rank = better
        h_rank  = self.elo_ranks.get(home_id, 200)
        a_rank  = self.elo_ranks.get(away_id, 200)
        # Flip sign: positive diff means home team is better ranked
        rank_diff = float(a_rank - h_rank)

        features = {
            # Tier 1
            "diff_win_pct":        _safe_diff(h_stats, a_stats, "win_pct"),
            "diff_avg_margin":     _safe_diff(h_stats, a_stats, "avg_margin"),
            "diff_sos":            _safe_diff(h_stats, a_stats, "sos"),
            "diff_rest_days":      _safe_diff(h_stats, a_stats, "rest_days"),
            "diff_games_last_7":   _safe_diff(h_stats, a_stats, "games_last_7"),
            "diff_win_streak":     _safe_diff(h_stats, a_stats, "win_streak"),
            "diff_h2h_win_pct_5":  h2h.get("h2h_win_pct_5",  0.5) - 0.5,
            "diff_h2h_win_pct_10": h2h.get("h2h_win_pct_10", 0.5) - 0.5,
            "diff_conf_win_pct":   0.0,
            # Tier 2
            "diff_efg_pct":     _safe_diff(h_stats, a_stats, "efg_pct"),
            "diff_opp_efg_pct": _safe_diff(h_stats, a_stats, "opp_efg_pct"),
            "diff_to_rate":     _safe_diff(h_stats, a_stats, "to_rate"),
            "diff_opp_to_rate": _safe_diff(h_stats, a_stats, "opp_to_rate"),
            "diff_orb_rate":    _safe_diff(h_stats, a_stats, "orb_rate"),
            "diff_drb_rate":    _safe_diff(h_stats, a_stats, "drb_rate"),
            "diff_ft_rate":     _safe_diff(h_stats, a_stats, "ft_rate"),
            "diff_opp_ft_rate": _safe_diff(h_stats, a_stats, "opp_ft_rate"),
            "diff_fg3_rate":    _safe_diff(h_stats, a_stats, "fg3_rate"),
            "diff_fg3_pct":     _safe_diff(h_stats, a_stats, "fg3_pct"),
            "diff_ast_to_ratio":_safe_diff(h_stats, a_stats, "ast_to_ratio"),
            "diff_blk_rate":    _safe_diff(h_stats, a_stats, "blk_rate"),
            "diff_stl_rate":    _safe_diff(h_stats, a_stats, "stl_rate"),
            "diff_off_eff":     _safe_diff(h_stats, a_stats, "off_eff"),
            "diff_def_eff":     _safe_diff(h_stats, a_stats, "def_eff"),
            "diff_net_eff":     _safe_diff(h_stats, a_stats, "net_eff"),
            "diff_pace":        _safe_diff(h_stats, a_stats, "pace"),
            # Tier 3 — Elo rank as NET proxy
            "diff_massey_avg_rank":  rank_diff,
            "diff_massey_best_rank": rank_diff,
            "diff_massey_n_systems": 0.0,
            "diff_massey_spread":    0.0,
            # Metadata
            "home_n_games": h_stats.get("n_games", 0),
            "away_n_games": a_stats.get("n_games", 0),
            "home_id":      home_id,
            "away_id":      away_id,
        }
        return features

    # ── Internal ─────────────────────────────────────────────────────────────

    def _find_id(self, name: str) -> Optional[str]:
        key = name.lower().strip()
        if key in self._name_map:
            return self._name_map[key]
        for k, tid in self._name_map.items():
            if key in k or k in key:
                return tid
        return None

    def _build_name_map(self):
        for g in self.games:
            for fid, fname in [("home_id", "home_name"), ("away_id", "away_name")]:
                tid  = g.get(fid)
                name = g.get(fname, "")
                if tid and name:
                    self._name_map[name.lower().strip()] = tid

    def _build_elo_rankings(self):
        self.elo_ranks = _build_elo_rankings(self.games)

    def _games_cache_path(self) -> Path:
        return CACHE_DIR / f"season_games_{self.season}.json"

    def _load_or_fetch_games(self, force: bool):
        cache_path = self._games_cache_path()
        today = date.today()

        if cache_path.exists() and not force:
            cached = json.loads(cache_path.read_text())
            self.games = cached.get("games", [])
            last_date_str = cached.get("last_date")
            if last_date_str:
                last_date = date.fromisoformat(last_date_str)
                if last_date >= today - timedelta(days=1):
                    return
                start = last_date + timedelta(days=1)
            else:
                start = SEASON_START
        else:
            self.games = []
            start = SEASON_START

        end = today - timedelta(days=1)
        current = start
        fetched = 0
        while current <= end:
            day_games = _fetch_scoreboard_day(current)
            self.games.extend(day_games)
            fetched += len(day_games)
            current += timedelta(days=1)
            if day_games:
                time.sleep(REQUEST_DELAY)

        if fetched > 0 or not cache_path.exists():
            cache_path.write_text(json.dumps({
                "games":     self.games,
                "last_date": today.isoformat(),
                "season":    self.season,
            }))

    def _enrich_with_summaries(self):
        """
        Fetch full box-score summaries for the most recent games that don't have them.
        We only fetch the last SUMMARY_LOOKBACK games per team to bound API calls.
        """
        # Find game IDs that need summaries (don't have full box score stats)
        needs_summary = set()
        # Get team's most recent game IDs
        team_game_dates: dict[str, list] = {}
        for g in self.games:
            for tid in [g["home_id"], g["away_id"]]:
                team_game_dates.setdefault(tid, []).append((g["date"], g["game_id"]))

        for tid, date_ids in team_game_dates.items():
            date_ids.sort(reverse=True)
            for _, gid in date_ids[:SUMMARY_LOOKBACK]:
                needs_summary.add(gid)

        # Find which of these already have summary data
        game_by_id = {g["game_id"]: g for g in self.games}
        # Only fetch games that don't yet have summary data, most recent first
        all_need = [
            gid for gid in needs_summary
            if gid in game_by_id and not game_by_id[gid].get("has_summary")
        ]
        # Sort by date descending so we fetch the most recent games first
        all_need.sort(key=lambda gid: game_by_id[gid]["date"], reverse=True)
        to_fetch = all_need[:SUMMARY_MAX_FETCH]

        if not to_fetch:
            return

        print(f"[SeasonStatsStore] Fetching {len(to_fetch)} game summaries "
              f"({len(all_need)} needed, capped at {SUMMARY_MAX_FETCH})...", file=sys.stderr)
        fetched = 0
        for gid in to_fetch:
            summary = _fetch_game_summary(gid)
            if summary:
                _merge_summary_stats(game_by_id[gid], summary)
                fetched += 1
            time.sleep(REQUEST_DELAY)

        print(f"[SeasonStatsStore] Got {fetched}/{len(to_fetch)} summaries", file=sys.stderr)

        # Persist enriched cache
        cache_path = self._games_cache_path()
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            cached["games"] = self.games
            cache_path.write_text(json.dumps(cached))
