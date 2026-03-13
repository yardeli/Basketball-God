"""
run_dashboard.py — One-command dashboard launcher for Basketball-God.

  1. Computes yesterday's P&L using saved bets log (real market odds) or falls
     back to predictions file if no log exists.
  2. Saves results to outputs/paper_bets_YYYYMMDD.json
     (the web server's /api/paperbets endpoint picks this up automatically)
  3. Kills any old server on port 5050 and starts a fresh one
  4. Saves today's Strong Bet signals with real market odds to
     outputs/bets_log_YYYYMMDD.json (after server is up)
  5. Opens http://localhost:5050 in your browser

Usage:
    python run_dashboard.py
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT        = Path(__file__).parent
OUTPUTS_DIR = ROOT / "outputs"
SERVER_PORT = 5050

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)

STAKE          = 50     # $ per simulated bet
CONF_THRESHOLD = 0.65   # minimum confidence to qualify as "Strong Bet"


# ── Payout helpers ─────────────────────────────────────────────────────────────

def market_payout(odds_raw: int, stake: float) -> float:
    """Return profit (not including stake) for a winning bet at American odds."""
    if odds_raw > 0:
        return round(stake * (odds_raw / 100), 2)
    else:
        return round(stake * (100 / abs(odds_raw)), 2)


def _prob_to_american(p: float) -> int:
    """Convert win probability to implied American odds (integer)."""
    if p >= 0.5:
        return -round(p / (1 - p) * 100)
    return round((1 - p) / p * 100)


def _american_str(odds_raw: int) -> str:
    return f"+{odds_raw}" if odds_raw > 0 else str(odds_raw)


# ── ESPN score fetcher ─────────────────────────────────────────────────────────

def fetch_scores(date_str: str) -> dict:
    """Return dict keyed by ESPN game_id with final scores."""
    url = f"{ESPN_SCOREBOARD}?dates={date_str}&groups=50&limit=200"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"  [!] ESPN fetch failed: {e}")
        return {}

    results = {}
    for event in data.get("events", []):
        gid         = event.get("id", "")
        competition = event.get("competitions", [{}])[0]
        status_type = competition.get("status", {}).get("type", {})
        if status_type.get("name") != "STATUS_FINAL":
            continue

        competitors = competition.get("competitors", [])
        if len(competitors) != 2:
            continue

        home = next((c for c in competitors if c["homeAway"] == "home"), None)
        away = next((c for c in competitors if c["homeAway"] == "away"), None)
        if not home or not away:
            continue

        home_score = int(home.get("score", 0) or 0)
        away_score = int(away.get("score", 0) or 0)
        home_name  = home.get("team", {}).get("displayName", "")
        away_name  = away.get("team", {}).get("displayName", "")
        winner     = home_name if home_score > away_score else away_name

        results[gid] = {
            "home_team": home_name, "away_team": away_name,
            "home_score": home_score, "away_score": away_score,
            "winner": winner,
        }
    return results


# ── P&L computation ────────────────────────────────────────────────────────────

def _names_match(a: str, b: str) -> bool:
    """Loose team-name comparison."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    words_a = a.split()[:2]
    return any(w in b for w in words_a)


def _build_team_index(scores: dict) -> dict:
    """Build a name→score_entry index for fallback matching."""
    idx = {}
    for gid, sc in scores.items():
        for role in ("home_team", "away_team"):
            name = sc[role].lower()
            idx[name] = sc
            words = name.split()
            if words:
                idx[words[0]] = sc
            if len(words) >= 2:
                idx[" ".join(words[:2])] = sc
    return idx


def _find_score(gid: str, bet: dict, scores: dict, team_idx: dict) -> dict | None:
    """Return score entry by game_id, then fall back to team-name matching."""
    if gid in scores:
        return scores[gid]
    for field in ("pick_team", "home_team", "away_team"):
        name = bet.get(field, "").lower().strip()
        if not name:
            continue
        if name in team_idx:
            return team_idx[name]
        word = name.split()[0] if name.split() else ""
        if word and word in team_idx:
            return team_idx[word]
    return None


def compute_prev_day_from_log(log: dict, scores: dict) -> dict | None:
    """
    Compute P&L from a bets_log file that has real market odds.
    Returns a summary dict or None if no settled trades.
    """
    bets     = log.get("bets", [])
    stake    = log.get("stake", STAKE)
    team_idx = _build_team_index(scores)

    trades, no_result = [], []

    for bet in bets:
        gid       = bet.get("game_id", "")
        matchup   = bet.get("matchup", "")
        bet_type  = bet.get("bet_type", "Moneyline")
        desc      = bet.get("description", "")
        odds_raw  = bet.get("market_odds_raw")
        odds_str  = bet.get("market_odds_str", "N/A")
        signal    = bet.get("signal", "")
        pick_team = bet.get("pick_team", "")
        line      = bet.get("line")

        sc = _find_score(gid, bet, scores, team_idx)
        if sc is None:
            no_result.append({
                "matchup": matchup, "bet_type": bet_type,
                "description": desc, "odds": odds_str,
                "stake": stake, "signal": signal,
            })
            continue

        hs    = sc["home_score"]
        as_   = sc["away_score"]
        total = hs + as_

        won = False
        if bet_type == "Moneyline":
            won = _names_match(sc["winner"], pick_team)

        elif bet_type == "Spread":
            if line is None:
                no_result.append({
                    "matchup": matchup, "bet_type": bet_type,
                    "description": desc, "odds": odds_str,
                    "stake": stake, "signal": signal,
                })
                continue
            is_home = _names_match(pick_team, sc["home_team"])
            margin  = (hs - as_) if is_home else (as_ - hs)
            won     = margin > line

        elif bet_type in ("Total Over", "Total Under"):
            if line is None:
                no_result.append({
                    "matchup": matchup, "bet_type": bet_type,
                    "description": desc, "odds": odds_str,
                    "stake": stake, "signal": signal,
                })
                continue
            won = (total > line) if bet_type == "Total Over" else (total < line)

        to_win = market_payout(odds_raw, stake) if odds_raw is not None else 0
        pnl    = to_win if won else -stake

        trades.append({
            "matchup":     matchup,
            "bet_type":    bet_type,
            "description": desc,
            "pick":        pick_team,
            "signal":      signal,
            "odds":        odds_str,
            "stake":       stake,
            "to_win":      to_win,
            "score":       f"{hs}-{as_}",
            "won":         won,
            "pnl":         pnl,
        })

    if not trades:
        return None

    wins      = [t for t in trades if t["won"]]
    losses    = [t for t in trades if not t["won"]]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate  = len(wins) / len(trades) * 100
    roi       = total_pnl / (len(trades) * stake) * 100
    best      = max(trades, key=lambda t: t["pnl"])
    worst     = min(trades, key=lambda t: t["pnl"])

    return {
        "date":          log.get("date", ""),
        "stake_per_bet": stake,
        "threshold":     CONF_THRESHOLD,
        "total_bets":    len(trades),
        "pending":       len(no_result),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(win_rate, 1),
        "total_pnl":     round(total_pnl, 2),
        "roi_pct":       round(roi, 1),
        "best_trade":    {"pick": best["description"], "pnl": best["pnl"]},
        "worst_trade":   {"pick": worst["description"], "pnl": worst["pnl"]},
        "trades":        trades,
        "pending_trades": no_result,
        "source":        "bets_log",
    }


def compute_prev_day_from_predictions(target: date, scores: dict) -> dict | None:
    """
    Fallback: compute P&L from predictions_YYYYMMDD.json using model-derived odds.
    """
    date_str  = target.strftime("%Y%m%d")
    pred_path = OUTPUTS_DIR / f"predictions_{date_str}.json"
    if not pred_path.exists():
        return None

    preds  = json.loads(pred_path.read_text())
    strong = [
        p for p in preds
        if max(p.get("home_win_prob", 0), p.get("away_win_prob", 0)) >= CONF_THRESHOLD
    ]
    if not strong:
        return None

    team_idx = _build_team_index(scores)
    trades, no_result = [], []

    for p in strong:
        gid       = p["game_id"]
        conf      = max(p["home_win_prob"], p["away_win_prob"])
        pick      = p["predicted_winner"]
        matchup   = f"{p['away_team']} @ {p['home_team']}"
        odds_raw  = _prob_to_american(conf)
        odds_str  = _american_str(odds_raw)
        to_win    = market_payout(odds_raw, STAKE)

        fake_bet = {"game_id": gid, "home_team": p["home_team"], "away_team": p["away_team"], "pick_team": pick}
        sc = _find_score(gid, fake_bet, scores, team_idx)
        if sc:
            hs, as_ = sc["home_score"], sc["away_score"]
            won = _names_match(sc["winner"], pick)
            pnl = to_win if won else -STAKE
            trades.append({
                "matchup": matchup, "bet_type": "Moneyline",
                "description": f"{pick} to win",
                "pick": pick, "signal": "STRONG BET",
                "odds": odds_str, "stake": STAKE, "to_win": to_win,
                "score": f"{hs}-{as_}", "won": won, "pnl": pnl,
            })
        else:
            no_result.append({
                "matchup": matchup, "bet_type": "Moneyline",
                "description": f"{pick} to win",
                "odds": odds_str, "stake": STAKE, "to_win": to_win,
            })

    if not trades:
        return None

    wins      = [t for t in trades if t["won"]]
    losses    = [t for t in trades if not t["won"]]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate  = len(wins) / len(trades) * 100
    roi       = total_pnl / (len(trades) * STAKE) * 100
    best      = max(trades, key=lambda t: t["pnl"])
    worst     = min(trades, key=lambda t: t["pnl"])

    return {
        "date":          str(target),
        "stake_per_bet": STAKE,
        "threshold":     CONF_THRESHOLD,
        "total_bets":    len(trades),
        "pending":       len(no_result),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(win_rate, 1),
        "total_pnl":     round(total_pnl, 2),
        "roi_pct":       round(roi, 1),
        "best_trade":    {"pick": best["pick"], "pnl": best["pnl"]},
        "worst_trade":   {"pick": worst["pick"], "pnl": worst["pnl"]},
        "trades":        trades,
        "pending_trades": no_result,
        "source":        "predictions_fallback",
    }


def compute_prev_day(target: date):
    date_str  = target.strftime("%Y%m%d")
    log_path  = OUTPUTS_DIR / f"bets_log_{date_str}.json"
    pred_path = OUTPUTS_DIR / f"predictions_{date_str}.json"

    if not log_path.exists() and not pred_path.exists():
        print(f"  [!] No bets log or predictions found for {target} — skipping P&L.")
        return None

    print(f"  Fetching ESPN scores for {target}...", end=" ", flush=True)
    scores = fetch_scores(date_str)
    print(f"found {len(scores)} completed games.")

    summary = None

    if log_path.exists():
        print(f"  Using real market odds from bets_log_{date_str}.json")
        log     = json.loads(log_path.read_text())
        summary = compute_prev_day_from_log(log, scores)

    if summary is None and pred_path.exists():
        print(f"  Falling back to model-derived odds from predictions_{date_str}.json")
        summary = compute_prev_day_from_predictions(target, scores)

    if summary is None:
        print("  No completed Strong Bet games found in ESPN data yet.")
        return None

    wins      = summary["wins"]
    losses    = summary["losses"]
    total_pnl = summary["total_pnl"]
    roi       = summary["roi_pct"]
    source    = summary.get("source", "")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"paper_bets_{date_str}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  P&L saved -> outputs/paper_bets_{date_str}.json")
    src_note = " (model odds — no bets log)" if source == "predictions_fallback" else " (real market odds)"
    print(f"  ({wins}W / {losses}L  |  ${total_pnl:+,.2f}  |  ROI {roi:+.1f}%){src_note}")

    return summary


# ── Bet logger ─────────────────────────────────────────────────────────────────

def save_bets_log(target: date, stake: float = STAKE):
    """
    Fetch today's Strong Bet signals from /api/daily and save them with
    real market odds to outputs/bets_log_YYYYMMDD.json.
    """
    date_str = target.strftime("%Y%m%d")
    out_path = OUTPUTS_DIR / f"bets_log_{date_str}.json"

    url = f"http://localhost:{SERVER_PORT}/api/daily"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            daily = json.loads(r.read())
    except Exception as e:
        print(f"  [!] Could not fetch /api/daily: {e}")
        return

    games = daily.get("games", [])
    if not games:
        print(f"  [!] /api/daily returned no games — bets log not saved.")
        return

    bets = []
    for game in games:
        home     = game.get("home_display") or game.get("home_team", "")
        away     = game.get("away_display") or game.get("away_team", "")
        matchup  = f"{away} @ {home}"
        gid      = game.get("id", "")
        commence = game.get("commence_time", "")

        for bet in game.get("bets", []):
            signal = bet.get("signal", "")
            if signal not in ("STRONG BET", "MODEL HIGHER"):
                continue

            bet_type   = bet.get("bet_type", "Moneyline")
            desc       = bet.get("description", "")
            edge       = bet.get("edge", 0)
            model_prob = bet.get("model_prob", 0)
            odds_raw   = bet.get("market_odds_raw")
            odds_str   = bet.get("market_odds", "N/A")
            best_book  = bet.get("best_book", "")
            pick_team  = bet.get("team", "")
            line       = bet.get("line")

            if not pick_team and "to win" in desc.lower():
                pick_team = desc.replace(" to win", "").strip()

            bets.append({
                "game_id":         gid,
                "matchup":         matchup,
                "home_team":       game.get("home_team", ""),
                "away_team":       game.get("away_team", ""),
                "commence_time":   commence,
                "bet_type":        bet_type,
                "description":     desc,
                "pick_team":       pick_team,
                "model_prob":      model_prob,
                "market_odds_raw": odds_raw,
                "market_odds_str": odds_str,
                "edge":            edge,
                "signal":          signal,
                "best_book":       best_book,
                "line":            line,
            })

    log = {
        "date":         str(target),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stake":        stake,
        "n_bets":       len(bets),
        "bets":         bets,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, indent=2))

    if bets:
        strong   = sum(1 for b in bets if b["signal"] == "STRONG BET")
        value    = sum(1 for b in bets if b["signal"] == "MODEL HIGHER")
        types    = {}
        for b in bets:
            types[b["bet_type"]] = types.get(b["bet_type"], 0) + 1
        type_str = ", ".join(f"{v} {k}" for k, v in sorted(types.items()))
        print(f"  Bets log saved -> outputs/bets_log_{date_str}.json")
        print(f"  ({strong} Strong Bets, {value} Value Bets | {type_str})")
        print(f"  Stake: ${stake:.0f}/bet")
    else:
        print(f"  No Strong/Value Bet signals found today — empty log saved.")


# ── Server management ──────────────────────────────────────────────────────────

def server_running():
    """Return True if something is already listening on SERVER_PORT."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", SERVER_PORT)) == 0


def kill_old_server():
    """Kill all processes currently holding SERVER_PORT."""
    if sys.platform == "win32":
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True
        )
        killed = []
        for line in result.stdout.splitlines():
            if f":{SERVER_PORT}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                if pid not in killed:
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True)
                    killed.append(pid)
        if killed:
            time.sleep(0.5)
    else:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{SERVER_PORT}"],
            capture_output=True, text=True
        )
        for pid in result.stdout.strip().splitlines():
            subprocess.run(["kill", "-9", pid], capture_output=True)
        time.sleep(0.5)


def start_server():
    """Launch web/server.py in a background process."""
    server_script = ROOT / "web" / "server.py"
    if not server_script.exists():
        print(f"  [!] web/server.py not found — cannot start server.")
        return None

    if sys.platform == "win32":
        proc = subprocess.Popen(
            [sys.executable, str(server_script)],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        proc = subprocess.Popen(
            [sys.executable, str(server_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    for _ in range(12):
        time.sleep(0.5)
        if server_running():
            return proc
    return proc


def open_browser():
    url = f"http://localhost:{SERVER_PORT}"
    if sys.platform == "win32":
        os.startfile(url)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", url])
    else:
        subprocess.Popen(["xdg-open", url])
    print(f"  Opening -> {url}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    today     = date.today()
    yesterday = today - timedelta(days=1)

    print(f"\n{'='*60}")
    print(f"  Basketball-God Dashboard Launcher")
    print(f"  Date: {today.strftime('%B %d, %Y')}")
    print(f"{'='*60}\n")

    # Step 1: Compute yesterday's P&L
    print(f"[1/4] Computing P&L for {yesterday.strftime('%B %d, %Y')}...")
    compute_prev_day(yesterday)

    # Step 2: (Re)start the web server
    print(f"\n[2/4] Starting web server on port {SERVER_PORT}...")
    if server_running():
        print(f"  Killing old server...")
        kill_old_server()
        time.sleep(0.5)
    start_server()
    if server_running():
        print(f"  Server started.")
    else:
        print(f"  [!] Server may still be starting — continuing anyway.")

    # Step 3: Save today's bets log with real market odds
    print(f"\n[3/4] Logging today's Strong Bet signals...")
    save_bets_log(today, stake=STAKE)

    # Step 4: Open the browser
    print(f"\n[4/4] Opening dashboard...")
    open_browser()

    print("\n  Done.\n")
