# Basketball-God 2.0 — Troubleshooting & FAQ

> Keep this doc handy. Every issue listed here is something that has actually happened.
> Issues are grouped by symptom so you can find the fix fast.

---

## Table of Contents

1. [Dashboard Won't Load (Not Found / Blank Page)](#1-dashboard-wont-load)
2. [No Games Showing ("No games found for today")](#2-no-games-showing)
3. [Games Showing But Predictions Are All 50/50](#3-predictions-all-5050)
4. [Port Already In Use / Multiple Server Processes](#4-port-conflicts)
5. [Model File Not Found / Model Won't Load](#5-model-wont-load)
6. [A/B Backtest or Training Script Crashes](#6-backtest-training-crashes)
7. [Live Games Panel Not Showing](#7-live-panel-not-showing)
8. [Bets / Edges Look Wrong](#8-bets-and-edges-look-wrong)
9. [API Key Errors (Odds API)](#9-api-key-errors)
10. [How To Start Everything From Scratch](#10-fresh-start)

---

## 1. Dashboard Won't Load

### Symptom
Browser shows **"Not Found"** at `localhost:5050` or `localhost:5051`

### Cause A — Wrong URL path
You navigated to something like `localhost:5050/dashboard` or `localhost:5050/index`.
The dashboard lives at the **root**: no path after the port.

**Fix:** Go to exactly `http://localhost:5051` (just the port, nothing after it)

---

### Cause B — Port conflict (multiple old server processes)
The most common cause. Old background server processes are still bound to the port,
and they intercept your request before Flask can respond.

**Fix:**
```
# In PowerShell:
netstat -ano | findstr :5051

# Kill each PID you see listed:
taskkill /F /PID <pid1> /PID <pid2> /PID <pid3>

# Then restart:
python web/server.py
```
If you keep hitting this, bump the port number by 1 in `web/server.py` (last line):
```python
app.run(host="0.0.0.0", port=5052, debug=False)  # change to unused port
```

---

### Cause C — Server crashed on startup
The terminal will show a Python traceback right after you run `python web/server.py`.

**Common crash causes and fixes:**

| Error message | Fix |
|---|---|
| `ModuleNotFoundError: xgboost` | `pip install xgboost` |
| `ModuleNotFoundError: flask` | `pip install flask` |
| `No module named 'season_stats_store'` | Make sure you're running from `C:\Users\yarden\Basketball-God2.0\` |
| `Model not found at phase6_regular_season/output/...` | Run `python phase7_v2/train_v2.py` first |
| Any other crash | Check the full traceback — the last line tells you exactly what failed |

---

## 2. No Games Showing

### Symptom
Dashboard loads fine but shows **"No games found for today"** and `0 game(s)`

### Cause A — Stale or empty odds cache (most common)
The cache file from earlier in the day expired (30-min TTL) and the API refresh failed silently.

**Fix:** Click the **"Refresh Lines"** button on the dashboard. This forces a live API call.

If that still shows 0 games, check the cache directly:
```bash
# In the project folder:
python -c "
import json, time
d = json.load(open('phase5_deploy/output/games_cache_2026-03-14.json'))
print('Games:', d['n_games'])
print('Error:', d.get('error'))
print('Age (min):', round((time.time() - d['fetched_at']) / 60, 1))
"
```

---

### Cause B — API key expired or invalid
The Odds API key in your `.env` file is no longer valid. The dashboard goes blank instead
of showing cached games.

**Fix:**
1. Check `C:\Users\yarden\Basketball-God2.0\.env` — look for `ODDS_API_KEY=...`
2. Go to [the-odds-api.com](https://the-odds-api.com), log in, copy your current key
3. Replace the value in `.env` and restart the server

> Note: As of the last fix (March 2026), the code now falls back to the stale cache
> when the API key is invalid, so you'll still see cached games with a warning banner
> instead of a blank screen.

---

### Cause C — All games already started (in-progress filter)
If it's late in the evening and all of today's games have already tipped off, the model
strips betting edges from in-progress games (pre-game model can't price live odds).
The games still appear but show no bet recommendations.

**This is expected behavior** — not a bug.

---

### Cause D — UTC vs PST date mismatch
Before March 2026: the cache was keyed to UTC midnight, not PST midnight. So at 4pm PST
(midnight UTC) the cache would reset and try to fetch a new day before games were posted.

**Fix:** Already patched. Cache is now keyed to PST midnight. No action needed unless
you've rolled back to an older version.

---

## 3. Predictions All 50/50

### Symptom
Every game shows exactly 50% home / 50% away win probability

### Cause A — Regular season model not loaded
The model pkl file is missing or failed to load. The predictor falls back to 50/50.

**Fix:**
```bash
# Check if the model file exists:
ls phase6_regular_season/output/regular_season_model.pkl

# If missing, retrain:
python phase7_v2/train_v2.py

# The output will be in phase7_v2/output/ — copy it:
cp phase7_v2/output/regular_season_model_v2.pkl phase6_regular_season/output/regular_season_model.pkl
```

---

### Cause B — Teams not found in SeasonStatsStore
The team names from the Odds API don't match any team in the rolling stats database.
The model predicts on league-average features, which produces ~50/50.

**Check:** Look at terminal output when the server starts — it will log
`[SeasonStatsStore] refreshed N teams`. If N is 0 or very low, the store failed to load.

**Fix:**
```bash
python -c "from season_stats_store import SeasonStatsStore; s = SeasonStatsStore(); s.refresh(); print(s.n_teams, 'teams loaded')"
```
If it errors, the database path is wrong or the DB hasn't been built. Run the data pipeline first.

---

### Cause C — Tournament routing during March Madness
March 15 – April 8: games automatically route to the **tournament model** (NCAAModel),
not the regular season model. The tournament model uses Elo + seed features, not rolling stats.
If Elo data is missing, it outputs 50/50.

**This is expected behavior** during the tournament window.

---

## 4. Port Conflicts

### Symptom
`OSError: [Errno 98] Address already in use` when starting the server,
OR browser shows "Not Found" even though `python web/server.py` appears to be running

### Fix — Kill all processes on the port
```powershell
# Check who's using the port:
netstat -ano | findstr :5051

# Kill by PID (repeat for each PID listed):
taskkill /F /PID 12345

# Or use the included batch file:
kill_server.bat
```

### Fix — Use a fresh port
Edit the last line of `web/server.py`:
```python
app.run(host="0.0.0.0", port=5052, debug=False)  # bump port number
```

### Prevention
Always stop the server with **Ctrl+C** in the terminal before closing the window.
Closing the terminal window without Ctrl+C leaves the process running in the background.

---

## 5. Model Won't Load

### Symptom
Server starts but logs: `[RegularSeasonModel] Model not found at .../regular_season_model.pkl`

### Cause
The model pkl file doesn't exist. This happens after a fresh clone or if the file was deleted.

### Fix
```bash
# Option 1 — Train the v2 model (takes ~5 min):
python phase7_v2/train_v2.py

# Then copy output to where the server looks for it:
cp phase7_v2/output/regular_season_model_v2.pkl phase6_regular_season/output/regular_season_model.pkl

# Option 2 — If you have a backup on GitHub:
git checkout phase6_regular_season/output/regular_season_model.pkl
```

---

## 6. Backtest / Training Script Crashes

### Common Error A — `IndexError: index N is out of bounds for axis 0`
**Cause:** The DataFrame still has its original parquet row indices (0–193,865) but the
sample weight array is 0-indexed from 0 to 81,829. Positional indexing fails.

**Fix:** Add `df = df.reset_index(drop=True)` right after loading and filtering the DataFrame,
before computing sample weights.

---

### Common Error B — `ValueError: Invalid format specifier`
**Cause:** Python f-strings don't support inline ternary expressions in format specs.
`f"{value:.3f if condition else 'N/A'}"` is invalid syntax inside `{}`.

**Fix:**
```python
# Wrong:
print(f"ats={ats_acc:.3f if ats_acc else 'N/A'}")

# Right:
ats_str = f"{ats_acc:.3f}" if ats_acc is not None else "N/A"
print(f"ats={ats_str}")
```

---

### Common Error C — Elo computation takes forever
**Normal behavior.** Elo must be computed sequentially (game by game) from 200K+ games.
This takes 2–5 minutes. The script will print progress — just wait.

---

### Common Error D — `backtest_v2.json not found` when running `backtest_ab.py`
**Cause:** You ran `backtest_ab.py` before running `train_v2.py`.

**Fix:** Always run in order:
```bash
python phase7_v2/train_v2.py      # step 1 — creates backtest_v2.json
python phase7_v2/backtest_ab.py   # step 2 — reads backtest_v2.json
```

---

## 7. Live Panel Not Showing

### Symptom
The "LIVE" badge in the top-right is orange but the live games panel is hidden

### Cause A — No games currently in progress
The live panel only appears when ESPN reports at least one game with a live clock.
If all games are pre-game or final, the panel stays hidden. This is correct.

### Cause B — ESPN API returned an error
The live scores come from ESPN's unofficial API. It can go down or rate-limit.

**Fix:** The panel auto-refreshes every 60 seconds. Wait a minute and it should recover.

### Cause C — Browser cached old JS
Hard-refresh the page: **Ctrl+Shift+R** (Windows) to force reload without cache.

---

## 8. Bets and Edges Look Wrong

### Symptom A — All bets show "AGREE" or "FADE", nothing is "STRONG VALUE"
**Cause:** The model's predicted probabilities are too close to the market's implied odds.
This can happen when:
- The team has fewer than 3 games in the rolling stats (sparse data → near-50/50 prediction)
- It's early in the season (November/December) and stats are thin

**Check:** Look at `home_n_games` and `away_n_games` in the API response.
If either is below 3, the model defaults to league-average features.

---

### Symptom B — Edge is huge (>40%) on every game
**Cause:** The model is outputting near-0 or near-1 probabilities, likely due to
extreme feature values from bad data.

**Fix:** Check `diff_massey_avg_rank` and `diff_elo` values. If either is in the
thousands, the feature pipeline has a join issue and is producing garbage values.
Re-run `python phase7_v2/train_v2.py` to rebuild clean features.

---

### Symptom C — Spread bets show wrong direction
**Cause:** The `model_spread` sign convention: **positive = home favored, negative = away favored**.
The market spread sign is the opposite on some books.

The code uses: `adj_spread = model_spread if is_home else -model_spread`
and then: `cover_prob(adj_spread, -market_line)` — the double-negative is intentional.

This is correct. If it still looks wrong, check that the book is reporting the spread
from the home team's perspective (DraftKings and FanDuel do; some smaller books don't).

---

## 9. API Key Errors

### The Odds API (game lines, spreads, totals)
- **Where the key lives:** `C:\Users\yarden\Basketball-God2.0\.env` → `ODDS_API_KEY=...`
- **Where to get a new key:** [the-odds-api.com](https://the-odds-api.com) → Dashboard
- **Free tier limits:** 500 requests/month. The dashboard uses ~1 request per refresh.
- **Signs of exhaustion:** Dashboard shows "API quota exhausted — showing cached games"
- **Fix:** Get a new key OR wait for monthly reset OR upgrade plan

### Checking remaining quota
The dashboard header shows remaining requests after each fetch.
You can also check directly:
```bash
python -c "
from odds_fetcher import fetch_todays_games
d = fetch_todays_games(force_refresh=True)
print('Requests remaining:', d.get('requests_remaining'))
print('Error:', d.get('error'))
"
```

---

## 10. Fresh Start (When All Else Fails)

Run these in order from `C:\Users\yarden\Basketball-God2.0\`:

```bash
# 1. Kill anything on port 5051
netstat -ano | findstr :5051
# (then taskkill /F /PID <each pid>)

# 2. Delete stale caches
del phase5_deploy\output\games_cache_*.json

# 3. Retrain the model (only needed if pkl is missing or corrupted)
python phase7_v2/train_v2.py

# 4. Start the server
python web/server.py

# 5. Open the dashboard
# http://localhost:5051
```

---

## Quick Reference Card

| Problem | First thing to try |
|---|---|
| "Not Found" in browser | Kill old processes, restart server, go to `localhost:5051` |
| "No games found" | Click "Refresh Lines" on dashboard |
| All predictions 50/50 | Check if model pkl exists; retrain if missing |
| Server won't start (port in use) | `taskkill /F /PID <pid>` or bump port in server.py |
| Training script crashes | Check terminal traceback; most common: reset_index missing |
| Live panel hidden | No games live right now — this is normal |
| Odds API errors | Check/replace `ODDS_API_KEY` in `.env` file |
| Bets all showing "AGREE" | Early season or sparse data — model is hedging near 50/50 |

---

*Last updated: March 2026*
