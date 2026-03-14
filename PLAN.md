# Basketball-God 2.0 — Development Plan

**Goal**: Extend tournament-grade accuracy (72%) to regular season NCAAB games (target: 60–65% SU, 54–58% ATS).

---

## Problem Statement

The existing model achieves 72% accuracy on NCAA Tournament games because at Selection Sunday, all 31 features are populated: season win%, margin, efficiency stats, Massey rankings, rest days, H2H, etc.

For daily regular season games, `daily_predictor.py` fills every feature with `0` except `elo_diff` and `home_away`. The model receives 2 signals instead of 31 — equivalent to a coin flip with a slight home-court nudge. That's why accuracy collapses to 33–40%.

---

## Architecture Decision: Parallel Model (Not Retrain)

Tournament games and regular season games have fundamentally different distributions:
- Tournament teams are pre-selected elite programs
- Every game is single-elimination
- Seeding adds a strong prior signal not present in regular season

Mixing them degrades both. **Two specialized models beat one general model.**

```
Incoming game
     │
     ├─ Is it March Madness? ──► NCAAModel (existing, unchanged)
     │
     └─ Regular season? ──────► RegularSeasonModel (new)
```

The router in `daily_predictor.py` detects game type by date and whether teams are seeded.

---

## Step 1 — Regular Season Data Pipeline (`phase6_regular_season/`)

### Historical Data
The Kaggle dataset already contains:
- `MRegularSeasonDetailedResults.csv` — 140K games, 2003–2025, full box scores
- `MRegularSeasonCompactResults.csv` — 200K+ games, 1985–2025, scores only

**We already have the data. We just never computed rolling season stats from it.**

### Live / Current Season Data
- Fetch completed game logs for the current season via **ESPN public scoreboard API** (no API key required, already configured in `config.py`)
- Compute rolling stats for every active D1 team daily and cache to `data/cache/season_stats_YYYY.json`

### Features to Compute (per team, rolling season-to-date before each game)
**Tier 1 (universal):**
- `win_pct`, `avg_margin`, `sos` (strength of schedule), `rest_days`, `games_last_7`, `win_streak`, `h2h_win_pct_5`, `h2h_win_pct_10`, `conf_win_pct`

**Tier 2 (2003+, box scores):**
- `efg_pct`, `opp_efg_pct`, `to_rate`, `opp_to_rate`, `orb_rate`, `drb_rate`, `ft_rate`, `opp_ft_rate`, `fg3_rate`, `fg3_pct`, `ast_to_ratio`, `blk_rate`, `stl_rate`, `off_eff`, `def_eff`, `net_eff`, `pace`

**Tier 3 (Massey/NET):**
- `massey_avg_rank`, `massey_best_rank`, `massey_n_systems`, `massey_spread`

All features expressed as **differentials** (home − away) matching the existing feature schema.

---

## Step 2 — SeasonStatsStore (`season_stats_store.py`)

New class that:
1. On startup (or daily cron), reads all completed games from the current season via ESPN API
2. Computes rolling features for every D1 team
3. Caches results to `data/cache/season_stats_YYYY.json`
4. On `predict_game(home, away)`, looks up both teams' current-season stats and builds the full feature row
5. Falls back to **league-average defaults** (not zero) for any feature that's genuinely unavailable

```python
store = SeasonStatsStore()
store.refresh()  # fetch + compute

features = store.get_matchup_features("Duke", "NC State")
# Returns full 31-feature dict with real data, never zeros
```

---

## Step 3 — League Averages (`data/league_averages.json`)

Pre-computed from 2019–2025 regular season data. Used as fallback for:
- Teams with < 3 games played (too early in season)
- Features genuinely unavailable (e.g., no box score data)

**Zero-fill is eliminated entirely.** A new team = league average, not 0.

---

## Step 4 — RegularSeasonModel (`regular_season_model.py`)

- **Training data**: Regular season games 2010–2025 (~100K games)
- **Feature set**: Same 31-feature vector, but populated with real rolling stats
- **Architecture**: XGBoost (same hyperparams as tournament model as baseline)
- **Validation**: Walk-forward CPCV by season (same as Phase 3)
  - For test season Y: train on ≤ Y-2, test on Y
  - Embargo of 1 season
  - Test years: 2022, 2023, 2024, 2025

---

## Step 5 — Validation & Backtest

Backtest on **2022–2025 regular season** (3 full seasons, ~12K games):

| Metric | Target |
|--------|--------|
| Straight-up accuracy (SU%) | ≥ 60% |
| Against the spread (ATS%) | ≥ 54% |
| Over/Under accuracy | ≥ 52% |

Comparison baselines:
- ELO-only prediction
- Home team always
- Higher win% team always

Output: `phase6_regular_season/output/regular_season_backtest.json`

---

## Step 6 — Daily Predictor Update (`daily_predictor.py`)

Add routing layer:

```python
def predict_game(self, home, away, game_date=None):
    if self._is_tournament_game(game_date):
        return self._predict_tournament(home, away)
    else:
        features = self.season_stats_store.get_matchup_features(home, away, game_date)
        return self._predict_regular_season(features, home, away)
```

- Tournament detection: game date falls within March 15 – April 7 window
- No changes to dashboard, server.py, or API response shape

---

## New Files

| File | Purpose |
|------|---------|
| `season_stats_store.py` | Fetches + caches current season rolling stats |
| `regular_season_model.py` | Trains/loads RegularSeasonModel |
| `phase6_regular_season/fetch.py` | ESPN game log fetcher for current season |
| `phase6_regular_season/features.py` | Rolling feature computation from historical + live data |
| `phase6_regular_season/train.py` | CPCV training pipeline for regular season model |
| `phase6_regular_season/backtest.py` | Validation: SU%, ATS%, O/U% on 2022–2025 |
| `data/league_averages.json` | Pre-computed fallback defaults per feature |

## Unchanged Files

- All tournament code (`phase1_data/` through `phase5_deploy/`)
- `web/server.py`, `web/templates/index.html`
- `odds_fetcher.py`
- `model_training.py`, `elo.py`, `config.py`

---

## Expected Outcomes

| Metric | Current (1.0) | Target (2.0) |
|--------|--------------|--------------|
| Tournament SU% | 72% | 72% (preserved) |
| Regular season SU% | 33–40% | 60–65% |
| Regular season ATS% | ~50% (random) | 54–58% |
| Daily feature completeness | 2 / 31 features | 29–31 / 31 features |

---

## Competitive Benchmarks

| Tool | Regular Season ATS% |
|------|-------------------|
| Leans.AI | 53–58% |
| SportsLine | 55–62% |
| **Basketball-God 2.0 (target)** | **54–58%** |
