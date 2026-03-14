# 🔮 Rusi's Crystal Ball — Basketball-God 2.0

A machine learning dashboard for NCAA basketball predictions, live betting edge analysis, and in-game hedge calculations. Built on 182,000+ historical games (2010–2025) with real-time ESPN data.

---

## Table of Contents

1. [How the Models Work](#how-the-models-work)
2. [What the Dashboard Shows](#what-the-dashboard-shows)
3. [Signals Explained](#signals-explained)
4. [Live Betting Panel](#live-betting-panel)
5. [Hedge Calculator](#hedge-calculator)
6. [Paper Trading](#paper-trading)
7. [Running the Server](#running-the-server)

---

## How the Models Work

The system has two completely separate models that are automatically routed based on the game date.

### Regular Season Model (Oct – Mar 14)

This is the primary model used for 95%+ of games. It is an **XGBoost + Logistic Regression ensemble** trained on ~182,000 regular season NCAAB games from 2010–2025.

**Architecture:**
- 65% XGBoost (gradient-boosted trees, 600 estimators, depth 5)
- 35% Logistic Regression
- Final probability = `0.65 × XGB_prob + 0.35 × LR_prob`
- Inputs are normalized with StandardScaler before both models

**Validation methodology (CPCV):**
Walk-forward cross-validation with a 1-season embargo to prevent data leakage. Each test season is evaluated on a model trained only on older seasons:
- Test on 2022 → trained on ≤ 2020
- Test on 2023 → trained on ≤ 2021
- Test on 2024 → trained on ≤ 2022
- Test on 2025 → trained on ≤ 2023

**Backtested accuracy: 70.2% average** (2022: 71.8%, 2023: 69.7%, 2024: 69.2%, 2025: 69.9%)

---

### Tournament Model (Mar 15 – Apr 8)

Used exclusively during March Madness. This is a separate XGBoost model trained on 202,529 tournament games (1985–2026) where seedings, bracket position, and historical matchup patterns are the primary signals. Tournament-validated accuracy: ~72%.

---

### The 30 Prediction Features

Every prediction is built from **differential features** — the home team's value minus the away team's value. A positive diff_win_pct means the home team has a higher win percentage; a negative value means the away team is better.

#### Tier 1 — Universal (9 features, always available)
These are computed from game results alone and are 100% filled for every team.

| Feature | What it measures |
|---|---|
| `diff_win_pct` | Season win percentage difference |
| `diff_avg_margin` | Average scoring margin difference (positive = home team wins by more) |
| `diff_sos` | Strength of schedule difference — accounts for who you beat, not just whether you won |
| `diff_rest_days` | Days since last game (home minus away) — fatigue and travel matter |
| `diff_games_last_7` | Games played in past 7 days — back-to-backs hurt |
| `diff_win_streak` | Current win/loss streak difference |
| `diff_h2h_win_pct_5` | Head-to-head win rate over past 5 matchups between these two teams |
| `diff_h2h_win_pct_10` | Head-to-head win rate over past 10 matchups |
| `diff_conf_win_pct` | Conference game win percentage — conference play is a better quality indicator than overall record |

#### Tier 2 — Box Score Stats (17 features, ~92% filled for 2015+ games)
Fetched from ESPN game summaries. These are the advanced efficiency metrics that separate elite teams from average ones.

| Feature | What it measures |
|---|---|
| `diff_efg_pct` | Effective FG% — weights 3-pointers as 1.5× a 2-pointer since they're worth more |
| `diff_opp_efg_pct` | Opponent effective FG% — measures defensive quality |
| `diff_to_rate` | Turnover rate (turnovers per possession) — lower is better offensively |
| `diff_opp_to_rate` | Opponent turnover rate forced — higher means better defense |
| `diff_orb_rate` | Offensive rebound rate — second-chance points are huge in college |
| `diff_drb_rate` | Defensive rebound rate — prevents opponent second chances |
| `diff_ft_rate` | Free throw rate (FTA/FGA) — getting to the line is a sustainable edge |
| `diff_opp_ft_rate` | Opponent free throw rate — foul-prone defense is a liability |
| `diff_fg3_rate` | 3-point attempt rate — teams that shoot more 3s are more volatile |
| `diff_fg3_pct` | 3-point shooting percentage |
| `diff_ast_to_ratio` | Assist-to-turnover ratio — measures ball movement and decision-making |
| `diff_blk_rate` | Block rate — rim protection |
| `diff_stl_rate` | Steal rate — active hands defense |
| `diff_off_eff` | Offensive efficiency (points per 100 possessions) — the gold standard offensive metric |
| `diff_def_eff` | Defensive efficiency (opponent points per 100 possessions) |
| `diff_net_eff` | Net efficiency = offensive efficiency minus defensive efficiency — the single best predictor of team quality |
| `diff_pace` | Possessions per game — fast teams playing slow teams creates volatility |

#### Tier 3 — Power Rankings (4 features, ~97% filled for 2015+ games)
Proxy for the NCAA NET rankings using an Elo rating system computed across all 6,135+ season games.

| Feature | What it measures |
|---|---|
| `diff_massey_avg_rank` | **#1 most important feature (37.9% of model weight)** — Elo power ranking difference. Built by running every team through all season games and ranking by final Elo. Proxy for the NET ranking used by the selection committee. |
| `diff_massey_best_rank` | Best ranking across multiple rating systems |
| `diff_massey_n_systems` | Consensus of how many rating systems agree |
| `diff_massey_spread` | Expected margin implied by Massey composite rankings |

---

### How Live Data Is Fetched

On startup, `SeasonStatsStore` fetches the entire current NCAAB season from ESPN's public scoreboard API — day by day from November 1 to yesterday. This includes:

- Every completed game result (for win%, margin, streak, H2H)
- Up to 200 full box-score summaries per refresh (for eFG%, TO rate, ORB, DRB, blocks, steals)
- Elo ratings computed from all 6,135+ game results → power ranking proxy

Features are computed as **rolling averages** over the team's season-to-date games. A team's `efg_pct` for today's prediction uses every game they've played this season. This means early-season predictions (3–5 games) have more uncertainty than mid/late-season predictions.

Fallback: If a team has fewer than 3 games, **league-average medians** from the 2019–2025 training data are used instead of zeros. This is a significant improvement over Basketball-God 1.0, which zero-filled 28 of 30 features, effectively making it a coin flip on regular season games.

---

## What the Dashboard Shows

### Model Performance (top section)

These are **tournament model** backtested statistics from Phase 4/5 analysis:

- **Tournament Accuracy** — what % of March Madness games did the model correctly predict the winner, averaged across test seasons
- **Accuracy Lift vs Baseline** — how much better the model is than simply always picking the higher seed
- **Avg ESPN Bracket Points** — if you filled out a bracket using the model's picks, how many points would you score on average in an ESPN bracket challenge
- **ECE (Expected Calibration Error)** — measures if the model's probabilities are honest. ECE of 0.02 means when the model says 70%, it actually wins ~70% of the time. Lower is better.

### Today's Games

Each game card shows:
- Team matchup (Away @ Home)
- If there's a **STRONG VALUE** signal, the pick and edge percentage are shown
- ⭐ = strong value bet identified by the model
- Games with value picks have a green border

Click any game card to expand the detail panel, which shows:
- **Win probability** for each team (e.g., "Duke 68.4% / UNC 31.6%")
- **Model odds** — converting the probability to American odds format (e.g., -215 means Duke must win $215 to profit $100)
- **Confidence level** — HIGH (≥65% probability), MED (55-64%), LOW (<55%)
- **REG SEASON MODEL** or **TOURNAMENT MODEL** badge — which model made this prediction
- **Game count** — `(31g / 28g)` means away team has 31 games this season, home team has 28 — more games = more reliable features
- **Bet table** — all available markets (moneyline, spread, totals) with edge and signal for each

---

## Signals Explained

The **signal** for each bet is the model's assessment of value against the market odds.

### How Edge Is Calculated

```
Edge = Model's probability − Market's implied probability

Market's implied probability:
  For negative odds (e.g. -150): |odds| / (|odds| + 100)  →  150/250 = 60%
  For positive odds (e.g. +130): 100 / (odds + 100)       →  100/230 = 43.5%
```

Example: Duke is -150 to win. The market implies a 60% chance. If the model says Duke has a 72% chance, the edge is `72% − 60% = +12%`.

Note: The raw implied probability includes the sportsbook's vig (typically 4-6%). The edge calculation uses the raw implied without removing vig, so a true "break-even" edge is around +3-4% after vig, not 0%.

### Signal Thresholds

| Signal | Edge | What it means |
|---|---|---|
| 🟢 **STRONG VALUE** | ≥ +8% | Model strongly disagrees with the market — the line appears to be mispriced. The model sees significantly more probability here than the market is pricing in. Highest conviction play. |
| 🔵 **MODEL HIGHER** | +4% to +8% | Model sees more value than the market, but not dramatically so. Still a positive expectation bet if the model is calibrated correctly. |
| ⚫ **AGREE** | -4% to +4% | Model and market are in rough agreement. No edge identified. |
| 🟠 **MARKET HIGHER** | -4% to -8% | The market is pricing this side higher than the model thinks it deserves. The model suggests fading this line. |
| 🔴 **FADE** | ≤ -8% | The market is dramatically higher than the model. If anything, consider the opposite side of this bet. |

### Moneyline Edge
Model win probability vs. market implied probability directly.

### Spread Edge
The model converts its win probability to an expected scoring margin using the inverse normal distribution (σ = 11 points, the standard deviation of NCAAB final margins). Then it computes the probability of covering the market spread:

```
Model spread → prob of covering market line = Φ((model_spread − market_line) / 10)
Edge = cover probability − market implied probability
```

### Totals Edge
The model uses a predicted total of 145 points (NCAAB average) and computes:
```
P(Over) = 1 − Φ((market_line − 145) / 15)
Edge = P(Over) − market implied probability for over
```

---

## Live Betting Panel

Appears automatically at the top of Today's Games when any NCAAB game is in progress. Refreshes every **60 seconds**.

For each live game you'll see:
- **Current score and clock**
- **Live win probability bar** — the orange bar fills proportionally to the home team's win probability
- **Home% / Away%** — current live win probabilities for each team
- **Market odds** — the current moneyline from the best available book
- **Live edge** — model's live probability minus what the market is implying right now

### How Live Win Probability Is Calculated

The live probability formula is standard sports analytics (Stern 1994):

```
pregame_spread = σ × Φ⁻¹(pregame_prob)       [convert prob to points]
frac_remaining = minutes_remaining / 40

remaining_uncertainty = 11.0 × √(frac_remaining)
expected_final_margin = current_margin + pregame_spread × frac_remaining
live_home_prob = Φ(expected_final_margin / remaining_uncertainty)
```

Where:
- **σ = 11.0 points** — the empirical standard deviation of NCAAB final scoring margins
- **Φ** is the standard normal CDF
- **pregame_spread** is the model's pre-game predicted margin (home perspective)
- **current_margin** is `home_score − away_score`
- **frac_remaining** shrinks as the game progresses, reducing remaining uncertainty

**Intuition:** Early in the game, most of the uncertainty is still ahead — the current score barely matters. With 2 minutes left and a 10-point lead, the score dominates completely. The `√(frac_remaining)` term handles this scaling naturally.

**Example:** Model had home team at 55% pre-game (spread ≈ +1.4 pts). Home team leads 76–72 with 3 min left.
```
frac_remaining = 3/40 = 0.075
remaining_uncertainty = 11 × √0.075 = 3.01 pts
future_expected = 1.4 × 0.075 = 0.1 pts
expected_final = (76−72) + 0.1 = 4.1 pts
live_prob = Φ(4.1 / 3.01) = Φ(1.36) = 91.3%
```
The home team that was a coin flip pre-game is now ~91% likely to win.

### Live Edge
```
live_edge = live_model_prob − market_implied_prob
```
A positive live edge means the model thinks a team is more likely to win than the current live odds suggest. This can happen when the market is slow to update or when your pre-game bet has become favorable.

---

## Hedge Calculator

Click **💰 Hedge** on any live game to open the hedge calculator.

### Inputs
- **Side** — which team you originally bet on (Home or Away)
- **Odds** — your original American odds at time of bet (e.g. `-110`, `+250`)
- **Stake** — how much you bet originally

### How Hedging Works

A hedge is a bet on the **opposite side** of your original bet to guarantee a profit or reduce your loss regardless of outcome.

**Full Hedge Stake** — the exact amount to bet on the other side to lock in equal profit no matter who wins:
```
original_payout = stake × decimal_odds   [what you'd collect if original bet wins]
original_profit = original_payout − stake

full_hedge_stake = original_profit / (live_decimal_odds_of_other_side − 1)
locked_profit = original_profit − full_hedge_stake
```

**Partial Hedge Stake** — Kelly criterion-based partial hedge, sized proportionally to how much the edge has moved against your original bet:
```
b = live_decimal_odds − 1
kelly_fraction = (live_prob × b − (1 − live_prob)) / b  [clipped to 0–1]
partial_stake = original_stake × kelly_fraction
```

### Hedge Recommendations

| Recommendation | Condition | Action |
|---|---|---|
| 🟢 **HOLD** | Live edge ≥ 0% | Original bet still has positive expected value — no hedge needed |
| 🟠 **PARTIAL HEDGE** | Edge flipped -5% to 0% | Reduce risk but don't eliminate upside — bet partial hedge amount |
| 🔵 **FULL HEDGE** | Edge < -5% and profit still available | Lock in guaranteed profit by betting the full hedge amount |
| 🔴 **CUT LOSSES** | Market fully moved against you | No profitable hedge available — original bet is likely lost |

---

## Paper Trading

Simulated bet tracking using your browser's local storage. No real money involved.

- **Stake $** — default bet size for each wager
- **Threshold** — minimum win probability for a bet to auto-populate (0.65 = only bets where model is ≥65% confident)
- Bets with **STRONG VALUE** or **MODEL HIGHER** signal are auto-added when games load
- Mark each bet **Won** or **Lost** manually after games finish
- **Day Summary** shows realized P&L, win rate, and ROI for today
- **New Day** archives today's bets to History and resets the slip

P&L calculation:
```
Win: profit = to_win amount (stake × (odds/100) for positive odds, stake × (100/|odds|) for negative)
Loss: profit = −stake
ROI = total_pnl / total_staked
```

---

## Running the Server

### First Time Setup
```bash
# Install dependencies
pip install flask xgboost scikit-learn pandas numpy requests

# Train the regular season model (required once)
cd Basketball-God2.0
python phase6_regular_season/train.py

# Pre-warm box score cache (optional but recommended — ~20 min)
# Fetches full game summaries for every game this season
python warmup_cache.py

# Start the server
python web/server.py
```

Open: **http://localhost:5050**

### Daily Use
```bash
python web/server.py
```

The server automatically:
1. Loads both models
2. Fetches all season game results from ESPN
3. Fetches up to 200 new box-score summaries (most recent games first)
4. Computes Elo power rankings from all season results
5. Starts serving predictions

### Keeping Data Fresh
- **↻ Refresh Lines** — re-fetches today's odds from The Odds API
- **↻ Update Stats** — pulls the latest completed game results from ESPN and recomputes all team stats. Run this if there were games last night and the predictions seem stale.
- **Live panel** auto-refreshes every 60 seconds during live games

### API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/daily` | Today's games with predictions and betting edges |
| `GET /api/daily?refresh=1` | Same, but forces fresh odds fetch |
| `GET /api/live` | In-progress games with live scores, win probs, and market odds |
| `POST /api/hedge` | Hedge calculation for an existing bet (body: `game_id`, `original_side`, `original_stake`, `original_odds`) |
| `POST /api/refresh` | Force-refresh ESPN season data and recompute team stats |
| `GET /api/model_status` | Model load status, game count, cache stats |
| `GET /api/odds` | Championship futures from The Odds API |
| `GET /api/data` | Tournament model backtest results and analytics |
| `GET /api/paperbets` | Previous day paper bet P&L |

---

## Architecture Overview

```
Basketball-God2.0/
├── web/
│   ├── server.py              — Flask server, all API routes
│   └── templates/index.html   — Single-page dashboard (all JS inline)
│
├── daily_predictor.py         — Routes predictions to correct model,
│                                computes betting edges for all markets
├── regular_season_model.py    — XGBoost+LR ensemble wrapper
├── season_stats_store.py      — ESPN data fetcher, rolling stat calculator,
│                                Elo power rankings
├── live_game_model.py         — In-game win probability (Stern formula),
│                                hedge calculator
├── odds_fetcher.py            — The Odds API integration (moneyline, spread, totals)
│
├── phase6_regular_season/
│   ├── train.py               — CPCV training pipeline
│   └── output/
│       └── regular_season_model.pkl  — Trained model bundle
│
└── warmup_cache.py            — One-time box score cache warmer
```
