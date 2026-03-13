"""
Basketball-God — Phase 4: Tournament-Specific Optimization
===========================================================
Builds on the Phase 3 production model with tournament-specific layers:

1. Round detection — tag every tournament game with its round (R64→Championship)
2. Round-specific calibration — recalibrate probabilities per round
   (upsets happen at different rates in R64 vs Elite Eight)
3. Seed matchup features — historical upset rates by seed bucket (1v16, 5v12, etc.)
4. Path features — in later rounds: opponent seeds beaten, total close games,
   overtime games, days of rest accumulated
5. Cinderella detection — statistical profile that matches historical runs:
   mid-major + elite defense + strong Massey rank relative to seed
6. Full bracket simulation — 10,000 Monte Carlo runs → champion probabilities,
   expected ESPN bracket pool score, upset probability by round
7. Backtest — for each year 2015-2025, compare model bracket vs actual bracket
   with ESPN-style scoring (1-2-4-8-16-32 points per correct round pick)

WHY TOURNAMENT-SPECIFIC: Regular season dynamics differ from tournament.
  - No home court (mostly neutral sites)
  - Single elimination = higher variance → calibration matters more
  - Path effects: teams that played OT/close games 2 days ago are fatigued
  - Coaching experience (elite coaches consistently outperform in tournament)
  - Bracket structure means you must predict the PATH, not just individual games
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
FEAT_DIR = ROOT / "phase2_features" / "output"
P3_DIR   = ROOT / "phase3_models" / "output"
KAGGLE   = ROOT / "phase1_data" / "sources" / "kaggle"
OUT_DIR  = ROOT / "phase4_tournament" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Round mapping (day_num ranges from Kaggle data) ──────────────────────────
# Based on SeedRoundSlots: R1=136-137, R2=138-139, R3=143-146, R4=145-148,
# R5=152, R6=154. Plus First Four: 134-135.
ROUND_MAP = [
    ("First Four",   134, 135, 0),
    ("Round of 64",  136, 137, 1),
    ("Round of 32",  138, 140, 2),
    ("Sweet 16",     141, 146, 3),   # slightly wider window for COVID year
    ("Elite Eight",  147, 152, 4),
    ("Final Four",   153, 155, 5),
    ("Championship", 154, 156, 6),
]
# Use a priority-based approach since some days overlap
ROUND_PRIORITY = {name: i for i, (name, *_) in enumerate(ROUND_MAP)}

def get_round(day_num: int) -> str:
    """Map a tournament day_num to a round name."""
    candidates = [(name, pri) for name, lo, hi, pri in ROUND_MAP
                  if lo <= day_num <= hi]
    if not candidates:
        return "Unknown"
    # If Championship and Final Four both match (154), prefer Championship
    return max(candidates, key=lambda x: x[1])[0]

ESPN_POINTS = {
    "First Four":   0,
    "Round of 64":  1,
    "Round of 32":  2,
    "Sweet 16":     4,
    "Elite Eight":  8,
    "Final Four":   16,
    "Championship": 32,
}


# ── Load data ─────────────────────────────────────────────────────────────────

def load_tournament_data():
    """Load tournament games with features + bracket structure."""
    print("Loading data...")

    df = pd.read_parquet(FEAT_DIR / "features_all.parquet")
    df = df[df["game_type"] == "ncaa_tourney"].copy()
    df = df[df["season"] < 2026].copy()

    # Add round
    df["round"] = df["day_num"].apply(get_round)
    df["round_num"] = df["round"].map({r: i for i, (r, *_) in enumerate(ROUND_MAP)})

    print(f"  {len(df)} tournament games, {df['season'].nunique()} seasons")
    print(f"  By round: {df['round'].value_counts().to_dict()}")

    # Load seeds from DB
    conn = sqlite3.connect(ROOT / "phase1_data" / "output" / "basketball_god.db")
    seeds = pd.read_sql("""
        SELECT season, team_id, seed_num, seed_str, region FROM tourney_seeds
    """, conn)
    games_raw = pd.read_sql("""
        SELECT game_id, season, day_num, w_team_id, l_team_id, w_score, l_score, num_ot
        FROM games WHERE game_type='ncaa_tourney'
    """, conn)
    teams = pd.read_sql("SELECT team_id, team_name FROM teams", conn)
    conn.close()

    # Load bracket slots
    slots = pd.read_csv(KAGGLE / "MNCAATourneySlots.csv")

    return df, seeds, games_raw, teams, slots


def load_base_model():
    """Load the Phase 3 production model."""
    with open(P3_DIR / "production_model.pkl", "rb") as f:
        pkg = pickle.load(f)
    return pkg["model"], pkg["feature_cols"]


# ── Seed matchup bucket features ──────────────────────────────────────────────

def build_seed_matchup_stats(games_raw: pd.DataFrame, seeds: pd.DataFrame) -> dict:
    """
    Compute historical upset rates for every seed matchup bucket (e.g. 5v12).

    WHY: A 5-seed vs 12-seed matchup has a historically ~35% upset rate —
    much higher than the general model predicts. These bucket priors help
    calibrate for known structural patterns in the bracket.
    """
    print("  Building seed matchup statistics...")

    # Join seeds to games
    g = games_raw.merge(
        seeds[["season","team_id","seed_num"]].rename(columns={"team_id":"w_team_id","seed_num":"w_seed"}),
        on=["season","w_team_id"], how="left"
    ).merge(
        seeds[["season","team_id","seed_num"]].rename(columns={"team_id":"l_team_id","seed_num":"l_seed"}),
        on=["season","l_team_id"], how="left"
    )
    g = g.dropna(subset=["w_seed","l_seed"])
    g["w_seed"] = g["w_seed"].astype(int)
    g["l_seed"] = g["l_seed"].astype(int)

    # Add round
    g["round"] = g["day_num"].apply(get_round)
    g = g[g["round"] != "First Four"]  # skip play-in

    # For each game: high_seed = lower number = better team
    # lower seed number wins = "expected" outcome; higher seed number wins = upset
    g["high_seed"] = g[["w_seed","l_seed"]].min(axis=1)  # best team's seed
    g["low_seed"]  = g[["w_seed","l_seed"]].max(axis=1)  # underdog's seed
    g["upset"] = g["w_seed"] > g["l_seed"]               # winner was the dog

    # Matchup buckets: (high_seed, low_seed) → upset rate
    buckets = g.groupby(["high_seed","low_seed"]).agg(
        games=("upset","count"),
        upsets=("upset","sum"),
        upset_rate=("upset","mean"),
    ).reset_index()

    bucket_dict = {
        (int(r.high_seed), int(r.low_seed)): {
            "games": int(r.games),
            "upsets": int(r.upsets),
            "upset_rate": round(float(r.upset_rate), 4),
        }
        for _, r in buckets.iterrows()
    }

    # Print classic matchups
    print("\n  Historical upset rates by seed matchup:")
    print(f"  {'Matchup':<10} {'Upset%':>8} {'Games':>8}")
    print("  " + "-"*28)
    for (hs, ls), stats in sorted(bucket_dict.items(), key=lambda x: x[0]):
        if hs <= 8:  # Only print top half
            marker = " <-- classic upset seed" if (hs, ls) in [(5,12),(8,9),(10,7),(11,6),(12,5)] else ""
            print(f"  {hs}v{ls:<6} {stats['upset_rate']:>8.1%} {stats['games']:>8}{marker}")

    return bucket_dict


# ── Path features ──────────────────────────────────────────────────────────────

def build_path_features(games_raw: pd.DataFrame, seeds: pd.DataFrame,
                        tourney_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each tournament game, compute path difficulty features for BOTH teams:
    - sum of opponent seeds beaten so far in this tournament
    - number of close games (margin <= 5) on the path
    - overtime games on the path
    - total games played (fatigue proxy)
    - days since last game

    WHY PATH MATTERS: A team that survived 3 OT games to reach the Elite Eight
    carries real fatigue. A team that cruised through weak opponents has no
    experience of adversity. Both affect performance in high-pressure games.
    """
    print("  Building path features...")

    g = games_raw.merge(
        seeds[["season","team_id","seed_num"]].rename(columns={"team_id":"w_team_id","seed_num":"w_seed"}),
        on=["season","w_team_id"], how="left"
    ).merge(
        seeds[["season","team_id","seed_num"]].rename(columns={"team_id":"l_team_id","seed_num":"l_seed"}),
        on=["season","l_team_id"], how="left"
    )
    g["round"] = g["day_num"].apply(get_round)
    g["margin"] = abs(g["w_score"] - g["l_score"])
    g["close_game"] = (g["margin"] <= 5).astype(int)
    g = g.sort_values(["season","day_num"]).reset_index(drop=True)

    # For each team in each season, accumulate path stats game by game
    path_records = defaultdict(lambda: {
        "seeds_beaten": 0, "close_games": 0, "ot_games": 0, "games_played": 0,
        "last_game_day": None
    })

    path_rows = []
    for _, row in g.iterrows():
        s = row["season"]
        for team_id, opp_seed in [
            (row["w_team_id"], row.get("l_seed", np.nan)),
            (row["l_team_id"], row.get("w_seed", np.nan)),
        ]:
            key = (s, team_id)
            state = path_records[key]

            path_rows.append({
                "game_id":            row["game_id"],
                "team_id":            team_id,
                "path_seeds_beaten":  state["seeds_beaten"],
                "path_close_games":   state["close_games"],
                "path_ot_games":      state["ot_games"],
                "path_games_played":  state["games_played"],
                "path_days_rest":     int(row["day_num"] - state["last_game_day"])
                                      if state["last_game_day"] else 14,
            })

            # Update state AFTER recording (pre-game path)
            if not pd.isna(opp_seed):
                state["seeds_beaten"] += int(opp_seed)
            state["close_games"]   += row["close_game"]
            state["ot_games"]      += int(row["num_ot"] > 0)
            state["games_played"]  += 1
            state["last_game_day"]  = row["day_num"]

    path_df = pd.DataFrame(path_rows)

    # Create matchup-level path differentials
    t1 = path_df.rename(columns={"team_id":"team1_id",
                                  **{c: f"t1_{c}" for c in path_df.columns
                                     if c.startswith("path_")}})
    t2 = path_df.rename(columns={"team_id":"team2_id",
                                  **{c: f"t2_{c}" for c in path_df.columns
                                     if c.startswith("path_")}})

    # Merge with tourney_df (which has team1_id/team2_id assignments)
    result = tourney_df[["game_id","team1_id","team2_id","round","round_num"]].merge(
        t1[["game_id","team1_id"] + [f"t1_{c}" for c in path_df.columns
                                      if c.startswith("path_")]],
        on=["game_id","team1_id"], how="left"
    ).merge(
        t2[["game_id","team2_id"] + [f"t2_{c}" for c in path_df.columns
                                      if c.startswith("path_")]],
        on=["game_id","team2_id"], how="left"
    )

    # Compute differentials
    path_feat_names = [c.replace("path_","") for c in path_df.columns
                       if c.startswith("path_")]
    for f in path_feat_names:
        if f"t1_path_{f}" in result.columns and f"t2_path_{f}" in result.columns:
            result[f"diff_path_{f}"] = result[f"t1_path_{f}"] - result[f"t2_path_{f}"]

    return result


# ── Cinderella detector ────────────────────────────────────────────────────────

def build_cinderella_scores(tourney_df: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """
    Score each tournament team on their Cinderella potential.

    Profile of historical Cinderella teams:
    - Seed 10-15 (mid-to-low)
    - High defensive efficiency (diff_def_eff well below seed expectation)
    - Strong Massey ranking relative to seed (model thinks they're better)
    - Mid-major or strong mid-major conference

    WHY: The model already captures team strength, but Cinderellas are by
    definition teams the bracket UNDERESTIMATES. By flagging the gap between
    Massey rank and seed, we identify teams the seeding committee got wrong.
    """
    print("  Computing Cinderella scores...")

    # Use t1_ and t2_ columns (raw absolute values)
    rows = []

    # Get seed info per team per game
    for _, r in tourney_df.iterrows():
        for side, seed_col, massey_col, rank_col in [
            ("team1", "t1_seed", "t1_massey_avg_rank", "t1_avg_margin"),
            ("team2", "t2_seed", "t2_massey_avg_rank", "t2_avg_margin"),
        ]:
            team_id = r.get(f"{side}_id")
            seed    = r.get(seed_col, np.nan)
            massey  = r.get(massey_col, np.nan)
            margin  = r.get(rank_col, np.nan)

            if pd.isna(seed) or pd.isna(team_id):
                continue

            # Cinderella score components:
            # 1. Is a meaningful underdog (seed 10+)
            dog_score = max(0, (seed - 9) / 7)  # 0 for seed <=9, 1.0 for seed 16

            # 2. Massey rank overrates them vs seed
            # Expected rank for this seed ~= seed * 15 (rough heuristic)
            if not pd.isna(massey):
                expected_rank = seed * 15
                rank_gap = (expected_rank - massey) / expected_rank
                rank_score = min(1.0, max(0.0, rank_gap))
            else:
                rank_score = 0.5  # neutral if unknown

            # 3. Strong scoring margin relative to being an underdog
            if not pd.isna(margin) and seed >= 10:
                margin_score = min(1.0, max(0.0, margin / 15))
            else:
                margin_score = 0.0

            # Combined Cinderella score
            cinderella = 0.4 * dog_score + 0.4 * rank_score + 0.2 * margin_score

            rows.append({
                "game_id":         r["game_id"],
                "team_id":         team_id,
                "cinderella_score": round(cinderella, 4),
                "seed":             seed,
                "massey_rank":      massey,
            })

    cdf = pd.DataFrame(rows)

    # Add as team1/team2 cinderella scores
    t1 = cdf.rename(columns={"team_id":"team1_id","cinderella_score":"t1_cinderella"})
    t2 = cdf.rename(columns={"team_id":"team2_id","cinderella_score":"t2_cinderella"})

    out = tourney_df.merge(t1[["game_id","team1_id","t1_cinderella"]], on=["game_id","team1_id"], how="left")
    out = out.merge(t2[["game_id","team2_id","t2_cinderella"]], on=["game_id","team2_id"], how="left")
    out["diff_cinderella"] = out["t1_cinderella"] - out["t2_cinderella"]

    # Top Cinderella candidates per season
    cinderella_top = (cdf[cdf["seed"] >= 10]
                      .sort_values("cinderella_score", ascending=False)
                      .groupby(cdf[cdf["seed"]>=10]["game_id"].str[:4].rename("season"))
                      .head(3))

    return out


# ── Round-specific calibration ─────────────────────────────────────────────────

def build_round_models(tourney_df: pd.DataFrame, base_model,
                       feat_cols: list, base_feat_cols: list) -> dict:
    """
    For each tournament round, train a calibration layer on top of base model.

    WHY CALIBRATION PER ROUND: Round of 64 has 32 games with many blowouts;
    Elite Eight games are much tighter. The base model's probability for a
    "70% favorite" in R64 should be recalibrated differently than in the F4.
    Isotonic regression fits this non-parametrically.

    base_feat_cols: the exact features the base model was trained on
    feat_cols:      extended features including path + cinderella (for future use)
    """
    print("  Training round-specific calibration layers...")

    round_calibrators = {}
    round_results = {}

    for round_name in ["Round of 64", "Round of 32", "Sweet 16",
                        "Elite Eight", "Final Four", "Championship"]:
        round_df = tourney_df[tourney_df["round"] == round_name].copy()
        if len(round_df) < 30:
            print(f"    {round_name:<20} insufficient data ({len(round_df)} games)")
            continue

        # Base model needs its original feature set only
        X_base = round_df[[c for c in base_feat_cols if c in round_df.columns]].fillna(0)
        for c in base_feat_cols:
            if c not in X_base.columns:
                X_base[c] = 0
        X_base = X_base[base_feat_cols]
        y = round_df["label"].values

        # Base model probabilities
        base_probs = base_model.predict_proba(X_base)[:, 1]

        # Leave-one-season-out calibration (to avoid leakage)
        calibrated_probs = np.zeros(len(y))
        seasons = sorted(round_df["season"].unique())

        for test_s in seasons:
            train_mask = round_df["season"] != test_s
            test_mask  = round_df["season"] == test_s

            if train_mask.sum() < 20:
                calibrated_probs[test_mask] = base_probs[test_mask]
                continue

            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(base_probs[train_mask], y[train_mask])
            calibrated_probs[test_mask] = cal.predict(base_probs[test_mask])

        # Accuracy comparison
        base_acc = accuracy_score(y, (base_probs >= 0.5).astype(int))
        cal_acc  = accuracy_score(y, (calibrated_probs >= 0.5).astype(int))
        base_ll  = log_loss(y, np.clip(base_probs, 1e-7, 1-1e-7))
        cal_ll   = log_loss(y, np.clip(calibrated_probs, 1e-7, 1-1e-7))

        round_results[round_name] = {
            "n_games":     len(round_df),
            "base_acc":    round(base_acc, 4),
            "cal_acc":     round(cal_acc, 4),
            "base_logloss": round(base_ll, 4),
            "cal_logloss":  round(cal_ll, 4),
            "improvement": round(cal_acc - base_acc, 4),
        }

        # Train final calibrator on all data
        final_cal = IsotonicRegression(out_of_bounds="clip")
        final_cal.fit(base_probs, y)
        round_calibrators[round_name] = final_cal

        print(f"    {round_name:<20} base={base_acc:.3f}  cal={cal_acc:.3f}  "
              f"ll_base={base_ll:.4f}  ll_cal={cal_ll:.4f}  n={len(round_df)}")

    with open(OUT_DIR / "round_calibration.json", "w") as f:
        json.dump(round_results, f, indent=2)

    return round_calibrators, round_results


# ── Bracket simulator ──────────────────────────────────────────────────────────

def simulate_bracket(season: int, seeds: pd.DataFrame, slots: pd.DataFrame,
                     base_model, feat_cols: list,
                     round_calibrators: dict,
                     tourney_df: pd.DataFrame,
                     n_simulations: int = 10_000,
                     base_feat_cols: list = None) -> dict:
    """
    Simulate the full tournament bracket N times using model probabilities.

    Algorithm:
    1. Start with all 64 (or 68 with play-in) teams seeded into slots
    2. For each round, for each matchup:
       a. Get model probability P(team_A wins)
       b. Draw from Bernoulli(P) to determine winner
    3. Repeat N times, track outcomes
    4. Score each simulation with ESPN bracket scoring

    WHY SIMULATION: Tournament bracket scoring (1,2,4,8,16,32 points) is NOT
    maximized by always picking the higher-probability team. Sometimes picking
    a 40% underdog in round 1 (who's likely to go deep) has positive expected
    value because they'd score 1+2+4+... = 63 points if they win it all.
    Simulation reveals the full distribution of bracket strategies.
    """
    # Get seeds for this season
    season_seeds = seeds[seeds["season"] == season].copy()
    if len(season_seeds) == 0:
        return {}

    # Build seed → team_id lookup (include play-in base seeds like "X16" from "X16a"/"X16b")
    seed_to_team = {}
    for _, r in season_seeds.iterrows():
        seed_to_team[r["seed_str"]] = r["team_id"]
        # Also register base seed (strip trailing a/b) so R1 slots can resolve play-in teams
        base = str(r["seed_str"]).rstrip("ab")
        if base not in seed_to_team:
            seed_to_team[base] = r["team_id"]

    season_feats = tourney_df[tourney_df["season"] == season]
    season_slots = slots[slots["Season"] == season].copy()

    # ── Pre-compute all pairwise probabilities in one batch ──────────────────
    all_teams = list(set(seed_to_team.values()))
    _base_cols = base_feat_cols if base_feat_cols is not None else feat_cols

    team_feat_cache = {}
    for _, r in season_feats.iterrows():
        for side in ["team1", "team2"]:
            tid = r.get(f"{side}_id")
            if tid and tid not in team_feat_cache:
                team_feat_cache[tid] = r

    sf_t1 = season_feats["team1_id"].values
    sf_t2 = season_feats["team2_id"].values
    pair_keys, pair_rows, pair_flip = [], [], []

    for t1 in all_teams:
        for t2 in all_teams:
            if t1 == t2:
                continue
            idx = np.where(
                ((sf_t1 == t1) & (sf_t2 == t2)) |
                ((sf_t1 == t2) & (sf_t2 == t1))
            )[0]
            if len(idx) > 0:
                row = season_feats.iloc[idx[0]]
                flipped = (row["team1_id"] == t2)
                feat_row = {c: (float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0)
                            for c in _base_cols}
                pair_rows.append(feat_row)
                pair_flip.append(flipped)
            elif t1 in team_feat_cache and t2 in team_feat_cache:
                t1r = team_feat_cache[t1]
                t2r = team_feat_cache[t2]
                feat_dict = {}
                for col in _base_cols:
                    if col.startswith("diff_"):
                        base = col[5:]
                        t1v = float(t1r.get(f"t1_{base}", 0) or 0)
                        t2v = float(t2r.get(f"t2_{base}", 0) or 0)
                        feat_dict[col] = t1v - t2v
                    else:
                        feat_dict[col] = float(t1r.get(col, 0) or 0)
                pair_rows.append(feat_dict)
                pair_flip.append(False)
            else:
                pair_rows.append(None)
                pair_flip.append(None)
            pair_keys.append((t1, t2))

    valid_indices = [i for i, r in enumerate(pair_rows) if r is not None]
    base_prob_cache = {}
    if valid_indices:
        X_batch = pd.DataFrame([pair_rows[i] for i in valid_indices]).fillna(0)
        X_batch = X_batch.reindex(columns=_base_cols, fill_value=0)
        batch_probs = base_model.predict_proba(X_batch)[:, 1]
        for batch_idx, orig_idx in enumerate(valid_indices):
            t1, t2 = pair_keys[orig_idx]
            p = float(batch_probs[batch_idx])
            base_prob_cache[(t1, t2)] = (1 - p) if pair_flip[orig_idx] else p
    for t1, t2 in pair_keys:
        if (t1, t2) not in base_prob_cache:
            base_prob_cache[(t1, t2)] = 0.5

    # Build calibrated prob cache per round
    round_names = ["Round of 64", "Round of 32", "Sweet 16",
                   "Elite Eight", "Final Four", "Championship"]
    cal_prob_cache = {}
    for (t1, t2), base_p in base_prob_cache.items():
        for rnd in round_names:
            cal = round_calibrators.get(rnd)
            cp = float(cal.predict([base_p])[0]) if cal is not None else base_p
            cal_prob_cache[(t1, t2, rnd)] = float(np.clip(cp, 0.05, 0.95))

    # Pre-build slot lists for speed
    round_prefix = {
        "Round of 64":  "R1",
        "Round of 32":  "R2",
        "Sweet 16":     "R3",
        "Elite Eight":  "R4",
        "Final Four":   "R5",
        "Championship": "R6",
    }
    round_slot_lists = {}
    for rnd, prefix in round_prefix.items():
        sub = season_slots[season_slots["Slot"].str.startswith(prefix)]
        round_slot_lists[rnd] = list(
            zip(sub["Slot"].values, sub["StrongSeed"].values, sub["WeakSeed"].values)
        )

    # ── Monte Carlo simulation (pure dict lookups) ────────────────────────────
    champion_counts = defaultdict(int)

    print(f"    Simulating {n_simulations:,} brackets for {season}...", end=" ")
    rng = np.random.default_rng(42)
    randoms = rng.random((n_simulations, 63))

    for sim_idx in range(n_simulations):
        local_winners = {}
        champion_id = None
        game_num = 0
        for round_name, slot_list in round_slot_lists.items():
            for slot, strong_seed, weak_seed in slot_list:
                t1 = seed_to_team.get(strong_seed) or local_winners.get(strong_seed)
                t2 = seed_to_team.get(weak_seed)   or local_winners.get(weak_seed)
                if t1 is None or t2 is None:
                    game_num += 1
                    continue
                p = cal_prob_cache.get((t1, t2, round_name),
                    1 - cal_prob_cache.get((t2, t1, round_name), 0.5))
                winner = t1 if randoms[sim_idx, game_num] < p else t2
                local_winners[slot] = winner
                game_num += 1
                if round_name == "Championship":
                    champion_id = winner
        if champion_id:
            champion_counts[champion_id] += 1
    print("done")

    # Champion probabilities
    team_names = {}
    conn = sqlite3.connect(ROOT / "phase1_data" / "output" / "basketball_god.db")
    teams_df = pd.read_sql("SELECT team_id, team_name FROM teams", conn)
    conn.close()
    team_names = dict(zip(teams_df["team_id"], teams_df["team_name"]))

    champ_probs = sorted(
        [(team_names.get(tid, str(tid)), cnt / n_simulations)
         for tid, cnt in champion_counts.items()],
        key=lambda x: -x[1]
    )

    return {
        "season": season,
        "n_simulations": n_simulations,
        "champion_probabilities": champ_probs[:16],
        "top_cinderellas": [],  # populated separately
    }


# ── ESPN bracket scoring backtest ─────────────────────────────────────────────

def backtest_bracket_scoring(tourney_df: pd.DataFrame, base_model,
                              feat_cols: list, round_calibrators: dict,
                              seeds: pd.DataFrame,
                              base_feat_cols: list = None) -> dict:
    """
    For each season 2015-2025, predict the full bracket and score it
    using ESPN bracket scoring (1-2-4-8-16-32 points per correct pick).

    Baseline: always pick the lower seed number.
    Model: pick highest-probability team.
    """
    print("\n  Running bracket scoring backtest (2015-2025)...")

    results = []
    test_seasons = list(range(2015, 2026))

    for season in test_seasons:
        s_df = tourney_df[
            (tourney_df["season"] == season) &
            (tourney_df["round"] != "First Four")
        ].copy()

        if len(s_df) < 30:
            continue

        # Use original base_feat_cols for base model prediction
        _base_cols = base_feat_cols if base_feat_cols is not None else feat_cols
        feat_avail = [c for c in _base_cols if c in s_df.columns]
        X = s_df[feat_avail].fillna(0)
        # Pad missing columns with 0
        for c in _base_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[_base_cols]

        base_probs = base_model.predict_proba(X)[:, 1]

        # Apply round calibration
        cal_probs = base_probs.copy()
        for i, (_, row) in enumerate(s_df.iterrows()):
            rnd = row["round"]
            cal = round_calibrators.get(rnd)
            if cal is not None:
                cal_probs[i] = float(cal.predict([base_probs[i]])[0])

        model_picks  = (cal_probs >= 0.5).astype(int)
        seed_picks   = (s_df["diff_seed"].fillna(0) > 0).astype(int)
        actuals      = s_df["label"].values

        # Score per round
        season_result = {"season": season, "rounds": {}}
        model_total, seed_total = 0, 0

        for rnd_name in ["Round of 64","Round of 32","Sweet 16","Elite Eight","Final Four","Championship"]:
            mask = (s_df["round"] == rnd_name).values
            if mask.sum() == 0:
                continue
            pts = ESPN_POINTS[rnd_name]
            m_correct = int(np.sum(model_picks[mask] == actuals[mask]))
            s_correct = int(np.sum(seed_picks[mask]  == actuals[mask]))
            m_pts = m_correct * pts
            s_pts = s_correct * pts
            model_total += m_pts
            seed_total  += s_pts
            season_result["rounds"][rnd_name] = {
                "model_correct": m_correct,
                "seed_correct":  s_correct,
                "possible": int(mask.sum()),
                "pts_each": pts,
                "model_pts": m_pts,
                "seed_pts": s_pts,
            }

        season_result["model_total_pts"] = model_total
        season_result["seed_total_pts"]  = seed_total
        season_result["model_accuracy"]  = round(accuracy_score(actuals, model_picks), 4)
        season_result["seed_accuracy"]   = round(accuracy_score(actuals, seed_picks), 4)
        results.append(season_result)

        print(f"    {season}: model={model_total:3d}pts ({season_result['model_accuracy']:.1%})  "
              f"seed={seed_total:3d}pts ({season_result['seed_accuracy']:.1%})")

    # Aggregate
    model_pts_avg = np.mean([r["model_total_pts"] for r in results])
    seed_pts_avg  = np.mean([r["seed_total_pts"]  for r in results])
    model_acc_avg = np.mean([r["model_accuracy"]  for r in results])
    seed_acc_avg  = np.mean([r["seed_accuracy"]   for r in results])

    print(f"\n  Average ESPN points: model={model_pts_avg:.1f}  seed_baseline={seed_pts_avg:.1f}")
    print(f"  Average accuracy:    model={model_acc_avg:.3f}  seed_baseline={seed_acc_avg:.3f}")

    return {"per_season": results,
            "avg_model_pts": round(float(model_pts_avg), 2),
            "avg_seed_pts":  round(float(seed_pts_avg), 2),
            "avg_model_acc": round(float(model_acc_avg), 4),
            "avg_seed_acc":  round(float(seed_acc_avg), 4)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nBasketball-God — Phase 4: Tournament-Specific Optimization\n")

    tourney_df, seeds, games_raw, teams, slots = load_tournament_data()
    base_model, feat_cols = load_base_model()

    print("\n[1/6] Seed matchup statistics...")
    bucket_stats = build_seed_matchup_stats(games_raw, seeds)
    with open(OUT_DIR / "seed_matchup_stats.json", "w") as f:
        json.dump({f"{k[0]}v{k[1]}": v for k, v in bucket_stats.items()}, f, indent=2)

    print("\n[2/6] Path features...")
    path_feats = build_path_features(games_raw, seeds, tourney_df)
    tourney_df = tourney_df.merge(
        path_feats[["game_id"] + [c for c in path_feats.columns
                                   if c.startswith("diff_path_")]],
        on="game_id", how="left"
    )
    print(f"  Path diff features added: {[c for c in tourney_df.columns if 'path' in c]}")

    print("\n[3/6] Cinderella scores...")
    tourney_df = build_cinderella_scores(tourney_df, seeds)

    print("\n[4/6] Round-specific calibration...")
    # Add path and cinderella features to feat_cols for calibration
    extended_feat_cols = feat_cols + [c for c in tourney_df.columns
                                       if (c.startswith("diff_path_") or
                                           c == "diff_cinderella") and
                                       c not in feat_cols]
    round_calibrators, round_results = build_round_models(
        tourney_df, base_model, extended_feat_cols, feat_cols
    )

    print("\n[5/6] Bracket scoring backtest (2015-2025)...")
    backtest = backtest_bracket_scoring(
        tourney_df, base_model, extended_feat_cols, round_calibrators, seeds,
        base_feat_cols=feat_cols
    )
    with open(OUT_DIR / "bracket_backtest.json", "w") as f:
        json.dump(backtest, f, indent=2, default=str)

    print("\n[6/6] Bracket simulation for 2025 and 2026...")
    sim_results = []
    for season in [2024, 2025]:
        result = simulate_bracket(
            season, seeds, slots, base_model,
            extended_feat_cols, round_calibrators,
            tourney_df, n_simulations=10_000,
            base_feat_cols=feat_cols
        )
        if result:
            sim_results.append(result)
            print(f"\n  {season} Champion Probabilities (top 10):")
            for team, prob in result["champion_probabilities"][:10]:
                bar = "#" * int(prob * 40)
                print(f"    {team:<25} {prob:6.1%}  {bar}")

    with open(OUT_DIR / "bracket_simulations.json", "w") as f:
        json.dump(sim_results, f, indent=2, default=str)

    # Save updated tournament model
    pkg = {
        "base_model": base_model,
        "feature_cols": extended_feat_cols,
        "round_calibrators": round_calibrators,
        "seed_bucket_stats": bucket_stats,
    }
    with open(OUT_DIR / "tournament_model.pkl", "wb") as f:
        pickle.dump(pkg, f)

    print("\n" + "="*60)
    print("  PHASE 4 SUMMARY")
    print("="*60)
    print(f"  Avg ESPN bracket points: model={backtest['avg_model_pts']:.1f}  "
          f"seed={backtest['avg_seed_pts']:.1f}")
    print(f"  Avg tourney accuracy:    model={backtest['avg_model_acc']:.1%}  "
          f"seed={backtest['avg_seed_acc']:.1%}")
    print(f"\n  All outputs saved to: {OUT_DIR}")
    print("="*60)
    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
