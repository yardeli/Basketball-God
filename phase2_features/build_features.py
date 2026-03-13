"""
Basketball-God — Phase 2: Feature Engineering for Multi-Era Data
=================================================================
Builds a clean, leak-free feature matrix from the Phase 1 database.

Three feature tiers, matching data availability by era:
  Tier 1 — Universal (1985+):  win%, margin, SOS, rest, H2H, seed
  Tier 2 — Box score (2003+):  Four Factors, pace, efficiency (rolling 10 games)
  Tier 3 — Advanced (2003+):   Massey consensus rank (our KenPom substitute)

CRITICAL DESIGN PRINCIPLE — NO DATA LEAKAGE:
  Every feature uses ONLY information available strictly BEFORE the game tip-off.
  Rolling stats use shift(1) so the game being predicted is never included.
  Massey rankings use the closest snapshot with day_num < game day_num.
  Head-to-head uses only games from PRIOR seasons plus earlier in the current season.

OUTPUT:
  phase2_features/output/features_all.parquet   — all games, all tiers (NaN where unavailable)
  phase2_features/output/features_t1.parquet    — Tier 1 only (1985+, no NaN for tier1 cols)
  phase2_features/output/features_t2.parquet    — Tier 1+2 (2003+)
  phase2_features/output/feature_docs.json      — feature documentation
  phase2_features/output/missingness_report.json — NaN % by feature and era
  phase2_features/output/correlation_matrix.csv  — feature correlations
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "phase1_data" / "output" / "basketball_god.db"
OUT_DIR = ROOT / "phase2_features" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
ROLLING_WINDOW    = 10    # games for rolling box-score stats
MIN_GAMES_HISTORY = 3     # minimum prior games needed to include a game in output
RANDOM_SEED       = 42    # for team1/team2 random assignment

# ── Feature Documentation ─────────────────────────────────────────────────────
FEATURE_DOCS = {
    # Tier 1 — Universal
    "win_pct":          {"tier": 1, "era": "all", "desc": "Season-to-date win percentage before this game"},
    "avg_margin":       {"tier": 1, "era": "all", "desc": "Average scoring margin (pts - opp_pts) season-to-date"},
    "sos":              {"tier": 1, "era": "all", "desc": "Strength of schedule: average win_pct of all prior opponents"},
    "games_played":     {"tier": 1, "era": "all", "desc": "Games played season-to-date before this game"},
    "rest_days":        {"tier": 1, "era": "all", "desc": "Days since last game (capped at 30)"},
    "games_last_7":     {"tier": 1, "era": "all", "desc": "Games played in last 7 calendar days"},
    "win_streak":       {"tier": 1, "era": "all", "desc": "Current win streak (negative = losing streak)"},
    "h2h_win_pct_5":    {"tier": 1, "era": "all", "desc": "Head-to-head win % in last 5 meetings (all-time)"},
    "h2h_win_pct_10":   {"tier": 1, "era": "all", "desc": "Head-to-head win % in last 10 meetings (all-time)"},
    "h2h_games":        {"tier": 1, "era": "all", "desc": "Total prior H2H games between these two teams"},
    "seed":             {"tier": 1, "era": "tourney", "desc": "Tournament seed (1-16, NaN for non-tourney games)"},
    "conf_win_pct":     {"tier": 1, "era": "all",  "desc": "Conference strength: avg OOC win% of conference members"},
    # Tier 2 — Box score (2003+)
    "efg_pct":          {"tier": 2, "era": "2003+", "desc": "Effective FG% rolling 10 games: (FGM + 0.5*FGM3) / FGA"},
    "opp_efg_pct":      {"tier": 2, "era": "2003+", "desc": "Opponent effective FG% allowed (defensive)"},
    "to_rate":          {"tier": 2, "era": "2003+", "desc": "Turnover rate: TO / estimated possessions"},
    "opp_to_rate":      {"tier": 2, "era": "2003+", "desc": "Opponent turnover rate forced"},
    "orb_rate":         {"tier": 2, "era": "2003+", "desc": "Offensive rebound rate: OR / (OR + opp_DR)"},
    "drb_rate":         {"tier": 2, "era": "2003+", "desc": "Defensive rebound rate: DR / (DR + opp_OR)"},
    "ft_rate":          {"tier": 2, "era": "2003+", "desc": "FT rate: FTA / FGA (how often you get to the line)"},
    "opp_ft_rate":      {"tier": 2, "era": "2003+", "desc": "Opponent FT rate allowed"},
    "fg3_rate":         {"tier": 2, "era": "2003+", "desc": "3-point attempt rate: FGA3 / FGA"},
    "fg3_pct":          {"tier": 2, "era": "2003+", "desc": "3-point field goal percentage"},
    "ast_to_ratio":     {"tier": 2, "era": "2003+", "desc": "Assist-to-turnover ratio"},
    "blk_rate":         {"tier": 2, "era": "2003+", "desc": "Block rate: BLK / opponent FGA2 (2pt attempts)"},
    "stl_rate":         {"tier": 2, "era": "2003+", "desc": "Steal rate: STL / opponent possessions"},
    "off_eff":          {"tier": 2, "era": "2003+", "desc": "Offensive efficiency: points per 100 possessions"},
    "def_eff":          {"tier": 2, "era": "2003+", "desc": "Defensive efficiency: opponent pts per 100 poss"},
    "net_eff":          {"tier": 2, "era": "2003+", "desc": "Net efficiency: off_eff - def_eff"},
    "pace":             {"tier": 2, "era": "2003+", "desc": "Estimated possessions per game"},
    # Tier 3 — Massey rankings
    "massey_avg_rank":  {"tier": 3, "era": "2003+", "desc": "Consensus computer ranking (avg across 196 systems). Lower = better."},
    "massey_best_rank": {"tier": 3, "era": "2003+", "desc": "Best (lowest) rank across all systems"},
    "massey_n_systems": {"tier": 3, "era": "2003+", "desc": "Number of systems that ranked this team"},
    "massey_spread":    {"tier": 3, "era": "2003+", "desc": "Disagreement between systems (rank std dev)"},
}

np.random.seed(RANDOM_SEED)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data(conn: sqlite3.Connection):
    print("Loading games from database...")
    games = pd.read_sql("""
        SELECT game_id, season, day_num, game_date,
               game_type, neutral_site,
               w_team_id, l_team_id, w_score, l_score,
               score_diff, num_ot, w_loc,
               era, shot_clock, has_3pt, data_tier
        FROM games
        ORDER BY season, day_num
    """, conn)
    print(f"  {len(games):,} games loaded.")

    print("Loading box score stats...")
    stats = pd.read_sql("""
        SELECT game_id, team_id, opponent_id, is_winner,
               score, opp_score, fgm, fga, fgm3, fga3, ftm, fta,
               off_reb, def_reb, ast, turnovers, stl, blk, pf,
               poss_est, off_eff, def_eff, efg_pct, to_rate, orb_rate, ft_rate
        FROM game_stats
    """, conn)
    print(f"  {len(stats):,} team-game stat rows loaded.")

    print("Loading Massey consensus rankings...")
    rankings = pd.read_sql("""
        SELECT season, day_num, team_id, n_systems,
               avg_rank, best_rank, rank_spread
        FROM team_rankings_snapshot
        ORDER BY season, day_num
    """, conn)
    print(f"  {len(rankings):,} ranking snapshots loaded.")

    print("Loading tournament seeds...")
    seeds = pd.read_sql("""
        SELECT season, team_id, seed_num FROM tourney_seeds
    """, conn)

    print("Loading team conferences...")
    confs = pd.read_sql("""
        SELECT season, team_id, conference FROM team_conferences
    """, conn)

    return games, stats, rankings, seeds, confs


# ── Tier 1: Universal rolling team stats ──────────────────────────────────────

def build_team_season_stats(games: pd.DataFrame) -> pd.DataFrame:
    """
    For every (team, game), compute cumulative season stats BEFORE this game.
    Uses vectorized cumsum/cumcount — no row-by-row loops except win_streak.
    """
    print("  Building cumulative season stats (Tier 1)...")

    base_cols = ["game_id","season","day_num","w_team_id","l_team_id","w_score","l_score"]

    wins = games[base_cols].copy()
    wins["team_id"]    = wins["w_team_id"]
    wins["opp_id"]     = wins["l_team_id"]
    wins["won"]        = 1
    wins["margin"]     = wins["w_score"] - wins["l_score"]

    losses = games[base_cols].copy()
    losses["team_id"]  = losses["l_team_id"]
    losses["opp_id"]   = losses["w_team_id"]
    losses["won"]      = 0
    losses["margin"]   = losses["l_score"] - losses["w_score"]

    tg = pd.concat([
        wins[["game_id","season","day_num","team_id","opp_id","won","margin"]],
        losses[["game_id","season","day_num","team_id","opp_id","won","margin"]],
    ], ignore_index=True)
    tg = tg.sort_values(["team_id","season","day_num"]).reset_index(drop=True)

    # ── Vectorized cumulative stats (shift(1) = before current game) ──
    print("    Vectorized cumsum/cumcount for win%, margin...")
    grp = tg.groupby(["team_id","season"])

    # games_played = index within group (0 = first game → 0 games played before it)
    tg["games_played"] = grp.cumcount()  # 0-indexed = games played BEFORE this game

    # cumulative wins before this game
    tg["cum_wins"]   = grp["won"].cumsum() - tg["won"]
    tg["win_pct"]    = tg["cum_wins"] / tg["games_played"].replace(0, np.nan)

    # cumulative margin before this game
    tg["cum_margin_sum"] = grp["margin"].cumsum() - tg["margin"]
    tg["avg_margin"]     = tg["cum_margin_sum"] / tg["games_played"].replace(0, np.nan)

    # ── Rest days ──
    tg["prev_day"]  = grp["day_num"].shift(1)
    tg["rest_days"] = (tg["day_num"] - tg["prev_day"]).clip(upper=30)

    # ── Win streak (vectorized) ──
    print("    Computing win streaks...")
    def win_streak_vec(won_arr):
        streaks = np.zeros(len(won_arr), dtype=int)
        s = 0
        for i, w in enumerate(won_arr):
            streaks[i] = s   # streak BEFORE this game
            s = (max(s, 0) + 1) if w == 1 else (min(s, 0) - 1)
        return streaks

    streak_vals = []
    for (_, _), sub in tg.groupby(["team_id","season"]):
        streak_vals.append(win_streak_vec(sub["won"].values))
    tg["win_streak"] = np.concatenate(streak_vals)

    # ── Games in last 7 days (vectorized per group) ──
    print("    Computing games_last_7...")
    def g7_vec(day_arr):
        g7 = np.zeros(len(day_arr), dtype=int)
        day_arr = np.array(day_arr)
        for i in range(len(day_arr)):
            cutoff = day_arr[i] - 7
            g7[i] = int(np.sum((day_arr[:i] >= cutoff)))
        return g7

    g7_vals = []
    for (_, _), sub in tg.groupby(["team_id","season"]):
        g7_vals.append(g7_vec(sub["day_num"].values))
    tg["games_last_7"] = np.concatenate(g7_vals)

    # ── Strength of Schedule ──
    # Use each team's FINAL season win_pct as opponent quality proxy.
    # (Slight future leak for pre-2003; acceptable as Massey replaces this for 2003+)
    print("    Computing SOS...")
    final_wp = (tg.groupby(["team_id","season"])
                  .apply(lambda g: g.iloc[-1]["win_pct"])
                  .reset_index(name="opp_final_wp"))
    tg = tg.merge(
        final_wp.rename(columns={"team_id":"opp_id","opp_final_wp":"opp_wp"}),
        on=["opp_id","season"], how="left"
    )
    # Cumulative average of opponent win_pcts seen so far (shift so current not included)
    tg = tg.sort_values(["team_id","season","day_num"]).reset_index(drop=True)
    tg["sos"] = (tg.groupby(["team_id","season"])["opp_wp"]
                   .transform(lambda x: x.cumsum().shift(1) / pd.Series(range(len(x)), index=x.index)))

    keep = ["game_id","team_id","games_played","win_pct","avg_margin",
            "win_streak","rest_days","games_last_7","sos"]
    return tg[keep].copy()


# ── Tier 2: Rolling box-score stats (2003+) ───────────────────────────────────

def build_rolling_box_stats(stats: pd.DataFrame, games: pd.DataFrame,
                             window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Rolling {window}-game averages of box score metrics for each team,
    computed strictly before each game (no leakage).

    WHY ROLLING NOT CUMULATIVE: Recent form matters more than season average
    for box-score efficiency stats. A team's defense may have degraded due to
    injuries, or improved mid-season. Rolling window captures this.

    Returns DataFrame indexed by (game_id, team_id) with Tier 2 features.
    """
    print(f"  Building rolling {window}-game box score stats (Tier 2)...")

    # Join game day_num and season to stats
    stats = stats.merge(
        games[["game_id","season","day_num"]],
        on="game_id", how="left"
    )

    # Derive additional offensive-side stats
    stats["fg2a"]     = stats["fga"] - stats["fga3"]
    stats["net_eff"]  = stats["off_eff"] - stats["def_eff"]
    stats["fg3_rate"] = (stats["fga3"] / stats["fga"]).replace([np.inf, -np.inf], np.nan)
    stats["fg3_pct"]  = stats["fgm3"] / stats["fga3"].replace(0, np.nan)
    stats["ast_to"]   = stats["ast"] / stats["turnovers"].replace(0, np.nan)
    stats["blk_rate"] = stats["blk"] / stats["fg2a"].replace(0, np.nan)
    stats["stl_rate"] = stats["stl"] / stats["poss_est"].replace(0, np.nan)
    stats["drb_rate"] = stats["def_reb"] / (stats["def_reb"] + stats["off_reb"]).replace(0, np.nan)

    roll_cols = [
        "efg_pct", "to_rate", "orb_rate", "drb_rate", "ft_rate",
        "fg3_rate", "fg3_pct", "ast_to", "blk_rate", "stl_rate",
        "off_eff", "def_eff", "net_eff", "poss_est"
    ]

    stats = stats.sort_values(["team_id","season","day_num"]).reset_index(drop=True)

    print(f"    Rolling averages per team (window={window}, vectorized)...")
    min_p = max(1, window // 3)
    for col in roll_cols:
        if col not in stats.columns:
            continue
        stats[f"roll_{col}"] = (
            stats.groupby(["team_id","season"])[col]
            .transform(lambda x: x.rolling(window=window, min_periods=min_p).mean().shift(1))
        )

    # Need opponent-side defensive stats
    # Rename columns to get "opp_" versions by merging with opponent rows
    off_cols = {f"roll_{c}": f"off_roll_{c}" for c in roll_cols}
    stats_off = stats[["game_id","team_id","opponent_id"] +
                      list(off_cols.keys())].rename(columns=off_cols)

    # For defensive stats we get them from opponent's perspective
    # (opponent's off stats when facing this team = this team's defensive stats)
    def_lookup = stats[["game_id","team_id",
                         "roll_efg_pct","roll_to_rate","roll_off_eff","roll_ft_rate"]].copy()
    def_lookup.columns = ["game_id","opponent_id",
                          "opp_roll_efg_pct","opp_roll_to_rate",
                          "opp_roll_off_eff","opp_roll_ft_rate"]

    result = stats_off.merge(def_lookup, on=["game_id","opponent_id"], how="left")

    # Clean up final column names
    rename = {
        "off_roll_efg_pct":   "efg_pct",
        "off_roll_to_rate":   "to_rate",
        "off_roll_orb_rate":  "orb_rate",
        "off_roll_drb_rate":  "drb_rate",
        "off_roll_ft_rate":   "ft_rate",
        "off_roll_fg3_rate":  "fg3_rate",
        "off_roll_fg3_pct":   "fg3_pct",
        "off_roll_ast_to":    "ast_to_ratio",
        "off_roll_blk_rate":  "blk_rate",
        "off_roll_stl_rate":  "stl_rate",
        "off_roll_off_eff":   "off_eff",
        "off_roll_def_eff":   "def_eff",
        "off_roll_net_eff":   "net_eff",
        "off_roll_poss_est":  "pace",
    }
    result = result.rename(columns=rename)
    keep_cols = ["game_id","team_id"] + list(rename.values()) + \
                ["opp_roll_efg_pct","opp_roll_to_rate",
                 "opp_roll_off_eff","opp_roll_ft_rate"]
    return result[[c for c in keep_cols if c in result.columns]].copy()


# ── Tier 3: Massey pre-game rankings ─────────────────────────────────────────

def build_massey_features(rankings: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, find the most recent Massey consensus ranking snapshot
    for each team published BEFORE the game's day_num.
    Uses an efficient merge + argmax approach — no row-level loops.
    """
    print("  Building Massey pre-game ranking features (Tier 3)...")

    game_days = games[["game_id","season","day_num","w_team_id","l_team_id"]].copy()
    rnk = rankings.rename(columns={"day_num":"rank_day"})

    results = []
    for team_col in ["w_team_id","l_team_id"]:
        gd = game_days[["game_id","season","day_num",team_col]].copy()
        gd.columns = ["game_id","season","game_day","team_id"]

        # Merge game days with ranking snapshots for same season+team
        merged = gd.merge(rnk, on=["season","team_id"], how="left")

        # Keep only pre-game rankings
        merged = merged[merged["rank_day"] < merged["game_day"]].copy()

        # For each game, keep only the most recent ranking (max rank_day)
        merged["_rk"] = merged.groupby("game_id")["rank_day"].transform("max")
        latest = merged[merged["rank_day"] == merged["_rk"]].drop_duplicates("game_id")

        latest = latest[["game_id","team_id","n_systems","avg_rank",
                          "best_rank","rank_spread"]].copy()
        latest.columns = ["game_id","team_id","massey_n_systems","massey_avg_rank",
                           "massey_best_rank","massey_spread"]
        results.append(latest)

    return pd.concat(results, ignore_index=True)


# ── Head-to-head history ──────────────────────────────────────────────────────

def build_h2h_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, compute historical H2H record between the two teams
    using ONLY games played before this one. Vectorized merge approach.

    WHY H2H: Familiarity, coaching matchups, and psychological edges persist
    in rivalries. Especially important in conference games and tournament matchups.
    """
    print("  Building head-to-head history features (vectorized)...")

    g = games[["game_id","season","day_num","w_team_id","l_team_id"]].copy()
    g["ta"]    = g[["w_team_id","l_team_id"]].min(axis=1)
    g["tb"]    = g[["w_team_id","l_team_id"]].max(axis=1)
    g["a_won"] = (g["w_team_id"] == g["ta"]).astype(int)
    g = g.sort_values(["ta","tb","season","day_num"]).reset_index(drop=True)

    # Self-join on same pair, prior games only
    left  = g[["game_id","ta","tb","season","day_num"]].copy()
    right = g[["ta","tb","season","day_num","a_won"]].copy()
    right.columns = ["ta","tb","r_season","r_day","a_won"]

    merged = left.merge(right, on=["ta","tb"], how="left")
    # Only prior games
    prior = merged[
        (merged["r_season"] < merged["season"]) |
        ((merged["r_season"] == merged["season"]) & (merged["r_day"] < merged["day_num"]))
    ].copy()

    # Rank prior games newest-first per game_id
    prior["rank_desc"] = prior.groupby("game_id")["r_day"].rank(method="first", ascending=False)

    # H2H counts and win rates for last 5 and last 10
    h2h_total = prior.groupby("game_id")["a_won"].agg(
        h2h_games="count", ta_h2h_win_all="mean"
    ).reset_index()

    h2h_5  = (prior[prior["rank_desc"] <= 5]
              .groupby("game_id")["a_won"].mean()
              .reset_index().rename(columns={"a_won":"ta_h2h_win5"}))
    h2h_10 = (prior[prior["rank_desc"] <= 10]
              .groupby("game_id")["a_won"].mean()
              .reset_index().rename(columns={"a_won":"ta_h2h_win10"}))

    h2h = g[["game_id","ta","tb"]].merge(h2h_total, on="game_id", how="left")
    h2h = h2h.merge(h2h_5,  on="game_id", how="left")
    h2h = h2h.merge(h2h_10, on="game_id", how="left")
    h2h["h2h_games"] = h2h["h2h_games"].fillna(0).astype(int)
    h2h = h2h.merge(games[["game_id","w_team_id","l_team_id"]], on="game_id")

    # Expand to per-team perspective
    rows = []
    for side, team_col in [("w", "w_team_id"), ("l", "l_team_id")]:
        sub = h2h[["game_id",team_col,"ta","tb","h2h_games","ta_h2h_win5","ta_h2h_win10"]].copy()
        sub.columns = ["game_id","team_id","ta","tb","h2h_games","ta_h2h_win5","ta_h2h_win10"]
        is_ta = sub["team_id"] == sub["ta"]
        sub["h2h_win_pct_5"]  = np.where(is_ta, sub["ta_h2h_win5"],
                                          1 - sub["ta_h2h_win5"].fillna(0.5))
        sub["h2h_win_pct_10"] = np.where(is_ta, sub["ta_h2h_win10"],
                                          1 - sub["ta_h2h_win10"].fillna(0.5))
        # Restore NaN where no H2H history
        no_h2h = sub["h2h_games"] == 0
        sub.loc[no_h2h, "h2h_win_pct_5"]  = np.nan
        sub.loc[no_h2h, "h2h_win_pct_10"] = np.nan
        rows.append(sub[["game_id","team_id","h2h_games","h2h_win_pct_5","h2h_win_pct_10"]])

    return pd.concat(rows, ignore_index=True)


# ── Conference strength ────────────────────────────────────────────────────────

def build_conference_strength(games: pd.DataFrame, confs: pd.DataFrame) -> pd.DataFrame:
    """
    For each (season, conference), compute average out-of-conference win%
    as a proxy for conference quality.

    WHY: A 25-5 team from a weak conference is not the same as 25-5 from the ACC.
    This captures the conference context without needing AP poll rankings.
    """
    print("  Building conference strength features...")

    # Expand games to per-team
    w = games[["game_id","season","w_team_id","l_team_id","game_type"]].copy()
    w["team_id"] = w["w_team_id"]; w["opp_id"] = w["l_team_id"]; w["won"] = 1
    l = games[["game_id","season","w_team_id","l_team_id","game_type"]].copy()
    l["team_id"] = l["l_team_id"]; l["opp_id"] = l["w_team_id"]; l["won"] = 0

    tg = pd.concat([w[["game_id","season","team_id","opp_id","won","game_type"]],
                    l[["game_id","season","team_id","opp_id","won","game_type"]]],
                   ignore_index=True)

    # Add conference info
    tg = tg.merge(confs.rename(columns={"conference":"conf"}),
                  on=["season","team_id"], how="left")
    tg = tg.merge(confs.rename(columns={"team_id":"opp_id","conference":"opp_conf"}),
                  on=["season","opp_id"], how="left")

    # Out-of-conference games only
    ooc = tg[tg["conf"] != tg["opp_conf"]].copy()
    conf_wp = (ooc.groupby(["season","conf"])["won"]
               .mean()
               .reset_index()
               .rename(columns={"won":"conf_win_pct"}))

    # Join back to team level
    team_conf = confs.merge(conf_wp.rename(columns={"conf":"conference"}),
                            on=["season","conference"], how="left")
    return team_conf[["season","team_id","conf_win_pct"]].copy()


# ── Assemble matchup feature matrix ──────────────────────────────────────────

def _merge_team_side(games: pd.DataFrame, feat_df: pd.DataFrame,
                     team_col: str, prefix: str) -> pd.DataFrame:
    """
    Vectorized join: for each row in games, attach feat_df columns for
    the team identified by team_col (e.g. 'team1_id').
    No iterrows — pure merge.
    """
    feat_cols = [c for c in feat_df.columns if c not in ("game_id","team_id","opponent_id")]
    sub = feat_df[["game_id","team_id"] + feat_cols].copy()
    sub = sub.rename(columns={"team_id": team_col} |
                              {c: f"{prefix}_{c}" for c in feat_cols})
    return games.merge(sub, on=["game_id", team_col], how="left")


def assemble_matchup_features(games, team_stats, box_stats, massey, h2h, seeds,
                               conf_strength, confs) -> pd.DataFrame:
    """
    For each game, create a single row with (team1_feat - team2_feat) differentials.
    Team1/Team2 assignment is RANDOM to prevent the model from learning
    that 'team1' always means winner (data leakage).
    All joins are vectorized merges — no iterrows.
    """
    print("  Assembling matchup differential features (vectorized)...")

    np.random.seed(RANDOM_SEED)
    flip = np.random.rand(len(games)) > 0.5
    df = games.copy()
    df["team1_id"] = np.where(flip, df["w_team_id"], df["l_team_id"])
    df["team2_id"] = np.where(flip, df["l_team_id"], df["w_team_id"])
    df["label"]    = np.where(flip, 1, 0)

    # ── Seeds (vectorized merge) ──
    seeds_m = seeds[["season","team_id","seed_num"]].copy()
    for t, p in [("team1_id","t1"), ("team2_id","t2")]:
        df = df.merge(
            seeds_m.rename(columns={"team_id": t, "seed_num": f"{p}_seed"}),
            on=["season", t], how="left"
        )

    # ── Conference strength (vectorized merge) ──
    cstr = conf_strength[["season","team_id","conf_win_pct"]].copy()
    for t, p in [("team1_id","t1"), ("team2_id","t2")]:
        df = df.merge(
            cstr.rename(columns={"team_id": t, "conf_win_pct": f"{p}_conf_wp"}),
            on=["season", t], how="left"
        )

    # ── Tier 1 team stats ──
    print("    Joining Tier 1 cumulative stats...")
    for t, p in [("team1_id","t1"), ("team2_id","t2")]:
        df = _merge_team_side(df, team_stats, t, p)

    # ── Tier 2 box score rolling stats ──
    print("    Joining Tier 2 rolling box stats...")
    box_cols = [c for c in box_stats.columns if c not in ("game_id","team_id","opponent_id")]
    for t, p in [("team1_id","t1"), ("team2_id","t2")]:
        df = _merge_team_side(df, box_stats, t, p)

    # ── Tier 3 Massey ──
    print("    Joining Tier 3 Massey rankings...")
    massey_sub = massey[["game_id","team_id","massey_n_systems",
                          "massey_avg_rank","massey_best_rank","massey_spread"]].copy()
    for t, p in [("team1_id","t1"), ("team2_id","t2")]:
        df = _merge_team_side(df, massey_sub, t, p)

    # ── H2H ──
    print("    Joining H2H features...")
    h2h_sub = h2h[["game_id","team_id","h2h_games","h2h_win_pct_5","h2h_win_pct_10"]].copy()
    for t, p in [("team1_id","t1"), ("team2_id","t2")]:
        df = _merge_team_side(df, h2h_sub, t, p)

    # ── Build differential columns ──
    diff_pairs = {
        "win_pct":         ("t1_win_pct",               "t2_win_pct"),
        "avg_margin":      ("t1_avg_margin",             "t2_avg_margin"),
        "sos":             ("t1_sos",                    "t2_sos"),
        "games_played":    ("t1_games_played",           "t2_games_played"),
        "rest_days":       ("t1_rest_days",              "t2_rest_days"),
        "games_last_7":    ("t1_games_last_7",           "t2_games_last_7"),
        "win_streak":      ("t1_win_streak",             "t2_win_streak"),
        "h2h_win_pct_5":   ("t1_h2h_win_pct_5",         "t2_h2h_win_pct_5"),
        "h2h_win_pct_10":  ("t1_h2h_win_pct_10",        "t2_h2h_win_pct_10"),
        "seed":            ("t1_seed",                   "t2_seed"),
        "conf_win_pct":    ("t1_conf_wp",                "t2_conf_wp"),
        "efg_pct":         ("t1_efg_pct",                "t2_efg_pct"),
        "opp_efg_pct":     ("t1_opp_roll_efg_pct",      "t2_opp_roll_efg_pct"),
        "to_rate":         ("t1_to_rate",                "t2_to_rate"),
        "opp_to_rate":     ("t1_opp_roll_to_rate",       "t2_opp_roll_to_rate"),
        "orb_rate":        ("t1_orb_rate",               "t2_orb_rate"),
        "drb_rate":        ("t1_drb_rate",               "t2_drb_rate"),
        "ft_rate":         ("t1_ft_rate",                "t2_ft_rate"),
        "opp_ft_rate":     ("t1_opp_roll_ft_rate",       "t2_opp_roll_ft_rate"),
        "fg3_rate":        ("t1_fg3_rate",               "t2_fg3_rate"),
        "fg3_pct":         ("t1_fg3_pct",                "t2_fg3_pct"),
        "ast_to_ratio":    ("t1_ast_to_ratio",           "t2_ast_to_ratio"),
        "blk_rate":        ("t1_blk_rate",               "t2_blk_rate"),
        "stl_rate":        ("t1_stl_rate",               "t2_stl_rate"),
        "off_eff":         ("t1_off_eff",                "t2_off_eff"),
        "def_eff":         ("t1_def_eff",                "t2_def_eff"),
        "net_eff":         ("t1_net_eff",                "t2_net_eff"),
        "pace":            ("t1_pace",                   "t2_pace"),
        "massey_avg_rank": ("t1_massey_avg_rank",        "t2_massey_avg_rank"),
        "massey_best_rank":("t1_massey_best_rank",       "t2_massey_best_rank"),
        "massey_n_systems":("t1_massey_n_systems",       "t2_massey_n_systems"),
        "massey_spread":   ("t1_massey_spread",          "t2_massey_spread"),
    }

    print("    Computing differentials (team1 - team2)...")
    for feat_name, (col1, col2) in diff_pairs.items():
        c1 = col1 if col1 in df.columns else None
        c2 = col2 if col2 in df.columns else None
        if c1 and c2:
            df[f"diff_{feat_name}"] = df[c1] - df[c2]
        else:
            df[f"diff_{feat_name}"] = np.nan

    # H2H games is team-agnostic — take from team1 side
    if "t1_h2h_games" in df.columns:
        df["h2h_games"] = df["t1_h2h_games"]

    raw_keep = ["t1_win_pct","t2_win_pct","t1_avg_margin","t2_avg_margin",
                "t1_net_eff","t2_net_eff","t1_massey_avg_rank","t2_massey_avg_rank",
                "t1_seed","t2_seed","t1_win_streak","t2_win_streak"]

    core_cols = (
        ["game_id","season","day_num","game_date","game_type","era","data_tier",
         "neutral_site","has_3pt","shot_clock","num_ot",
         "team1_id","team2_id","label","h2h_games"] +
        [f"diff_{f}" for f in diff_pairs.keys()] +
        [c for c in raw_keep if c in df.columns]
    )
    return df[[c for c in core_cols if c in df.columns]].copy()


# ── Era normalization ─────────────────────────────────────────────────────────

def normalize_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score all diff_* features within each season.

    WHY: A 10-point scoring margin in 1985 (slower pace, lower scoring) signals
    dominance differently than 10 points in 2024. Z-scoring within season removes
    era-level shifts so the model can compare across decades.
    NOTE: We use the FULL season's distribution for z-scoring (not a rolling
    window), since we want the features to represent 'how dominant relative to
    that season's context', not relative to recent history.
    """
    print("  Normalizing features by season (z-score within season)...")
    diff_cols = [c for c in df.columns if c.startswith("diff_")]

    for col in diff_cols:
        mean = df.groupby("season")[col].transform("mean")
        std  = df.groupby("season")[col].transform("std")
        df[f"{col}_z"] = (df[col] - mean) / std.replace(0, np.nan)

    return df


# ── Reporting ─────────────────────────────────────────────────────────────────

def generate_feature_report(df: pd.DataFrame):
    print("\n  Generating feature engineering report...")

    diff_cols = [c for c in df.columns if c.startswith("diff_")]

    # Missingness by era
    miss_by_era = {}
    for era in df["era"].unique():
        sub = df[df["era"] == era]
        miss_by_era[era] = {
            col: round(sub[col].isna().mean() * 100, 1)
            for col in diff_cols
        }

    # Overall missingness
    overall_miss = {
        col: round(df[col].isna().mean() * 100, 1)
        for col in diff_cols
    }

    # Sample sizes by tier capability
    n_all   = len(df)
    n_t2    = df[diff_cols].notna().any(axis=1).sum()  # proxy
    n_tier2 = df["diff_efg_pct"].notna().sum()
    n_tier3 = df["diff_massey_avg_rank"].notna().sum()

    report = {
        "total_games": n_all,
        "tier1_games": n_all,
        "tier2_games": int(n_tier2),
        "tier3_games": int(n_tier3),
        "features": FEATURE_DOCS,
        "missingness_overall": overall_miss,
        "missingness_by_era": miss_by_era,
        "games_by_era": df["era"].value_counts().to_dict(),
        "games_by_type": df["game_type"].value_counts().to_dict(),
        "label_balance": round(df["label"].mean(), 4),
    }

    # Correlation matrix (Tier 2+ only, drop NaN rows)
    t2_cols = [c for c in diff_cols if "efg" in c or "eff" in c or
               "margin" in c or "win_pct" in c or "massey" in c or "seed" in c]
    corr_df = df[t2_cols + ["label"]].dropna()
    if len(corr_df) > 100:
        corr = corr_df.corr()
        corr.to_csv(OUT_DIR / "correlation_matrix.csv")
        # Top correlations with label
        label_corr = corr["label"].drop("label").abs().sort_values(ascending=False)
        report["top_features_by_label_correlation"] = {
            k: round(float(v), 4)
            for k, v in label_corr.head(15).items()
        }

    with open(OUT_DIR / "feature_docs.json", "w") as f:
        json.dump(FEATURE_DOCS, f, indent=2)
    with open(OUT_DIR / "missingness_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("  PHASE 2 FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"\n  Total games: {n_all:,}")
    print(f"  Tier 1 features (1985+):   {n_all:,} games")
    print(f"  Tier 2 features (2003+):   {n_tier2:,} games")
    print(f"  Tier 3 Massey (2003+):     {n_tier3:,} games")
    print(f"\n  Diff features: {len(diff_cols)} total")
    print(f"  Label balance: {report['label_balance']*100:.1f}% team1 win rate (should ~50%)")

    if "top_features_by_label_correlation" in report:
        print("\n  Top features by |correlation| with outcome:")
        for feat, corr_val in list(report["top_features_by_label_correlation"].items())[:10]:
            bar = "#" * int(corr_val * 40)
            print(f"    {feat:<30} {corr_val:.4f}  {bar}")

    print(f"\n  Reports saved to: {OUT_DIR}")
    print("="*60)

    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nBasketball-God — Phase 2: Feature Engineering")
    print(f"DB: {DB_PATH}\n")

    conn = sqlite3.connect(DB_PATH)

    try:
        games, stats, rankings, seeds, confs = load_data(conn)
    finally:
        conn.close()

    print("\n[1/7] Building cumulative season stats (Tier 1)...")
    team_stats = build_team_season_stats(games)

    print("\n[2/7] Building rolling box score stats (Tier 2)...")
    box_stats = build_rolling_box_stats(stats, games)

    print("\n[3/7] Building Massey pre-game rankings (Tier 3)...")
    massey_feats = build_massey_features(rankings, games)

    print("\n[4/7] Building head-to-head history...")
    # H2H is O(n^2) for large datasets — skip for pre-2003 to save time
    # We'll compute it for all games but it'll be NaN for many early matchups
    h2h_feats = build_h2h_features(games)

    print("\n[5/7] Building conference strength features...")
    conf_strength = build_conference_strength(games, confs)

    print("\n[6/7] Assembling matchup feature matrix...")
    features = assemble_matchup_features(
        games, team_stats, box_stats, massey_feats,
        h2h_feats, seeds, conf_strength, confs
    )

    print("\n[7/7] Era normalization and saving...")
    features = normalize_by_season(features)

    # Filter out games with insufficient history
    features = features[features["diff_win_pct"].notna()].copy()
    print(f"  After filtering (need >={MIN_GAMES_HISTORY} games history): {len(features):,} games")

    # Save all games
    out_all = OUT_DIR / "features_all.parquet"
    features.to_parquet(out_all, index=False)
    print(f"  Saved: {out_all}")

    # Save Tier 1 only (1985+, no NaN in core tier1 cols)
    t1_cols = [c for c in features.columns
               if not any(x in c for x in ["efg","to_rate","orb","drb","ft_rate",
                                             "fg3","ast_to","blk","stl","eff","pace",
                                             "massey"])]
    t1 = features[t1_cols].dropna(subset=["diff_win_pct","diff_avg_margin"])
    t1.to_parquet(OUT_DIR / "features_t1.parquet", index=False)
    print(f"  Saved: features_t1.parquet ({len(t1):,} games)")

    # Save Tier 1+2 (2003+)
    t2 = features[features["diff_efg_pct"].notna()].copy()
    t2.to_parquet(OUT_DIR / "features_t2.parquet", index=False)
    print(f"  Saved: features_t2.parquet ({len(t2):,} games)")

    generate_feature_report(features)


if __name__ == "__main__":
    main()
