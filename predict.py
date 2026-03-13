"""
Predict — CLI and API for predicting any D1 men's basketball game.

Usage:
  python predict.py --today              # Predict all games today
  python predict.py --team1 "Duke" --team2 "North Carolina"  # Head-to-head
  python predict.py --date 20260315      # Predict games on a specific date
  python predict.py --pipeline           # Run full pipeline (scrape → train → predict)
  python predict.py --backtest           # Walk-forward backtest across seasons
"""
import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from data_scraper import scrape_all_teams, scrape_all_seasons, fetch_today_games
from elo import EloSystem, build_elo_ratings
from feature_engineering import compute_rolling_stats, create_matchup_features, get_feature_matrix
from text_pipeline import get_text_signals, load_cached_signals
from model_training import NCAAModel, train_walk_forward, train_production_model


def load_or_scrape_data(force_scrape: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached data or scrape fresh."""
    all_games_path = config.PROCESSED_DIR / "all_games.csv"
    teams_path = config.RAW_DIR / "teams.csv"

    if all_games_path.exists() and teams_path.exists() and not force_scrape:
        print("[Pipeline] Loading cached data...")
        games = pd.read_csv(all_games_path)
        teams = pd.read_csv(teams_path)
        print(f"  {len(games)} games, {len(teams)} teams loaded from cache")
        return teams, games

    print("[Pipeline] Scraping fresh data...")
    teams = scrape_all_teams()
    games = scrape_all_seasons()
    return teams, games


def run_full_pipeline(force_scrape: bool = False) -> NCAAModel:
    """
    Full pipeline: scrape → elo → features → train → save.
    """
    print("\n" + "=" * 60)
    print("  NCAA GENERATOR v2.0 — FULL PIPELINE")
    print("=" * 60)

    # 1. Data
    teams_df, games_df = load_or_scrape_data(force_scrape)

    # 2. Elo ratings
    elo_system, games_with_elo = build_elo_ratings(games_df)

    # 3. Rolling stats
    games_with_stats = compute_rolling_stats(games_with_elo)

    # 4. Text signals (current only — historical games won't have text)
    text_signals = load_cached_signals()
    if text_signals is None:
        try:
            text_signals = get_text_signals(teams_df)
        except Exception as e:
            print(f"  [WARN] Text signals failed: {e}")
            text_signals = None

    # 5. Matchup features
    matchups = create_matchup_features(games_with_stats, text_signals)

    # Save processed data
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matchups.to_csv(config.PROCESSED_DIR / "matchup_features.csv", index=False)

    # 6. Train production model
    model = train_production_model(matchups)

    # 7. Walk-forward backtest
    wf_results = train_walk_forward(matchups)
    with open(config.OUTPUTS_DIR / "walk_forward_results.json", "w") as f:
        json.dump(wf_results, f, indent=2, default=str)

    print("\n[Pipeline] Done!")
    return model


def predict_today(model: NCAAModel = None, elo_system: EloSystem = None):
    """Predict all of today's D1 games."""
    print("\n" + "=" * 60)
    print(f"  TODAY'S PREDICTIONS — {datetime.now().strftime('%B %d, %Y')}")
    print("=" * 60)

    # Load model if not provided
    if model is None:
        model = NCAAModel()
        try:
            model.load()
        except Exception:
            print("  [ERROR] No trained model found. Run --pipeline first.")
            return []

    # Load Elo ratings
    if elo_system is None:
        all_games_path = config.PROCESSED_DIR / "all_games.csv"
        if all_games_path.exists():
            games = pd.read_csv(all_games_path)
            elo_system, _ = build_elo_ratings(games)
        else:
            elo_system = EloSystem()

    # Fetch today's games
    today_games = fetch_today_games()
    if not today_games:
        print("  No games scheduled today.")
        return []

    # Get text signals
    text_signals = load_cached_signals()
    if text_signals is None:
        try:
            text_signals = get_text_signals()
        except Exception:
            text_signals = {"injuries": {}, "sentiment": {}}

    predictions = []
    for game in today_games:
        home_id = game["home_id"]
        away_id = game["away_id"]
        home_team = game["home_team"]
        away_team = game["away_team"]

        # Build feature row
        home_elo = elo_system.get_rating(home_id)
        away_elo = elo_system.get_rating(away_id)

        feature_row = {
            "elo_diff": home_elo - away_elo,
            "net_rating_diff": 0,  # Would need rolling stats
            "win_pct_diff": 0,
            "points_pg_diff": 0,
            "opp_points_pg_diff": 0,
            "momentum_diff": 0,
            "home_away": 0 if game.get("neutral_site", 0) else 1,
            "rest_days_diff": 0,
            "conf_game": 0,
            "power_conf_diff": 0,
            "injury_impact_diff": (
                text_signals.get("injuries", {}).get(home_team, 0) -
                text_signals.get("injuries", {}).get(away_team, 0)
            ),
            "sentiment_diff": (
                text_signals.get("sentiment", {}).get(home_team, 0) -
                text_signals.get("sentiment", {}).get(away_team, 0)
            ),
        }

        # Ensure all expected features exist
        for feat in config.MATCHUP_FEATURES:
            if feat not in feature_row:
                feature_row[feat] = 0

        X = pd.DataFrame([feature_row])[model.feature_names]
        proba = model.predict_proba(X)[0]

        pred = {
            "game_id": game["game_id"],
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": round(home_elo, 1),
            "away_elo": round(away_elo, 1),
            "home_win_prob": round(float(proba), 3),
            "away_win_prob": round(1.0 - float(proba), 3),
            "predicted_winner": home_team if proba > 0.5 else away_team,
            "confidence": round(float(max(proba, 1 - proba)), 3),
            "status": game.get("status", ""),
        }
        predictions.append(pred)

    # Sort by confidence
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    # Print
    print(f"\n  {'AWAY':<28} {'HOME':<28} {'PICK':<20} {'CONF':>6}")
    print("  " + "-" * 86)
    for p in predictions:
        marker = "*" if p["predicted_winner"] == p["home_team"] else " "
        away_marker = "*" if p["predicted_winner"] == p["away_team"] else " "
        print(f"  {away_marker}{p['away_team']:<27} {marker}{p['home_team']:<27} "
              f"{p['predicted_winner']:<20} {p['confidence']:>5.1%}")

    print(f"\n  {len(predictions)} games predicted")

    # Save
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.OUTPUTS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"  Saved to {output_path}")

    return predictions


def predict_matchup(team1: str, team2: str, neutral: bool = False,
                    model: NCAAModel = None, elo_system: EloSystem = None) -> dict:
    """Predict a specific head-to-head matchup."""
    if model is None:
        model = NCAAModel()
        model.load()

    if elo_system is None:
        all_games_path = config.PROCESSED_DIR / "all_games.csv"
        if all_games_path.exists():
            games = pd.read_csv(all_games_path)
            elo_system, _ = build_elo_ratings(games)
        else:
            elo_system = EloSystem()

    # Load teams to find IDs
    teams_path = config.RAW_DIR / "teams.csv"
    if teams_path.exists():
        teams = pd.read_csv(teams_path)
        t1 = teams[teams["name"].str.contains(team1, case=False, na=False)]
        t2 = teams[teams["name"].str.contains(team2, case=False, na=False)]

        t1_id = int(t1.iloc[0]["espn_id"]) if len(t1) > 0 else 0
        t2_id = int(t2.iloc[0]["espn_id"]) if len(t2) > 0 else 0
        t1_name = t1.iloc[0]["name"] if len(t1) > 0 else team1
        t2_name = t2.iloc[0]["name"] if len(t2) > 0 else team2
    else:
        t1_id, t2_id = 0, 0
        t1_name, t2_name = team1, team2

    home_elo = elo_system.get_rating(t1_id)
    away_elo = elo_system.get_rating(t2_id)

    feature_row = {feat: 0 for feat in config.MATCHUP_FEATURES}
    feature_row["elo_diff"] = home_elo - away_elo
    feature_row["home_away"] = 0 if neutral else 1

    X = pd.DataFrame([feature_row])[model.feature_names]
    proba = model.predict_proba(X)[0]

    result = {
        "home_team": t1_name,
        "away_team": t2_name,
        "home_elo": round(home_elo, 1),
        "away_elo": round(away_elo, 1),
        "home_win_prob": round(float(proba), 3),
        "away_win_prob": round(1.0 - float(proba), 3),
        "predicted_winner": t1_name if proba > 0.5 else t2_name,
        "confidence": round(float(max(proba, 1 - proba)), 3),
        "neutral_site": neutral,
    }

    print(f"\n  {t2_name} @ {t1_name}")
    print(f"  Elo: {t1_name} {home_elo:.0f} vs {t2_name} {away_elo:.0f}")
    print(f"  Prediction: {result['predicted_winner']} ({result['confidence']:.1%} confidence)")
    print(f"  Win probabilities: {t1_name} {result['home_win_prob']:.1%} / "
          f"{t2_name} {result['away_win_prob']:.1%}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCAA Generator v2.0 — Predict any D1 game")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run full pipeline: scrape → train → predict")
    parser.add_argument("--scrape", action="store_true",
                        help="Force fresh data scrape (used with --pipeline)")
    parser.add_argument("--backtest", action="store_true",
                        help="Run walk-forward backtest")
    parser.add_argument("--today", action="store_true",
                        help="Predict today's games")
    parser.add_argument("--team1", type=str, help="Home team name (for head-to-head)")
    parser.add_argument("--team2", type=str, help="Away team name (for head-to-head)")
    parser.add_argument("--neutral", action="store_true",
                        help="Neutral site game (for head-to-head)")
    parser.add_argument("--date", type=str,
                        help="Predict games on a specific date (YYYYMMDD)")

    args = parser.parse_args()

    if args.pipeline:
        run_full_pipeline(force_scrape=args.scrape)

    elif args.backtest:
        # Load processed matchups or run pipeline first
        matchups_path = config.PROCESSED_DIR / "matchup_features.csv"
        if matchups_path.exists():
            matchups = pd.read_csv(matchups_path)
            results = train_walk_forward(matchups)
            config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(config.OUTPUTS_DIR / "walk_forward_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
        else:
            print("No processed data found. Run --pipeline first.")

    elif args.today:
        predict_today()

    elif args.team1 and args.team2:
        predict_matchup(args.team1, args.team2, neutral=args.neutral)

    else:
        parser.print_help()
        print("\nExamples:")
        print('  python predict.py --pipeline              # Full pipeline')
        print('  python predict.py --today                 # Today\'s predictions')
        print('  python predict.py --team1 "Duke" --team2 "UNC"  # Head-to-head')
        print('  python predict.py --backtest              # Walk-forward test')
