"""
Basketball-God — Web Dashboard Server
Run:  python web/server.py
Then open: http://localhost:5050
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, send_file, request as flask_request

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__,
            template_folder=str(ROOT / "web" / "templates"),
            static_folder=str(ROOT / "web" / "static"))

# ── Lazy-loaded daily predictor ───────────────────────────────────────────────
_daily_predictor = None

def get_daily_predictor():
    global _daily_predictor
    if _daily_predictor is None:
        try:
            from daily_predictor import DailyPredictor
            _daily_predictor = DailyPredictor().load()
        except Exception as e:
            print(f"[server] Warning: could not load daily predictor: {e}", file=sys.stderr)
    return _daily_predictor


def _load_json(path: Path) -> dict | list:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@app.route("/")
def index():
    # Use send_file so Jinja2 doesn't mangle JS curly braces
    return send_file(str(ROOT / "web" / "templates" / "index.html"))


@app.route("/api/data")
def api_data():
    """Aggregate all pre-computed results into one payload."""
    p4 = ROOT / "phase4_tournament" / "output"
    p5 = ROOT / "phase5_deploy" / "output"

    backtest      = _load_json(p4 / "bracket_backtest.json")
    sims_raw      = _load_json(p4 / "bracket_simulations.json")
    calibration   = _load_json(p4 / "round_calibration.json")
    seed_matchups = _load_json(p4 / "seed_matchup_stats.json")
    bootstrap     = _load_json(p5 / "bootstrap_ci.json")
    worst_case    = _load_json(p5 / "worst_case.json")
    cal_audit     = _load_json(p5 / "calibration_audit.json")
    importance    = _load_json(p5 / "feature_importance.json")
    report        = _load_json(p5 / "final_report.json")

    champ_by_season = {}
    if isinstance(sims_raw, list):
        for s in sims_raw:
            champ_by_season[str(s["season"])] = s.get("champion_probabilities", [])

    return jsonify({
        "backtest":          backtest,
        "champion_probs":    champ_by_season,
        "seed_matchups":     seed_matchups,
        "bootstrap":         bootstrap,
        "worst_case":        worst_case,
        "calibration":       cal_audit,
        "importance":        importance,
        "report":            report,
        "round_calibration": calibration,
    })


@app.route("/api/odds")
def api_odds():
    """Live championship futures from The Odds API (cached 1 hour)."""
    try:
        from odds_fetcher import fetch_championship_odds
        data = fetch_championship_odds(force_refresh=False)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "teams": []})


@app.route("/api/daily")
def api_daily():
    """Today's NCAAB games with model predictions and betting edge calculations."""
    try:
        from odds_fetcher import fetch_todays_games
        from daily_predictor import calculate_game_edges

        force = flask_request.args.get("refresh") == "1"
        odds_data = fetch_todays_games(force_refresh=force)

        if odds_data.get("error") and not odds_data.get("games"):
            return jsonify({
                "games": [], "error": odds_data["error"],
                "date": odds_data.get("date", ""), "n_games": 0,
                "cached": odds_data.get("cached", False),
                "requests_remaining": odds_data.get("requests_remaining", -1),
            })

        predictor = get_daily_predictor()
        now_utc   = datetime.now(timezone.utc)

        enriched_games = []
        for game in odds_data.get("games", []):
            home = game["home_team"]
            away = game["away_team"]

            # Determine if game has already started
            in_progress = False
            try:
                ct = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
                in_progress = ct <= now_utc
            except Exception:
                pass

            if in_progress:
                # Show the game but strip edges — live odds are meaningless to a pre-game model
                enriched_games.append({
                    "id":            game["id"],
                    "home_team":     home,
                    "away_team":     away,
                    "home_display":  home,
                    "away_display":  away,
                    "commence_time": game["commence_time"],
                    "in_progress":   True,
                    "prediction":    {"prob_home_wins": 0.5, "prob_away_wins": 0.5,
                                      "confidence": "low", "model_data_available": False},
                    "bets":          [],
                    "best_edge":     0,
                    "n_value_bets":  0,
                })
                continue

            if predictor and predictor.ready:
                prediction = predictor.predict_game(home, away)
            else:
                prediction = {
                    "home_team": home, "away_team": away,
                    "home_display": home, "away_display": away,
                    "home_id": None, "away_id": None,
                    "prob_home_wins": 0.50, "prob_away_wins": 0.50,
                    "model_spread": 0.0, "predicted_total": None,
                    "confidence": "low", "model_data_available": False,
                }

            bets = calculate_game_edges(prediction, game)

            enriched_games.append({
                "id":            game["id"],
                "home_team":     home,
                "away_team":     away,
                "home_display":  prediction["home_display"],
                "away_display":  prediction["away_display"],
                "commence_time": game["commence_time"],
                "in_progress":   False,
                "prediction": {
                    "prob_home_wins":  prediction["prob_home_wins"],
                    "prob_away_wins":  prediction["prob_away_wins"],
                    "model_spread":    prediction["model_spread"],
                    "predicted_total": prediction["predicted_total"],
                    "confidence":      prediction["confidence"],
                    "data_available":  prediction["model_data_available"],
                    "model_type":      prediction.get("model_type", "unknown"),
                    "home_n_games":    prediction.get("home_n_games"),
                    "away_n_games":    prediction.get("away_n_games"),
                },
                "bets":         bets,
                "best_edge":    bets[0]["edge"] if bets else 0,
                "n_value_bets": sum(1 for b in bets if b["edge"] >= 0.04),
            })

        # Upcoming games first (sorted by value), then in-progress at the end
        upcoming   = [g for g in enriched_games if not g["in_progress"]]
        live_games = [g for g in enriched_games if g["in_progress"]]
        upcoming.sort(key=lambda g: (-g["n_value_bets"], -g["best_edge"]))
        enriched_games = upcoming + live_games

        return jsonify({
            "games":              enriched_games,
            "date":               odds_data.get("date", ""),
            "n_games":            len(enriched_games),
            "requests_remaining": odds_data.get("requests_remaining", -1),
            "cached":             odds_data.get("cached", False),
            "error":              odds_data.get("error"),
            "model_ready":        predictor.ready if predictor else False,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "games": [], "error": str(e), "date": "",
            "n_games": 0, "cached": False,
            "requests_remaining": -1, "model_ready": False,
        })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Force-refresh the season stats store (fetch latest game results from ESPN)."""
    global _daily_predictor
    try:
        predictor = get_daily_predictor()
        if predictor and predictor.rs_store:
            predictor.rs_store.refresh(force=True)
            return jsonify({
                "ok": True,
                "games": len(predictor.rs_store.games),
                "teams_ranked": len(predictor.rs_store.elo_ranks),
            })
        return jsonify({"ok": False, "error": "Regular season model not loaded"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/model_status")
def api_model_status():
    """Return status of both models."""
    predictor = get_daily_predictor()
    return jsonify({
        "tournament_model": bool(predictor and predictor.model),
        "regular_season_model": bool(predictor and predictor.rs_model and predictor.rs_model.ready),
        "season_games_loaded": len(predictor.rs_store.games) if (predictor and predictor.rs_store) else 0,
        "teams_ranked": len(predictor.rs_store.elo_ranks) if (predictor and predictor.rs_store) else 0,
        "summaries_cached": sum(1 for g in predictor.rs_store.games if g.get("has_summary"))
                            if (predictor and predictor.rs_store) else 0,
    })


@app.route("/api/paperbets")
def api_paperbets():
    """Load the latest paper bets P&L data from outputs directory."""
    try:
        outputs_dir = ROOT / "outputs"
        paper_bet_files = sorted(outputs_dir.glob("paper_bets_*.json"), reverse=True)
        if paper_bet_files:
            data = json.loads(paper_bet_files[0].read_text(encoding="utf-8"))
            return jsonify(data)
        else:
            return jsonify({
                "error": "No paper bets data found yet",
                "message": "Run python run_dashboard.py to generate today's P&L"
            })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("\n  Basketball-God Web Dashboard")
    print("  Open: http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
