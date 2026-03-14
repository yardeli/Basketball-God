"""
warmup_cache.py — Pre-warm the season stats cache for Basketball-God 2.0

Run this ONCE after initial setup to fetch all game summaries in the background.
After this completes, the server starts instantly with full box-score features.

Usage:
    python warmup_cache.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")

from season_stats_store import (
    SeasonStatsStore, CACHE_DIR, SUMMARY_MAX_FETCH,
    _get, _fetch_game_summary, _merge_summary_stats,
    REQUEST_DELAY
)
import json, time
from pathlib import Path

def main():
    print("Basketball-God 2.0 — Cache Warm-Up")
    print("This fetches full box-score summaries for all season games.")
    print("Run once; subsequent server starts are instant.\n")

    store = SeasonStatsStore()

    # Load the existing cache
    cache_path = CACHE_DIR / f"season_games_{store.season}.json"
    if not cache_path.exists():
        print("No game cache found. Running full refresh first...")
        store.refresh()
        return

    cached = json.loads(cache_path.read_text())
    games  = cached.get("games", [])
    game_by_id = {g["game_id"]: g for g in games}

    needs = [g for g in games if not g.get("has_summary")]
    # Most recent first
    needs.sort(key=lambda g: g["date"], reverse=True)

    total = len(needs)
    print(f"Games needing summaries: {total}")
    print(f"Estimated time: ~{total * 0.4 / 60:.1f} minutes\n")

    fetched = 0
    for i, g in enumerate(needs):
        summary = _fetch_game_summary(g["game_id"])
        if summary:
            _merge_summary_stats(g, summary)
            fetched += 1
        if (i + 1) % 50 == 0:
            # Save progress every 50 games
            cached["games"] = games
            cache_path.write_text(json.dumps(cached))
            pct = (i + 1) / total * 100
            print(f"  Progress: {i+1}/{total} ({pct:.0f}%) — {fetched} summaries fetched")
        time.sleep(REQUEST_DELAY)

    cached["games"] = games
    cache_path.write_text(json.dumps(cached))
    print(f"\nDone. {fetched}/{total} summaries cached.")
    print("Server will now start with full box-score features on every game.")

if __name__ == "__main__":
    main()
