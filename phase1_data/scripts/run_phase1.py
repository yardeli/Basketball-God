"""
Phase 1 Master Pipeline — Orchestrate all data ingestion and produce summary report.

Usage:
  python run_phase1.py                  # Run everything
  python run_phase1.py --espn-only      # Only ESPN data
  python run_phase1.py --report-only    # Just regenerate the summary report
  python run_phase1.py --espn-seasons 2020-2026  # Specific ESPN season range
"""
import sys
import os
import json
import argparse
import sqlite3
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from phase1_data.schema import init_database
from phase1_data.team_normalization import TeamNormalizer
from phase1_data.scripts.ingest_espn import ingest_all_espn
from phase1_data.scripts.ingest_kaggle import ingest_all_kaggle


def generate_summary_report(conn: sqlite3.Connection) -> dict:
    """Generate a comprehensive summary of the unified database."""
    print("\n" + "=" * 60)
    print("  PHASE 1 SUMMARY REPORT")
    print("=" * 60)

    report = {}

    # Total games
    cursor = conn.execute("SELECT COUNT(*) FROM games")
    total_games = cursor.fetchone()[0]
    report["total_games"] = total_games
    print(f"\n  Total games in database: {total_games:,}")

    # Games by source
    cursor = conn.execute("""
        SELECT data_source, COUNT(*) as cnt
        FROM games GROUP BY data_source ORDER BY cnt DESC
    """)
    by_source = {row[0]: row[1] for row in cursor.fetchall()}
    report["by_source"] = by_source
    print(f"\n  Games by source:")
    for src, cnt in by_source.items():
        print(f"    {src}: {cnt:,}")

    # Games by decade
    cursor = conn.execute("""
        SELECT (season / 10) * 10 as decade, COUNT(*) as cnt
        FROM games GROUP BY decade ORDER BY decade
    """)
    by_decade = {f"{row[0]}s": row[1] for row in cursor.fetchall()}
    report["by_decade"] = by_decade
    print(f"\n  Games by decade:")
    for decade, cnt in by_decade.items():
        bar = "█" * (cnt // 2000)
        print(f"    {decade}: {cnt:>7,} {bar}")

    # Games by season
    cursor = conn.execute("""
        SELECT season, COUNT(*) as cnt
        FROM games GROUP BY season ORDER BY season
    """)
    by_season = {row[0]: row[1] for row in cursor.fetchall()}
    report["by_season"] = by_season
    min_season = min(by_season.keys()) if by_season else 0
    max_season = max(by_season.keys()) if by_season else 0
    print(f"\n  Season range: {min_season} to {max_season} ({len(by_season)} seasons)")

    # Data completeness
    cursor = conn.execute("""
        SELECT data_completeness_tier, COUNT(*) as cnt
        FROM games GROUP BY data_completeness_tier ORDER BY data_completeness_tier
    """)
    by_tier = {f"tier_{row[0]}": row[1] for row in cursor.fetchall()}
    report["data_completeness"] = by_tier
    print(f"\n  Data completeness:")
    tier_labels = {"tier_1": "Full box score", "tier_2": "Basic stats", "tier_3": "Score only"}
    for tier, cnt in by_tier.items():
        pct = cnt / total_games * 100 if total_games > 0 else 0
        print(f"    {tier_labels.get(tier, tier)}: {cnt:>7,} ({pct:.1f}%)")

    # Games by type
    cursor = conn.execute("""
        SELECT game_type, COUNT(*) as cnt
        FROM games GROUP BY game_type ORDER BY cnt DESC
    """)
    by_type = {row[0]: row[1] for row in cursor.fetchall()}
    report["by_game_type"] = by_type
    print(f"\n  Games by type:")
    for gtype, cnt in by_type.items():
        print(f"    {gtype}: {cnt:,}")

    # Era distribution
    cursor = conn.execute("""
        SELECT era, COUNT(*) as cnt
        FROM games GROUP BY era ORDER BY MIN(season)
    """)
    by_era = {row[0]: row[1] for row in cursor.fetchall()}
    report["by_era"] = by_era
    print(f"\n  Games by era:")
    for era, cnt in by_era.items():
        print(f"    {era}: {cnt:,}")

    # Teams
    cursor = conn.execute("SELECT COUNT(*) FROM teams")
    n_teams = cursor.fetchone()[0]
    report["total_teams"] = n_teams

    cursor = conn.execute("SELECT COUNT(*) FROM team_aliases")
    n_aliases = cursor.fetchone()[0]
    report["total_aliases"] = n_aliases
    print(f"\n  Teams: {n_teams} (with {n_aliases} aliases)")

    # Home win rate by era (sanity check)
    cursor = conn.execute("""
        SELECT era, AVG(home_win) as hw_rate, COUNT(*) as cnt
        FROM games
        WHERE neutral_site = 0
        GROUP BY era ORDER BY MIN(season)
    """)
    hw_by_era = {}
    print(f"\n  Home win rate by era (non-neutral, sanity check):")
    for row in cursor.fetchall():
        era, rate, cnt = row
        hw_by_era[era] = {"rate": round(float(rate), 4), "games": cnt}
        print(f"    {era}: {rate:.1%} ({cnt:,} games)")
    report["home_win_by_era"] = hw_by_era

    # COVID year check
    cursor = conn.execute("""
        SELECT COUNT(*) FROM games WHERE season = 2021
    """)
    covid_games = cursor.fetchone()[0]
    report["covid_2021_games"] = covid_games
    print(f"\n  COVID year (2020-21): {covid_games} games")

    # Missing data flags
    anomalies = []
    # Check for seasons with suspiciously few games
    for season, cnt in by_season.items():
        if season >= 1985 and cnt < 1000 and season != 2021:
            anomalies.append(f"Season {season}: only {cnt} games (expected ~5000+)")
    if anomalies:
        report["anomalies"] = anomalies
        print(f"\n  Anomalies detected:")
        for a in anomalies:
            print(f"    ⚠ {a}")

    # Data sources
    cursor = conn.execute("SELECT * FROM data_sources")
    sources = []
    for row in cursor.fetchall():
        sources.append({
            "name": row[0],
            "games": row[1],
            "seasons": row[2],
            "updated": row[3],
            "notes": row[4],
        })
    report["data_sources"] = sources

    return report


def run_phase1(espn_seasons: list[int] = None, espn_only: bool = False,
               report_only: bool = False, skip_sports_ref: bool = True):
    """Execute the full Phase 1 pipeline."""
    print("\n" + "=" * 60)
    print("  BASKETBALL GOD — PHASE 1: DATA ACQUISITION")
    print("=" * 60)
    print("  Building the most comprehensive NCAA D1 game database")
    print("  Sources: ESPN API + Kaggle + Sports Reference")

    db_path = str(Path(__file__).parent.parent / "output" / "basketball_god.db")
    conn = init_database(db_path)
    normalizer = TeamNormalizer()

    # Load existing ESPN teams for name resolution
    teams_csv = Path(__file__).parent.parent.parent / "data" / "raw" / "teams.csv"
    if teams_csv.exists():
        normalizer.load_espn_teams(str(teams_csv))

    if not report_only:
        # ── ESPN (2003-2026) ──
        if espn_seasons is None:
            espn_seasons = list(range(2003, 2027))
        ingest_all_espn(conn, normalizer, espn_seasons)

        # ── Kaggle (1985-2024) ──
        if not espn_only:
            ingest_all_kaggle(conn, normalizer)

        # ── Sports Reference (optional, slow) ──
        if not espn_only and not skip_sports_ref:
            from phase1_data.scripts.ingest_sports_ref import ingest_sports_ref_stats
            ingest_sports_ref_stats(conn, normalizer, list(range(2015, 2026)))

        # Save team normalization state
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        normalizer.save(str(output_dir / "team_names.json"))

        print(f"\n{normalizer.get_unresolved_report()}")

    # ── Summary Report ──
    report = generate_summary_report(conn)

    # Save report
    output_dir = Path(__file__).parent.parent / "output"
    report_path = output_dir / "phase1_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to: {report_path}")

    # Log experiment
    conn.execute("""
        INSERT INTO experiment_log (phase, description, metrics)
        VALUES ('phase1', 'Data acquisition pipeline run',  ?)
    """, (json.dumps({
        "total_games": report["total_games"],
        "sources": report.get("by_source", {}),
        "seasons": len(report.get("by_season", {})),
    }, default=str),))
    conn.commit()

    conn.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Data Acquisition")
    parser.add_argument("--espn-only", action="store_true",
                        help="Only ingest ESPN data (fastest)")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip ingestion, just generate report")
    parser.add_argument("--espn-seasons", type=str, default=None,
                        help="ESPN season range (e.g., '2020-2026')")
    parser.add_argument("--include-sports-ref", action="store_true",
                        help="Include Sports Reference scraping (slow)")
    args = parser.parse_args()

    espn_seasons = None
    if args.espn_seasons:
        start, end = map(int, args.espn_seasons.split("-"))
        espn_seasons = list(range(start, end + 1))

    run_phase1(
        espn_seasons=espn_seasons,
        espn_only=args.espn_only,
        report_only=args.report_only,
        skip_sports_ref=not args.include_sports_ref,
    )
