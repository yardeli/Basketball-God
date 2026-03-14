"""
test_pl.py — Compute and print P&L for a specific date without starting the server.

Usage:
    python test_pl.py              # uses yesterday
    python test_pl.py 20260312     # uses specified date
"""
import sys
from datetime import date, timedelta
from pathlib import Path
import json

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from run_dashboard import (
    OUTPUTS_DIR, STAKE,
    fetch_scores,
    compute_prev_day_from_log,
    compute_prev_day_from_predictions,
)


def run(target: date):
    date_str  = target.strftime("%Y%m%d")
    log_path  = OUTPUTS_DIR / f"bets_log_{date_str}.json"
    pred_path = OUTPUTS_DIR / f"predictions_{date_str}.json"

    print(f"\n{'='*64}")
    print(f"  P&L Report — {target.strftime('%B %d, %Y')}")
    print(f"{'='*64}\n")

    if not log_path.exists() and not pred_path.exists():
        print(f"  No bets_log or predictions found for {date_str}. Exiting.")
        return

    print(f"  Fetching ESPN scores for {date_str}...", end=" ", flush=True)
    scores = fetch_scores(date_str)
    print(f"found {len(scores)} completed games.\n")

    summary = None
    if log_path.exists():
        print(f"  Source: bets_log_{date_str}.json (real market odds)\n")
        log     = json.loads(log_path.read_text())
        summary = compute_prev_day_from_log(log, scores)
    if summary is None and pred_path.exists():
        print(f"  Source: predictions_{date_str}.json (model-derived odds)\n")
        summary = compute_prev_day_from_predictions(target, scores)

    if summary is None:
        print("  No settled trades found yet — games may still be in progress.")
        return

    wins  = summary["wins"]
    loss  = summary["losses"]
    pend  = summary["pending"]
    pnl   = summary["total_pnl"]
    roi   = summary["roi_pct"]
    stake = summary["stake_per_bet"]

    print(f"  Result:  {wins}W / {loss}L  ({pend} pending)")
    print(f"  P&L:     ${pnl:+,.2f}")
    print(f"  ROI:     {roi:+.1f}%")
    print(f"  Stake:   ${stake}/bet\n")

    col_w = [28, 14, 8, 8, 8, 7, 7]
    header = f"  {'Matchup':<{col_w[0]}} {'Bet':<{col_w[1]}} {'Odds':>{col_w[2]}} {'ToWin':>{col_w[3]}} {'Score':<{col_w[4]}} {'W/L':>{col_w[5]}} {'P&L':>{col_w[6]}}"
    print(header)
    print("  " + "-" * (sum(col_w) + len(col_w)))

    for t in summary["trades"]:
        result = "WIN" if t["won"] else "LOSS"
        desc   = t.get("description") or t.get("pick", "")
        label  = t["matchup"][:26] if len(t["matchup"]) > 26 else t["matchup"]
        desc_s = desc[:13] if len(desc) > 13 else desc
        print(
            f"  {label:<{col_w[0]}} {desc_s:<{col_w[1]}} "
            f"{t['odds']:>{col_w[2]}} {t['to_win']:>{col_w[3]}.2f} "
            f"{t['score']:<{col_w[4]}} {result:>{col_w[5]}} "
            f"${t['pnl']:>+{col_w[6]-1}.2f}"
        )

    if summary.get("pending_trades"):
        print(f"\n  Pending ({pend} bets — ESPN scores not yet available):")
        for t in summary["pending_trades"]:
            desc = t.get("description", t.get("pick", ""))
            print(f"    {t['matchup']}  |  {desc}  |  {t['odds']}")

    print()
    out_path = OUTPUTS_DIR / f"paper_bets_{date_str}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved -> outputs/paper_bets_{date_str}.json\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ds = sys.argv[1]
        target = date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
    else:
        target = date.today() - timedelta(days=1)

    run(target)
