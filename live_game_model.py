"""
live_game_model.py — In-game win probability and hedge calculations
===================================================================
Adjusts the pre-game model probability in real time based on:
  - Current score differential
  - Time remaining in the game
  - Pre-game model spread as a prior

Formula (standard sports analytics approach):
  At time t into a 40-minute game:
    remaining_uncertainty = σ_pregame × √(minutes_remaining / 40)
    expected_final_margin = current_margin + pregame_spread × (minutes_remaining / 40)
    live_win_prob = Φ(expected_final_margin / remaining_uncertainty)

σ_pregame = 11.0 points (NCAAB standard deviation of final margin)

Hedge calculator:
  Given a pre-game bet on team A, compute the optimal live hedge on team B
  to lock in guaranteed profit or minimize loss.
"""

import math
from typing import Optional

SIGMA_PREGAME   = 11.0   # NCAAB standard deviation of final margin
TOTAL_MINUTES   = 40.0   # College basketball regulation


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def parse_clock(period: int, clock_str: str) -> float:
    """
    Convert ESPN period + clock string to minutes remaining in the game.
    Period 1 = first half, Period 2 = second half.
    OT periods add 5 minutes each.

    Returns minutes remaining (float).
    """
    try:
        parts = clock_str.replace(",", "").strip().split(":")
        mins  = float(parts[0])
        secs  = float(parts[1]) if len(parts) > 1 else 0.0
        clock_mins = mins + secs / 60.0
    except (ValueError, IndexError):
        clock_mins = 0.0

    if period == 1:
        # First half: 20 min remaining in half + full second half
        return clock_mins + 20.0
    elif period == 2:
        # Second half: clock_mins remaining
        return max(clock_mins, 0.0)
    else:
        # OT: each period is 5 minutes
        return max(clock_mins, 0.0)


def live_win_prob(
    pregame_home_prob: float,
    home_score: int,
    away_score: int,
    minutes_remaining: float,
    sigma: float = SIGMA_PREGAME,
) -> dict:
    """
    Compute live home win probability given current game state.

    pregame_home_prob : Pre-game model probability for home team (0-1)
    home_score        : Current home score
    away_score        : Current away score
    minutes_remaining : Minutes left in regulation (0 = final)
    sigma             : Pre-game scoring spread std dev (default 11 for NCAAB)

    Returns dict with:
      live_home_prob, live_away_prob, expected_final_margin,
      remaining_uncertainty, pregame_spread
    """
    # Clamp pre-game prob
    p = max(0.001, min(0.999, pregame_home_prob))

    # Convert pre-game prob → expected final margin (home perspective)
    # Φ(spread/σ) = p  →  spread = σ × Φ⁻¹(p)
    # Approximate inverse normal via rational approximation
    pregame_spread = _inv_normal(p) * sigma

    # Current score differential (positive = home leading)
    current_margin = home_score - away_score

    if minutes_remaining <= 0:
        # Game over — result is determined
        if current_margin > 0:
            return {"live_home_prob": 1.0, "live_away_prob": 0.0,
                    "expected_final_margin": current_margin,
                    "remaining_uncertainty": 0.0, "pregame_spread": round(pregame_spread, 1)}
        elif current_margin < 0:
            return {"live_home_prob": 0.0, "live_away_prob": 1.0,
                    "expected_final_margin": current_margin,
                    "remaining_uncertainty": 0.0, "pregame_spread": round(pregame_spread, 1)}
        else:
            return {"live_home_prob": 0.5, "live_away_prob": 0.5,
                    "expected_final_margin": 0.0,
                    "remaining_uncertainty": 0.0, "pregame_spread": round(pregame_spread, 1)}

    # Remaining uncertainty scales with √(time_remaining / total_time)
    frac_remaining    = min(minutes_remaining / TOTAL_MINUTES, 1.0)
    remaining_sigma   = sigma * math.sqrt(frac_remaining)

    # Expected final margin = current margin + model's expected future scoring
    future_expected   = pregame_spread * frac_remaining
    expected_final    = current_margin + future_expected

    if remaining_sigma < 0.01:
        h_prob = 1.0 if expected_final > 0 else (0.0 if expected_final < 0 else 0.5)
    else:
        h_prob = _normal_cdf(expected_final / remaining_sigma)

    return {
        "live_home_prob":        round(h_prob, 3),
        "live_away_prob":        round(1 - h_prob, 3),
        "expected_final_margin": round(expected_final, 1),
        "remaining_uncertainty": round(remaining_sigma, 2),
        "pregame_spread":        round(pregame_spread, 1),
        "current_margin":        current_margin,
        "minutes_remaining":     round(minutes_remaining, 1),
    }


def _inv_normal(p: float) -> float:
    """Rational approximation of inverse normal CDF (Abramowitz & Stegun)."""
    p = max(0.001, min(0.999, p))
    if p >= 0.5:
        sign = 1
        q = p
    else:
        sign = -1
        q = 1 - p
    t = math.sqrt(-2 * math.log(1 - q))
    z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
        1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
    return sign * z


# ── Hedge Calculator ──────────────────────────────────────────────────────────

def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal (1.0 = no profit, 2.0 = 1:1)."""
    if american > 0:
        return 1 + american / 100
    return 1 + 100 / abs(american)


def decimal_to_american(decimal: float) -> int:
    if decimal >= 2.0:
        return int((decimal - 1) * 100)
    return int(-100 / (decimal - 1))


def compute_hedge(
    original_stake: float,
    original_american_odds: int,
    original_side: str,            # "home" or "away"
    live_home_american: Optional[int],
    live_away_american: Optional[int],
    live_home_prob: float,
    live_away_prob: float,
) -> dict:
    """
    Given an existing pre-game bet, compute hedge recommendations.

    Returns:
      full_hedge_stake    : Stake on opposite side to guarantee profit
      full_hedge_profit   : Guaranteed locked-in profit from full hedge
      partial_hedge_stake : Kelly-optimal partial hedge stake
      live_edge           : Current edge on original bet (model vs live market)
      recommendation      : "HOLD", "PARTIAL HEDGE", "FULL HEDGE", "CUT LOSSES"
    """
    dec_orig = american_to_decimal(original_american_odds)
    original_payout = original_stake * dec_orig   # total return including stake
    original_profit = original_payout - original_stake

    if original_side == "home":
        hedge_american = live_away_american
        hedge_prob     = live_away_prob
        hold_prob      = live_home_prob
    else:
        hedge_american = live_home_american
        hedge_prob     = live_home_prob
        hold_prob      = live_away_prob

    result = {
        "original_stake":    original_stake,
        "original_odds":     original_american_odds,
        "original_profit":   round(original_profit, 2),
        "live_edge":         None,
        "full_hedge_stake":  None,
        "full_hedge_profit": None,
        "partial_hedge_stake": None,
        "recommendation":    "HOLD",
        "explanation":       "",
    }

    if hedge_american is None:
        result["recommendation"] = "HOLD"
        result["explanation"]    = "No live odds available for hedge side"
        return result

    dec_hedge = american_to_decimal(hedge_american)

    # Full hedge: bet H on opposite so that both outcomes break even
    # If original wins: original_payout - original_stake - H
    # If hedge wins:    H × dec_hedge - H - original_stake
    # Set equal: H = (original_payout - original_stake) / (dec_hedge - 1 + 1)
    #            H = original_profit / (dec_hedge - 1)
    if dec_hedge > 1.0:
        full_hedge_stake  = original_profit / (dec_hedge - 1)
        full_hedge_profit = original_profit - full_hedge_stake
    else:
        full_hedge_stake  = 0.0
        full_hedge_profit = 0.0

    # Live edge on original bet
    if hedge_american is not None:
        live_implied_hold = 1 / dec_hedge  # implied prob of hedge side
        live_implied_orig = 1 - live_implied_hold
        live_edge = hold_prob - live_implied_orig
        result["live_edge"] = round(live_edge, 4)

    # Kelly partial hedge
    if dec_hedge > 1.0:
        b = dec_hedge - 1
        kelly_frac = (hedge_prob * b - (1 - hedge_prob)) / b
        kelly_frac = max(0.0, min(kelly_frac, 1.0))
        partial_hedge_stake = round(original_stake * kelly_frac, 2)
    else:
        partial_hedge_stake = 0.0

    result["full_hedge_stake"]    = round(full_hedge_stake, 2)
    result["full_hedge_profit"]   = round(full_hedge_profit, 2)
    result["partial_hedge_stake"] = partial_hedge_stake

    # Recommendation logic
    if live_edge is not None:
        if live_edge >= 0.06:
            result["recommendation"] = "HOLD"
            result["explanation"]    = f"Model still has +{live_edge*100:.1f}% edge — original bet is live, hold."
        elif live_edge >= 0.0:
            result["recommendation"] = "HOLD"
            result["explanation"]    = f"Slight edge remaining (+{live_edge*100:.1f}%) — hold or partial hedge."
        elif live_edge >= -0.05:
            result["recommendation"] = "PARTIAL HEDGE"
            result["explanation"]    = (f"Edge has flipped ({live_edge*100:.1f}%). "
                                        f"Consider ${partial_hedge_stake:.0f} partial hedge to reduce risk.")
        elif full_hedge_profit > 0:
            result["recommendation"] = "FULL HEDGE"
            result["explanation"]    = (f"Lock in ${full_hedge_profit:.2f} guaranteed profit "
                                        f"by betting ${full_hedge_stake:.0f} on the other side.")
        else:
            result["recommendation"] = "CUT LOSSES"
            result["explanation"]    = "Market has fully moved against your bet. No profitable hedge available."

    return result


# ── ESPN live score fetcher ────────────────────────────────────────────────────

def fetch_live_scores() -> list[dict]:
    """
    Fetch in-progress NCAAB game scores from ESPN scoreboard.
    Returns list of live game dicts.
    """
    import requests
    try:
        r = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard",
            params={"groups": 50, "limit": 200},
            timeout=8,
            headers={"User-Agent": "Basketball-God/2.0 (research)"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    live = []
    for evt in data.get("events", []):
        comp       = evt.get("competitions", [{}])[0]
        status_obj = comp.get("status", {}).get("type", {})
        state      = status_obj.get("name", "")

        if state not in ("STATUS_IN_PROGRESS", "STATUS_HALFTIME"):
            continue

        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        period    = comp.get("status", {}).get("period", 2)
        clock_str = comp.get("status", {}).get("displayClock", "0:00")
        halftime  = state == "STATUS_HALFTIME"
        mins_rem  = 0.0 if halftime else parse_clock(period, clock_str)

        live.append({
            "game_id":         evt.get("id"),
            "home_id":         home.get("team", {}).get("id"),
            "home_name":       home.get("team", {}).get("displayName", ""),
            "away_id":         away.get("team", {}).get("id"),
            "away_name":       away.get("team", {}).get("displayName", ""),
            "home_score":      int(home.get("score", 0) or 0),
            "away_score":      int(away.get("score", 0) or 0),
            "period":          period,
            "clock":           "HALF" if halftime else clock_str,
            "minutes_remaining": mins_rem,
            "halftime":        halftime,
        })

    return live
