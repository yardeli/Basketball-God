"""
Text Data Pipeline — Ingest news, injury reports, and sentiment signals.

Sources:
  - RSS feeds (ESPN, CBS, RotoWire) for college basketball news
  - RotoWire injury report page for current injuries
  - Simple keyword-based sentiment (no heavy NLP dependencies)

Output: per-team signals that feed into matchup features:
  - injury_impact: estimated impact of injuries (0 = healthy, negative = hurt)
  - sentiment: positive/negative news momentum (-1 to +1)
"""
import re
import sys
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": config.USER_AGENT})


# ─── RSS FEED PARSING ─────────────────────────────────────────────────────────

def fetch_rss_feed(url: str) -> list[dict]:
    """Fetch and parse an RSS feed into article dicts (simple XML parsing)."""
    try:
        resp = SESSION.get(url, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        time.sleep(0.5)
    except Exception as e:
        print(f"  [WARN] RSS fetch failed: {url} — {e}")
        return []

    text = resp.text
    articles = []

    # Simple regex-based XML parsing (avoids lxml/feedparser dependency)
    items = re.findall(r"<item>(.*?)</item>", text, re.DOTALL)
    for item in items:
        title = _extract_tag(item, "title")
        desc = _extract_tag(item, "description")
        link = _extract_tag(item, "link")
        pub_date = _extract_tag(item, "pubDate")

        if title:
            articles.append({
                "title": _clean_html(title),
                "description": _clean_html(desc),
                "link": link,
                "pub_date": pub_date,
                "source": url,
            })

    return articles


def _extract_tag(xml: str, tag: str) -> str:
    """Extract content from an XML tag."""
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Handle CDATA
        cdata = re.match(r"<!\[CDATA\[(.*?)\]\]>", content, re.DOTALL)
        if cdata:
            return cdata.group(1).strip()
        return content
    return ""


def _clean_html(text: str) -> str:
    """Strip HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_all_news() -> list[dict]:
    """Fetch articles from all configured RSS feeds."""
    print("[Text] Fetching news from RSS feeds...")
    all_articles = []

    for name, url in config.RSS_FEEDS.items():
        articles = fetch_rss_feed(url)
        for a in articles:
            a["feed_name"] = name
        all_articles.extend(articles)
        print(f"  {name}: {len(articles)} articles")

    print(f"  Total: {len(all_articles)} articles")
    return all_articles


# ─── INJURY PARSING ──────────────────────────────────────────────────────────

def fetch_injury_report() -> list[dict]:
    """
    Fetch current injury report from RotoWire.
    Returns list of {team, player, status, injury, impact_estimate}.
    """
    print("[Text] Fetching injury report...")

    try:
        resp = SESSION.get(config.ROTOWIRE_INJURY_URL, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [WARN] Injury report fetch failed: {e}")
        return []

    html = resp.text
    injuries = []

    # Parse injury rows — RotoWire uses table structure
    # Look for player entries with status indicators
    rows = re.findall(
        r'class="[^"]*injury[^"]*"[^>]*>.*?</tr>',
        html, re.DOTALL | re.IGNORECASE
    )

    # Fallback: look for common injury status patterns
    # Format: "Player Name (Team) — Status: Injury Description"
    status_patterns = re.findall(
        r'(?:Out|Doubtful|Questionable|Probable|Day-To-Day)\s*[-–—]\s*\w+',
        html, re.IGNORECASE
    )

    # Simplified: extract team-level injury counts from page text
    # More robust than trying to parse exact HTML structure
    team_sections = re.findall(
        r'<h[23][^>]*>([^<]+)</h[23]>.*?(?=<h[23]|$)',
        html, re.DOTALL
    )

    for section in team_sections:
        # Count injury statuses
        out_count = len(re.findall(r'\bOut\b', section, re.IGNORECASE))
        doubtful_count = len(re.findall(r'\bDoubtful\b', section, re.IGNORECASE))
        questionable_count = len(re.findall(r'\bQuestionable\b', section, re.IGNORECASE))
        day_to_day_count = len(re.findall(r'\bDay-To-Day\b', section, re.IGNORECASE))

        # Extract team name from section header
        header = re.match(r'([^<]+)', section)
        if header and (out_count + doubtful_count + questionable_count + day_to_day_count) > 0:
            team_name = header.group(1).strip()
            injuries.append({
                "team": team_name,
                "out": out_count,
                "doubtful": doubtful_count,
                "questionable": questionable_count,
                "day_to_day": day_to_day_count,
                "impact_estimate": _estimate_injury_impact(
                    out_count, doubtful_count, questionable_count, day_to_day_count
                ),
            })

    print(f"  Found injury data for {len(injuries)} teams")
    return injuries


def _estimate_injury_impact(out: int, doubtful: int, questionable: int, dtd: int) -> float:
    """
    Estimate negative impact of injuries on team performance.
    Returns a negative number (0 = healthy, -1 = severely hurt).
    Weights: Out=-0.15, Doubtful=-0.10, Questionable=-0.05, DTD=-0.02 per player.
    """
    impact = -(out * 0.15 + doubtful * 0.10 + questionable * 0.05 + dtd * 0.02)
    return max(impact, -1.0)  # Cap at -1.0


# ─── SENTIMENT ANALYSIS ──────────────────────────────────────────────────────

# Keyword-based sentiment — simple but effective for sports news
POSITIVE_KEYWORDS = {
    "win", "won", "victory", "dominant", "surge", "streak", "ranked", "top",
    "impressive", "blowout", "upset win", "clinch", "return", "healthy",
    "comeback", "momentum", "rolling", "hot", "undefeated", "strong",
    "breakout", "star", "elite", "powerhouse", "championship", "contender",
}

NEGATIVE_KEYWORDS = {
    "loss", "lost", "upset", "injury", "injured", "out", "suspend",
    "struggle", "slump", "collapse", "turnover", "foul", "miss",
    "disappointing", "eliminated", "benched", "torn", "fracture", "sprain",
    "concussion", "indefinitely", "sideline", "underperform", "crisis",
    "fired", "coaching change", "transfer", "decommit",
}


def analyze_sentiment(text: str) -> float:
    """
    Simple keyword-based sentiment score.
    Returns -1.0 to +1.0 (negative to positive).
    """
    if not text:
        return 0.0

    words = set(text.lower().split())
    pos_count = sum(1 for w in POSITIVE_KEYWORDS if w in words or w in text.lower())
    neg_count = sum(1 for w in NEGATIVE_KEYWORDS if w in words or w in text.lower())

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total


def compute_team_sentiment(articles: list[dict], teams_df: pd.DataFrame) -> dict[str, float]:
    """
    Map articles to teams and compute per-team sentiment.
    Returns {team_name: sentiment_score}.
    """
    print("[Text] Computing team sentiment...")

    team_names = set()
    if teams_df is not None and len(teams_df) > 0:
        for col in ["name", "short_name", "abbreviation"]:
            if col in teams_df.columns:
                team_names.update(teams_df[col].dropna().str.lower().tolist())

    team_sentiments: dict[str, list[float]] = {}

    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        sentiment = analyze_sentiment(text)

        if sentiment == 0:
            continue

        # Match article to teams
        text_lower = text.lower()
        for _, team in teams_df.iterrows() if teams_df is not None else []:
            team_name = str(team.get("name", "")).lower()
            short = str(team.get("short_name", "")).lower()
            abbrev = str(team.get("abbreviation", "")).lower()

            if (team_name and team_name in text_lower) or \
               (short and len(short) > 3 and short in text_lower):
                key = team.get("name", "")
                if key not in team_sentiments:
                    team_sentiments[key] = []
                team_sentiments[key].append(sentiment)

    # Average sentiment per team
    result = {}
    for team, scores in team_sentiments.items():
        result[team] = round(sum(scores) / len(scores), 4)

    print(f"  Sentiment computed for {len(result)} teams")
    return result


# ─── COMBINED TEXT SIGNALS ────────────────────────────────────────────────────

def get_text_signals(teams_df: pd.DataFrame = None) -> dict:
    """
    Fetch and combine all text-derived signals.
    Returns {
        "injuries": {team_name: impact_score},
        "sentiment": {team_name: sentiment_score},
    }
    """
    articles = fetch_all_news()
    injuries = fetch_injury_report()
    sentiment = compute_team_sentiment(articles, teams_df)

    injury_map = {}
    for inj in injuries:
        injury_map[inj["team"]] = inj["impact_estimate"]

    signals = {
        "injuries": injury_map,
        "sentiment": sentiment,
        "fetched_at": datetime.now().isoformat(),
        "n_articles": len(articles),
        "n_teams_with_injuries": len(injury_map),
        "n_teams_with_sentiment": len(sentiment),
    }

    # Cache
    cache_path = config.CACHE_DIR / "text_signals.json"
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(signals, f, indent=2)
    print(f"[Text] Cached signals to {cache_path}")

    return signals


def load_cached_signals() -> dict | None:
    """Load cached text signals if they exist and are recent (< 6 hours old)."""
    cache_path = config.CACHE_DIR / "text_signals.json"
    if not cache_path.exists():
        return None

    with open(cache_path) as f:
        signals = json.load(f)

    fetched = datetime.fromisoformat(signals.get("fetched_at", "2000-01-01"))
    if datetime.now() - fetched > timedelta(hours=6):
        return None

    return signals


if __name__ == "__main__":
    signals = get_text_signals()
    print(f"\nInjury impacts: {len(signals['injuries'])} teams")
    for team, impact in sorted(signals["injuries"].items(), key=lambda x: x[1])[:10]:
        print(f"  {team}: {impact:.3f}")

    print(f"\nSentiment scores: {len(signals['sentiment'])} teams")
    for team, sent in sorted(signals["sentiment"].items(), key=lambda x: x[1])[:10]:
        print(f"  {team}: {sent:+.3f}")
