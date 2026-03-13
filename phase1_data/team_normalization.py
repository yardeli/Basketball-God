"""
Team Name Normalization — Map the zoo of NCAA team names to canonical IDs.

NCAA teams have been called different things across different sources:
  - "UConn" vs "Connecticut" vs "University of Connecticut"
  - "NC State" vs "North Carolina State" vs "N.C. State"
  - "Miami (FL)" vs "Miami" vs "Miami Hurricanes"
  - Teams that changed names (e.g., "Southwest Missouri State" → "Missouri State")
  - Conference changes over decades

This module builds a mapping table and resolves names fuzzy-matching style.
Unresolvable names get logged for manual review.
"""
import re
import json
from pathlib import Path
from difflib import SequenceMatcher


# ── Known aliases (hand-curated for common discrepancies) ──
# Format: canonical_name -> [list of known aliases]
KNOWN_ALIASES = {
    "Connecticut Huskies": ["UConn", "Connecticut", "UCONN Huskies"],
    "North Carolina Tar Heels": ["UNC", "North Carolina", "N. Carolina"],
    "North Carolina State Wolfpack": ["NC State", "N.C. State", "North Carolina St."],
    "Southern California Trojans": ["USC", "Southern Cal"],
    "Louisiana State Tigers": ["LSU", "Louisiana St."],
    "Virginia Commonwealth Rams": ["VCU"],
    "Brigham Young Cougars": ["BYU", "Brigham Young"],
    "Texas Christian Horned Frogs": ["TCU"],
    "Southern Methodist Mustangs": ["SMU"],
    "Central Florida Knights": ["UCF"],
    "Nevada-Las Vegas Rebels": ["UNLV"],
    "Texas-El Paso Miners": ["UTEP"],
    "Texas-San Antonio Roadrunners": ["UTSA"],
    "Texas-Arlington Mavericks": ["UT Arlington", "UTA"],
    "Miami Hurricanes": ["Miami (FL)", "Miami FL"],
    "Miami (OH) RedHawks": ["Miami (Ohio)", "Miami Ohio", "Miami OH"],
    "Saint Joseph's Hawks": ["St. Joseph's", "St Joseph's", "Saint Joseph's"],
    "Saint Mary's Gaels": ["St. Mary's", "Saint Mary's (CA)", "St Mary's"],
    "Saint John's Red Storm": ["St. John's", "St John's"],
    "Saint Louis Billikens": ["St. Louis", "Saint Louis"],
    "Saint Peter's Peacocks": ["St. Peter's", "Saint Peter's"],
    "Saint Bonaventure Bonnies": ["St. Bonaventure"],
    "Saint Francis Red Flash": ["St. Francis (PA)", "Saint Francis PA"],
    "Loyola Chicago Ramblers": ["Loyola-Chicago", "Loyola (IL)"],
    "Loyola Marymount Lions": ["Loyola-Marymount", "LMU"],
    "Loyola (MD) Greyhounds": ["Loyola Maryland", "Loyola-Maryland"],
    "Long Island University Sharks": ["LIU", "LIU Brooklyn", "Long Island"],
    "Florida International Panthers": ["FIU"],
    "Florida Atlantic Owls": ["FAU"],
    "Middle Tennessee Blue Raiders": ["Middle Tennessee St.", "MTSU"],
    "Mississippi Rebels": ["Ole Miss"],
    "Mississippi State Bulldogs": ["Miss. State", "Mississippi St."],
    "Pittsburgh Panthers": ["Pitt"],
    "Massachusetts Minutemen": ["UMass"],
    "Wisconsin-Milwaukee Panthers": ["Milwaukee"],
    "Wisconsin-Green Bay Phoenix": ["Green Bay"],
    "Louisiana-Lafayette Ragin' Cajuns": ["Louisiana", "UL Lafayette", "Louisiana-Lafayette"],
    "Louisiana-Monroe Warhawks": ["ULM", "Louisiana-Monroe"],
    "UT Rio Grande Valley Vaqueros": ["UTRGV", "Texas-Pan American"],
    "Southeast Missouri State Redhawks": ["SE Missouri St.", "SEMO"],
    "Missouri State Bears": ["Southwest Missouri St.", "Southwest Missouri State"],
    "Arkansas-Little Rock Trojans": ["Little Rock", "UALR"],
    "Arkansas-Pine Bluff Golden Lions": ["Ark.-Pine Bluff", "UAPB"],
    "Bethune-Cookman Wildcats": ["Bethune-Cookman", "B-CU"],
    "Cal State Bakersfield Roadrunners": ["CSU Bakersfield", "CSUB"],
    "Cal State Fullerton Titans": ["CSU Fullerton", "CSUF"],
    "Cal State Northridge Matadors": ["CSU Northridge", "CSUN"],
    "Stephen F. Austin Lumberjacks": ["SFA", "Stephen F Austin"],
    "Sam Houston State Bearkats": ["Sam Houston", "SHSU"],
    "George Washington Revolutionaries": ["George Washington", "GW"],
}

# ── Name cleaning patterns ──
STRIP_PATTERNS = [
    r"\s+Univ\.?$", r"\s+University$", r"^University of\s+",
    r"\s+College$", r"^The\s+",
]


class TeamNormalizer:
    """Resolves team names across different data sources to canonical IDs."""

    def __init__(self):
        self.canonical_to_id: dict[str, int] = {}
        self.alias_to_canonical: dict[str, str] = {}
        self.unresolved: list[dict] = []
        self._next_id = 1

        # Load known aliases
        for canonical, aliases in KNOWN_ALIASES.items():
            self._register(canonical, aliases)

    def _register(self, canonical: str, aliases: list[str] = None):
        """Register a canonical team name and its aliases."""
        clean = self._clean_name(canonical)

        if canonical not in self.canonical_to_id:
            self.canonical_to_id[canonical] = self._next_id
            self._next_id += 1

        # Map canonical to itself
        self.alias_to_canonical[clean] = canonical
        self.alias_to_canonical[canonical.lower()] = canonical

        if aliases:
            for alias in aliases:
                self.alias_to_canonical[alias.lower()] = canonical
                self.alias_to_canonical[self._clean_name(alias)] = canonical

    def _clean_name(self, name: str) -> str:
        """Normalize a team name for matching."""
        name = name.strip()
        for pattern in STRIP_PATTERNS:
            name = re.sub(pattern, "", name, flags=re.IGNORECASE)
        # Remove common suffixes like "State" abbreviations
        name = re.sub(r"\s+", " ", name).strip()
        return name.lower()

    def resolve(self, name: str, source: str = "unknown") -> tuple[str, int]:
        """
        Resolve a team name to (canonical_name, team_id).
        If no match found, tries fuzzy matching.
        If still no match, registers as new and logs for review.
        """
        if not name:
            return ("Unknown", 0)

        clean = self._clean_name(name)

        # Exact match
        if clean in self.alias_to_canonical:
            canonical = self.alias_to_canonical[clean]
            return (canonical, self.canonical_to_id[canonical])

        # Try lowercase exact
        if name.lower() in self.alias_to_canonical:
            canonical = self.alias_to_canonical[name.lower()]
            return (canonical, self.canonical_to_id[canonical])

        # Fuzzy match against all known names
        best_match, best_score = self._fuzzy_match(name)
        if best_score >= 0.85:
            # Close enough — register as alias
            self.alias_to_canonical[clean] = best_match
            self.alias_to_canonical[name.lower()] = best_match
            return (best_match, self.canonical_to_id[best_match])

        # New team — register it
        self._register(name)
        self.unresolved.append({
            "original_name": name,
            "source": source,
            "best_fuzzy_match": best_match,
            "fuzzy_score": round(best_score, 3) if best_match else 0,
        })
        return (name, self.canonical_to_id[name])

    def _fuzzy_match(self, name: str) -> tuple[str | None, float]:
        """Find the best fuzzy match among known canonical names."""
        clean = self._clean_name(name)
        best_match = None
        best_score = 0.0

        for canonical in self.canonical_to_id:
            score = SequenceMatcher(None, clean, self._clean_name(canonical)).ratio()
            if score > best_score:
                best_score = score
                best_match = canonical

        return (best_match, best_score)

    def load_espn_teams(self, teams_csv_path: str):
        """Load team names from ESPN teams CSV and register them."""
        import pandas as pd
        df = pd.read_csv(teams_csv_path)
        for _, row in df.iterrows():
            name = row.get("name", "")
            if name:
                self._register(name, [
                    row.get("short_name", ""),
                    row.get("abbreviation", ""),
                ])
                # Store ESPN ID mapping
                if name in self.canonical_to_id:
                    # We'll store this in the DB later
                    pass

    def get_unresolved_report(self) -> str:
        """Generate a report of unresolved team names for manual review."""
        if not self.unresolved:
            return "All team names resolved successfully!"

        lines = [f"UNRESOLVED TEAM NAMES ({len(self.unresolved)} total):", ""]
        for entry in self.unresolved:
            match_info = ""
            if entry["best_fuzzy_match"]:
                match_info = f" (closest: {entry['best_fuzzy_match']} @ {entry['fuzzy_score']:.0%})"
            lines.append(f"  [{entry['source']}] {entry['original_name']}{match_info}")

        return "\n".join(lines)

    def save(self, path: str):
        """Save normalization state to JSON."""
        data = {
            "canonical_to_id": self.canonical_to_id,
            "alias_to_canonical": self.alias_to_canonical,
            "unresolved": self.unresolved,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load normalization state from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.canonical_to_id = data["canonical_to_id"]
        self.alias_to_canonical = data["alias_to_canonical"]
        self.unresolved = data.get("unresolved", [])
        self._next_id = max(self.canonical_to_id.values(), default=0) + 1


if __name__ == "__main__":
    norm = TeamNormalizer()

    # Test resolution
    test_names = [
        "Duke Blue Devils", "Duke", "UConn", "Connecticut",
        "NC State", "North Carolina State Wolfpack", "UNLV",
        "Ole Miss", "Miami (FL)", "Miami (OH) RedHawks",
        "St. John's", "Saint John's Red Storm",
        "Some Random School Nobody Heard Of",
    ]

    print("Team Name Resolution Tests:")
    for name in test_names:
        canonical, tid = norm.resolve(name, source="test")
        print(f"  {name:40s} → {canonical} (id={tid})")

    print(f"\n{norm.get_unresolved_report()}")
