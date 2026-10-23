"""Canonical team names.

football-data.co.uk names (``Man City``, ``Nott'm Forest``, ...) are the
canonical form because the historical database is built from those files.
Fixture CSVs and API providers (Sportmonks, bookmakers) are mapped onto them.
"""

from __future__ import annotations

_ALIASES: dict[str, str] = {
    # Premier League / Championship full names and common variants
    "afc bournemouth": "Bournemouth",
    "arsenal fc": "Arsenal",
    "aston villa fc": "Aston Villa",
    "birmingham city": "Birmingham",
    "blackburn rovers": "Blackburn",
    "bolton wanderers": "Bolton",
    "brentford fc": "Brentford",
    "brighton & hove albion": "Brighton",
    "brighton and hove albion": "Brighton",
    "brighton hove albion": "Brighton",
    "bristol city fc": "Bristol City",
    "burnley fc": "Burnley",
    "cardiff city": "Cardiff",
    "charlton athletic": "Charlton",
    "chelsea fc": "Chelsea",
    "coventry city": "Coventry",
    "crystal palace fc": "Crystal Palace",
    "derby county": "Derby",
    "everton fc": "Everton",
    "fulham fc": "Fulham",
    "huddersfield town": "Huddersfield",
    "hull city": "Hull",
    "ipswich town": "Ipswich",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "lincoln city": "Lincoln",
    "liverpool fc": "Liverpool",
    "luton town": "Luton",
    "manchester city": "Man City",
    "man. city": "Man City",
    "manchester united": "Man United",
    "manchester utd": "Man United",
    "man utd": "Man United",
    "man. united": "Man United",
    "middlesbrough fc": "Middlesbrough",
    "millwall fc": "Millwall",
    "newcastle united": "Newcastle",
    "newcastle utd": "Newcastle",
    "norwich city": "Norwich",
    "nottingham forest": "Nott'm Forest",
    "nottingham": "Nott'm Forest",
    "nottm forest": "Nott'm Forest",
    "oxford united": "Oxford",
    "peterborough united": "Peterboro",
    "peterborough": "Peterboro",
    "plymouth argyle": "Plymouth",
    "portsmouth fc": "Portsmouth",
    "preston north end": "Preston",
    "queens park rangers": "QPR",
    "reading fc": "Reading",
    "rotherham united": "Rotherham",
    "sheffield utd": "Sheffield United",
    "sheffield wednesday": "Sheffield Weds",
    "southampton fc": "Southampton",
    "spurs": "Tottenham",
    "stoke city": "Stoke",
    "sunderland afc": "Sunderland",
    "swansea city": "Swansea",
    "tottenham hotspur": "Tottenham",
    "watford fc": "Watford",
    "west bromwich albion": "West Brom",
    "west bromwich": "West Brom",
    "west ham united": "West Ham",
    "west ham utd": "West Ham",
    "wigan athletic": "Wigan",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton": "Wolves",
    "wrexham afc": "Wrexham",
    "wycombe wanderers": "Wycombe",
}


def default_team_aliases() -> dict[str, str]:
    """Lower-cased alias → canonical name."""
    return dict(_ALIASES)


def canonical_team_name(name: str, aliases: dict[str, str] | None = None) -> str:
    """Map a provider/fixture team name onto the canonical football-data name."""
    cleaned = " ".join(name.strip().split())
    return (aliases or _ALIASES).get(cleaned.casefold(), cleaned)
