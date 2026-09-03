"""Feature registry with provenance (§9)."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class FeatureValue:
    value: float
    source: str  # e.g., "football-data E1", "sportmonks xG"
    computed_at: datetime
    available_at: datetime  # when it becomes knowable (kickoff+lag)


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Collection of FeatureValue per match per team group."""

    match_id: str
    as_of: datetime
    groups: dict[str, dict[str, FeatureValue]]  # group -> feature -> value


# Registry of feature groups and their provenance assumptions

FEATURE_GROUPS: dict[str, dict[str, str]] = {
    "match_performance": {
        "goals": "football-data FTHG/FTAG, available at known_at (kickoff+115/118m)",
        "sot": "football-data HST/AST, same availability, used as covariate β_sot per league",
        "shots": "HS/AS, same",
        "corners": "HC/AC, same",
    },
    "context": {
        "rest_days": "derived from match.date, knowable at forecast (no lag)",
        "home_away": "fixture, knowable at schedule",
        "competition": "fixture, knowable",
    },
    "dynamics": {
        "time_decay_form": "exponential w=0.5^{age/half_life}, per InformationSet",
        "promoted_uncertainty": "flag if team has <10 historical matches in competition",
    },
}

# League-specific SOT covariate shrinkage (hierarchical partial pooling)
# β_sot[league] ~ Normal(β_global, 0.1)
SOT_BETA_GLOBAL: float = 0.25
SOT_BETA_LEAGUE: dict[str, float] = {
    "EPL": 0.28,
    "Championship": 0.22,
    "default": 0.25,
}


def get_sot_beta(competition: str) -> float:
    return SOT_BETA_LEAGUE.get(competition, SOT_BETA_LEAGUE["default"])
