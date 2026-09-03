"""Historical rolling features with strict pre-match information boundaries."""

from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import json
import math

from parlay.data.schemas import Match
from parlay.data.validation import validate_matches
from .registry import FeatureSet, FeatureValue


def build_pre_match_features(matches: list[Match], *, window: int = 5, forecast_lead_minutes: int = 60) -> dict[str, dict[str, float]]:
    """Build rolling features using InformationSet semantics (§1).

    Features for a match may only use historical matches whose
    result_known_at <= forecast_timestamp (kickoff - lead).
    Falls back to date < row.date when kickoff unavailable.
    """
    if window < 1:
        raise ValueError("window must be positive")
    rows = sorted(validate_matches(matches), key=lambda row: (row.date, row.match_id))
    # History stores (known_at, date, goals_for, goals_against, won, corners..., shots...)
    from datetime import datetime, timedelta, timezone, time as dtime
    from parlay.data.information import InformationSet, RESULT_LAG_MINUTES
    history: dict[str, list[tuple[datetime, date, int, int, bool, int, int, int, int]]] = {}
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        # Forecast timestamp for this row
        if row.kickoff_at is not None:
            ft = row.kickoff_at
            if ft.tzinfo is None:
                ft = ft.replace(tzinfo=timezone.utc)
            forecast_ts = ft - timedelta(minutes=forecast_lead_minutes)
        else:
            forecast_ts = datetime.combine(row.date, dtime(0,0), tzinfo=timezone.utc)
        team_history = {}
        for team in (row.home_team, row.away_team):
            # Filter knowable: known_at <= forecast_ts (per-competition lag inside InformationSet)
            prior_all_knowable = [item for item in history.get(team, []) if item[0] <= forecast_ts]
            prior = prior_all_knowable[-window:]
            count = len(prior)
            # indices: 0=known_at, 1=date, 2=goals_for, 3=goals_against, 4=won, 5=corners_for, 6=corners_against, 7=shots_for, 8=shots_against
            goals_for = sum(item[2] for item in prior)
            goals_against = sum(item[3] for item in prior)
            wins = sum(item[4] for item in prior)
            corners_for = sum(item[5] for item in prior)
            corners_against = sum(item[6] for item in prior)
            shots_for = sum(item[7] for item in prior)
            shots_against = sum(item[8] for item in prior)

            # Calculate rest days since previous match (using actual date, not known_at)
            rest_days = float((row.date - prior_all_knowable[-1][1]).days) if prior_all_knowable else 7.0

            team_history[team] = {
                "matches": float(count),
                "goals_for_per_match": goals_for / count if count else 0.0,
                "goals_against_per_match": goals_against / count if count else 0.0,
                "win_rate": wins / count if count else 0.0,
                "corners_for_per_match": corners_for / count if count else 0.0,
                "corners_against_per_match": corners_against / count if count else 0.0,
                "shots_for_per_match": shots_for / count if count else 0.0,
                "shots_against_per_match": shots_against / count if count else 0.0,
                "rest_days": rest_days,
            }
        home = team_history[row.home_team]
        away = team_history[row.away_team]
        output[row.match_id] = {
            "home_matches": home["matches"],
            "away_matches": away["matches"],
            "home_goals_for_per_match": home["goals_for_per_match"],
            "away_goals_for_per_match": away["goals_for_per_match"],
            "home_goals_against_per_match": home["goals_against_per_match"],
            "away_goals_against_per_match": away["goals_against_per_match"],
            "home_win_rate": home["win_rate"],
            "away_win_rate": away["win_rate"],
            "home_corners_for_per_match": home["corners_for_per_match"],
            "away_corners_for_per_match": away["corners_for_per_match"],
            "home_corners_against_per_match": home["corners_against_per_match"],
            "away_corners_against_per_match": away["corners_against_per_match"],
            "home_shots_for_per_match": home["shots_for_per_match"],
            "away_shots_for_per_match": away["shots_for_per_match"],
            "home_rest_days": home["rest_days"],
            "away_rest_days": away["rest_days"],
        }
        # Compute known_at for this match to store in history (for future rows)
        if row.kickoff_at is not None:
            ka = row.kickoff_at
            if ka.tzinfo is None:
                ka = ka.replace(tzinfo=timezone.utc)
            lag = RESULT_LAG_MINUTES.get(row.competition, RESULT_LAG_MINUTES["default"])
            known_at = ka + timedelta(minutes=lag)
        else:
            known_at = datetime.combine(row.date, dtime(23,59), tzinfo=timezone.utc)
        history.setdefault(row.home_team, []).append((
            known_at, row.date,
            row.home_goals or 0,
            row.away_goals or 0,
            (row.home_goals or 0) > (row.away_goals or 0),
            row.home_corners or 0,
            row.away_corners or 0,
            row.home_shots or 0,
            row.away_shots or 0,
        ))
        history.setdefault(row.away_team, []).append((
            known_at, row.date,
            row.away_goals or 0,
            row.home_goals or 0,
            (row.away_goals or 0) > (row.home_goals or 0),
            row.away_corners or 0,
            row.home_corners or 0,
            row.away_shots or 0,
            row.home_shots or 0,
        ))
    return output


def build_pre_match_feature_sets(
    matches: list[Match], *, window: int = 5, forecast_lead_minutes: int = 60,
    source: str = "historical_match_results",
) -> dict[str, FeatureSet]:
    """Build rolling features with explicit computed/available timestamps.

    The legacy dictionary builder remains unchanged for compatibility. This
    companion API makes provenance part of the data contract for new callers.
    """
    values = build_pre_match_features(
        matches, window=window, forecast_lead_minutes=forecast_lead_minutes
    )
    result: dict[str, FeatureSet] = {}
    for match in validate_matches(matches):
        if match.kickoff_at is not None:
            kickoff = match.kickoff_at
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
            as_of = kickoff - timedelta(minutes=forecast_lead_minutes)
        else:
            as_of = datetime.combine(match.date, datetime.min.time(), timezone.utc)
        feature_values = {
            name: FeatureValue(
                value=float(value),
                source=source,
                computed_at=as_of,
                available_at=as_of,
            )
            for name, value in values.get(match.match_id, {}).items()
        }
        result[match.match_id] = FeatureSet(
            match_id=match.match_id,
            as_of=as_of,
            groups={"historical": feature_values},
        )
    return result


def features_json(features: dict[str, float]) -> str:
    return json.dumps(features, sort_keys=True, separators=(",", ":"))
