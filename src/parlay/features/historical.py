"""Historical rolling features with strict pre-match information boundaries."""

from dataclasses import asdict
from datetime import date
import json
import math

from parlay.data.schemas import Match
from parlay.data.validation import validate_matches


def build_pre_match_features(matches: list[Match], *, window: int = 5) -> dict[str, dict[str, float]]:
    """Build rolling goals/form/corner features using only earlier completed matches.

    The result is keyed by match_id. Matches on the same date are excluded from
    one another, which is conservative when kickoff timestamps are unavailable.
    """
    if window < 1:
        raise ValueError("window must be positive")
    rows = sorted(validate_matches(matches), key=lambda row: (row.date, row.match_id))
    # History tuple: (date, goals_for, goals_against, won, corners_for, corners_against, shots_for, shots_against)
    history: dict[str, list[tuple[date, int, int, bool, int, int, int, int]]] = {}
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        team_history = {}
        for team in (row.home_team, row.away_team):
            prior = [item for item in history.get(team, []) if item[0] < row.date][-window:]
            count = len(prior)
            goals_for = sum(item[1] for item in prior)
            goals_against = sum(item[2] for item in prior)
            wins = sum(item[3] for item in prior)
            corners_for = sum(item[4] for item in prior)
            corners_against = sum(item[5] for item in prior)
            shots_for = sum(item[6] for item in prior)
            shots_against = sum(item[7] for item in prior)

            # Calculate rest days since previous match
            prior_all = [item for item in history.get(team, []) if item[0] < row.date]
            rest_days = float((row.date - prior_all[-1][0]).days) if prior_all else 7.0

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
        history.setdefault(row.home_team, []).append((
            row.date,
            row.home_goals or 0,
            row.away_goals or 0,
            (row.home_goals or 0) > (row.away_goals or 0),
            row.home_corners or 0,
            row.away_corners or 0,
            row.home_shots or 0,
            row.away_shots or 0,
        ))
        history.setdefault(row.away_team, []).append((
            row.date,
            row.away_goals or 0,
            row.home_goals or 0,
            (row.away_goals or 0) > (row.home_goals or 0),
            row.away_corners or 0,
            row.home_corners or 0,
            row.away_shots or 0,
            row.home_shots or 0,
        ))
    return output


def features_json(features: dict[str, float]) -> str:
    return json.dumps(features, sort_keys=True, separators=(",", ":"))
