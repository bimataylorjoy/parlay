"""Generate an auditable EPL totals report from synced Sportmonks data."""

import csv
from datetime import date
from pathlib import Path

from parlay.data.database import ResearchDatabase
from parlay.models.poisson import score_matrix
from parlay.models.team_strength import fit_team_strength
from parlay.prediction.markets import totals_expected_value, totals_settlement_probabilities


def main() -> None:
    database = ResearchDatabase("data/parlay.sqlite")
    rows = database.load_matches()
    history = [row for row in rows if row.home_goals is not None and row.away_goals is not None]
    fixtures = [row for row in rows if row.match_id.startswith("sportmonks:") and row.home_goals is None and row.away_goals is None]
    output = Path("outputs/epl-totals-final.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = []
    for fixture in fixtures:
        training = [row for row in history if row.date < fixture.date]
        teams = {team for row in training for team in (row.home_team, row.away_team)}
        odds = {row.selection: row.odds for row in database.load_odds(match_id=fixture.match_id) if row.market == "totals_2.5"}
        if fixture.home_team not in teams or fixture.away_team not in teams:
            result.append({"fixture_id": fixture.match_id, "fixture": f"{fixture.home_team} vs {fixture.away_team}", "status": "insufficient_history"})
            continue
        model = fit_team_strength(training, model="dixon_coles", estimator="mle", as_of=fixture.date, half_life_days=365)
        matrix = model.score_matrix(fixture.home_team, fixture.away_team)
        line = 2.5
        probabilities = totals_settlement_probabilities(matrix, line)
        over_ev = totals_expected_value({"over": probabilities["over"], "under": probabilities["under"], "push": probabilities["push"]}, odds["over"]) if "over" in odds and "under" in odds else None
        under_ev = probabilities["under"] * (odds["under"] - 1) - probabilities["over"] if "over" in odds and "under" in odds else None
        result.append({"fixture_id": fixture.match_id, "kickoff_at": fixture.kickoff_at.isoformat() if fixture.kickoff_at else "", "fixture": f"{fixture.home_team} vs {fixture.away_team}", "status": "ok", "line": line, "over_probability": probabilities["over"], "push_probability": probabilities["push"], "under_probability": probabilities["under"], "over_odds": odds.get("over"), "under_odds": odds.get("under"), "over_ev": over_ev, "under_ev": under_ev})
    fields = sorted({key for row in result for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(result)
    for row in result:
        print(row)
    print(f"Wrote {len(result)} rows to {output}")


if __name__ == "__main__":
    main()
