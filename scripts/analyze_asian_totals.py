"""Analyze Asian goal lines from the synced model and screenshot prices."""

from datetime import date
from pathlib import Path
import csv

from parlay.data.database import ResearchDatabase
from parlay.models.team_strength import fit_team_strength


def indonesian_to_decimal(value: float) -> float:
    if value == 0:
        return 2.0
    return 1.0 + (1.0 / abs(value) if value < 0 else value)


def settlement(matrix, line: float, side: str) -> dict[str, float]:
    home, away = __import__("numpy").indices(matrix.shape)
    total = home + away
    if side == "over":
        win = total > line
        push = total == line
        loss = total < line
    else:
        win = total < line
        push = total == line
        loss = total > line
    return {"win": float(matrix[win].sum()), "push": float(matrix[push].sum()), "loss": float(matrix[loss].sum())}


def asian_ev(matrix, line: float, side: str, odds_decimal: float) -> float:
    p = settlement(matrix, line, side)
    return p["win"] * (odds_decimal - 1) - p["loss"]


def main() -> None:
    db = ResearchDatabase("data/parlay.sqlite")
    all_matches = db.load_matches()
    history = [m for m in all_matches if m.home_goals is not None and m.away_goals is not None]
    fixtures = {
        "Newcastle vs Bournemouth": ("Newcastle", "Bournemouth", date(2026, 9, 5)),
        "Manchester City vs Coventry": ("Man City", "Coventry", date(2026, 9, 5)),
        "Ipswich Town vs Liverpool": ("Ipswich", "Liverpool", date(2026, 9, 4)),
    }
    prices = {
        "Newcastle vs Bournemouth": [("3", 3.0, "over", -1.01), ("3", 3.0, "under", -1.14), ("2.5/3", 2.75, "over", -1.35), ("2.5/3", 2.75, "under", 1.13), ("3/3.5", 3.25, "over", 1.23), ("3/3.5", 3.25, "under", -1.49)],
        "Manchester City vs Coventry": [("3.5", 3.5, "over", -1.06), ("3.5", 3.5, "under", -1.08), ("3/3.5", 3.25, "over", -1.36), ("3/3.5", 3.25, "under", 1.40), ("3.5/4", 3.75, "over", 1.14), ("3.5/4", 3.75, "under", -1.36)],
        "Ipswich Town vs Liverpool": [("3/3.5", 3.25, "over", 1.01), ("3/3.5", 3.25, "under", -1.17), ("3", 3.0, "over", -1.31), ("3", 3.0, "under", 1.11), ("3.5", 3.5, "over", 1.23), ("3.5", 3.5, "under", -1.49)],
    }
    rows = []
    for name, (home, away, match_date) in fixtures.items():
        training = [m for m in history if m.date < match_date]
        model = fit_team_strength(training, model="dixon_coles", estimator="mle", as_of=match_date, half_life_days=365)
        matrix = model.score_matrix(home, away)
        for label, line, side, indo in prices[name]:
            decimal = indonesian_to_decimal(indo)
            p = settlement(matrix, line, side)
            rows.append({"fixture": name, "line": label, "side": side, "odds_indonesian": indo, "odds_decimal": decimal, "win_probability": p["win"], "push_probability": p["push"], "loss_probability": p["loss"], "ev": asian_ev(matrix, line, side, decimal)})
    output = Path("outputs/asian-totals-analysis.csv"); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for row in rows: print(row)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
