"""Plain-text rendering of CLI results."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from parlay.prediction.fixture import FixturePrediction


def pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def table(headers: Sequence[str], rows: Iterable[Sequence[object]], aligns: str | None = None) -> str:
    """Fixed-width table; ``aligns`` is a string of ``l``/``r`` per column."""
    body = [[str(cell) for cell in row] for row in rows]
    widths = [max([len(h), *(len(row[i]) for row in body)]) for i, h in enumerate(headers)]
    aligns = aligns or "l" + "r" * (len(headers) - 1)

    def line(cells: Sequence[str]) -> str:
        return "  ".join(
            c.ljust(w) if a == "l" else c.rjust(w) for c, w, a in zip(cells, widths, aligns, strict=True)
        )

    return "\n".join([line(headers), "  ".join("-" * w for w in widths), *(line(row) for row in body)])


def fixture_summary(predictions: Sequence[FixturePrediction]) -> str:
    rows = []
    for p in predictions:
        fx = p.fixture
        if not p.ok:
            rows.append([fx.match_date.isoformat(), fx.label, "-", "-", "-", "-", "-", p.message])
            continue
        o25 = p.totals.get(2.5, {}).get("over")
        best = p.value_bets[0] if p.value_bets else None
        tip = (
            f"{best.name} @{best.odds:.2f} EV {best.expected_value:+.1%} stake {best.stake:.2%}"
            if best
            else "-"
        )
        flag = p.diagnostics.decision if p.diagnostics and p.diagnostics.decision != "PASS" else ""
        exp_h, exp_a = p.expected_goals or (0.0, 0.0)
        rows.append(
            [
                fx.match_date.isoformat(),
                fx.label,
                f"{exp_h:.2f}-{exp_a:.2f}",
                pct(p.outcome["home_win"]),
                pct(p.outcome["draw"]),
                pct(p.outcome["away_win"]),
                pct(o25) if o25 is not None else "-",
                f"{tip} {flag}".strip(),
            ]
        )
    return table(["Date", "Fixture", "xG", "1", "X", "2", "O2.5", "Best value"], rows, "llrrrrrl")


def fixture_detail(p: FixturePrediction) -> str:
    if not p.ok or p.expected_goals is None:
        return f"{p.fixture.label}: {p.message}"
    fx = p.fixture
    exp_h, exp_a = p.expected_goals
    g = p.grouped
    o = p.outcome
    fair = " / ".join(f"{1 / o[key]:.2f}" for key in ("home_win", "draw", "away_win"))
    lines = [
        f"{fx.match_date.isoformat()}  {fx.label}",
        f"  xG {exp_h:.2f} - {exp_a:.2f}   1X2 {pct(o['home_win'])} {pct(o['draw'])} {pct(o['away_win'])}"
        f"   fair {fair}",
        f"  BTTS yes {pct(g['btts_yes'])}   win-to-nil H {pct(g['home_win_to_nil'])} "
        f"A {pct(g['away_win_to_nil'])}   double chance 1X {pct(g['double_chance_1x'])} "
        f"X2 {pct(g['double_chance_x2'])}",
    ]
    totals = "  ".join(
        f"{line:g}: O {pct(s['over'])}" + (f" (push {s['push']:.0%})" if s["push"] > 0.005 else "")
        for line, s in p.totals.items()
    )
    lines.append(f"  Goals  {totals}")
    handicaps = "  ".join(f"{line:+g}: {pct(s['win'])}" for line, s in p.handicaps.items())
    lines.append(f"  AH home win  {handicaps}")
    lines.append("  Correct score  " + "  ".join(f"{score} {prob:.1%}" for score, prob in p.top_scores))
    if p.expected_corners is not None:
        ch, ca = p.expected_corners
        corners = "  ".join(f"{line:g}: O {pct(s['over'])}" for line, s in p.corner_totals.items())
        lines.append(f"  Corners xC {ch:.1f} - {ca:.1f}  {corners}")
    priced = [q for q in p.quotes if q.odds]
    if priced:
        lines.append("  Priced selections:")
        for q in sorted(priced, key=lambda q: q.expected_value or 0.0, reverse=True):
            edge = f" edge {q.edge:+.1%}" if q.edge is not None else ""
            mark = "  <== value" if q in p.value_bets else ""
            lines.append(
                f"    {q.name:<28} odds {q.odds:6.2f}  fair {q.fair_odds:6.2f}  "
                f"EV {q.expected_value:+6.1%}{edge}{mark}"
            )
    if p.diagnostics and (p.diagnostics.anomaly_flags or p.diagnostics.decision != "PASS"):
        flags = ", ".join(p.diagnostics.anomaly_flags) or "none"
        lines.append(f"  Diagnostics: {p.diagnostics.decision} ({flags})")
    return "\n".join(lines)
