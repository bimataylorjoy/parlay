"""Bookmaker odds columns of football-data.co.uk files (results and fixtures).

Each bookmaker has up to three column families — 1X2, Over/Under 2.5 and
Asian handicap — and a closing variant formed by inserting ``C`` after the
prefix (``B365H`` → ``B365CH``, ``P>2.5`` → ``PC>2.5``, ``PAHH`` → ``PCAHH``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from .schemas import OddsSnapshot

MARKET_MAX = "market_max"  # best price across the bookmakers football-data tracks
MARKET_AVG = "market_avg"  # average price across them
EXCHANGE = "Betfair Exchange"
# Typical commission on net winnings; override with --commission.
EXCHANGE_COMMISSION = {EXCHANGE: 0.05, "Matchbook": 0.02, "Smarkets": 0.02}
EXCHANGES = frozenset(EXCHANGE_COMMISSION)
PSEUDO_BOOKMAKERS = frozenset({MARKET_MAX, MARKET_AVG})


@dataclass(frozen=True, slots=True)
class BookColumns:
    name: str
    x12: str | None = None
    totals: str | None = None
    asian: str | None = None


BOOKS: tuple[BookColumns, ...] = (
    BookColumns("Bet365", "B365", "B365", "B365"),
    BookColumns("Pinnacle", "PS", "P", "P"),
    BookColumns(EXCHANGE, "BFE", "BFE", "BFE"),
    BookColumns("Betfred", "BFD"),
    BookColumns("BetMGM", "BMGM"),
    BookColumns("BetVictor", "BV"),
    BookColumns("bwin", "BW"),
    BookColumns("Coral", "CL"),
    BookColumns("Interwetten", "IW"),
    BookColumns("Ladbrokes", "LB"),
    BookColumns("Paddy Power", "PP"),
    BookColumns("Sky Bet", "SKB"),
    BookColumns("William Hill", "WH"),
    BookColumns("VC Bet", "VC"),
    BookColumns(MARKET_MAX, "Max", "Max", "Max"),
    BookColumns(MARKET_AVG, "Avg", "Avg", "Avg"),
)


def totals_market(line: float) -> str:
    return f"totals_{line:g}"


def asian_market(line: float) -> str:
    """Asian handicap market key; ``line`` is the home handicap (``ah_-0.75``)."""
    return f"ah_{line + 0.0:g}"


def parse_market(market: str) -> tuple[str, float | None]:
    """``"totals_2.5"`` → ``("totals", 2.5)``; ``"1x2"`` → ``("1x2", None)``."""
    kind, _, line = market.partition("_")
    return kind, float(line) if line else None


def _price(value: str | None) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        price = float(value)
    except ValueError:
        return None
    return price if price > 1.0 else None


def _line(value: str | None) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        line = float(value)
    except ValueError:
        return None
    return line if line * 4 == int(line * 4) else None


def row_odds(
    row: Mapping[str, str | None],
    match_id: str,
    *,
    pre_match_at: datetime,
    closing_at: datetime,
) -> tuple[list[OddsSnapshot], int]:
    """All complete odds groups in one CSV row, plus the number of malformed groups skipped."""
    snapshots: list[OddsSnapshot] = []
    skipped = 0
    for closing in (False, True):
        c = "C" if closing else ""
        captured = closing_at if closing else pre_match_at
        ah_line = _line(row.get("AHCh" if closing else "AHh"))
        for book in BOOKS:
            groups: list[tuple[str, tuple[tuple[str, str], ...]]] = []
            if book.x12:
                p = book.x12 + c
                groups.append(("1x2", (("home", f"{p}H"), ("draw", f"{p}D"), ("away", f"{p}A"))))
            if book.totals:
                p = book.totals + c
                groups.append((totals_market(2.5), (("over", f"{p}>2.5"), ("under", f"{p}<2.5"))))
            if book.asian and ah_line is not None:
                p = book.asian + c
                groups.append((asian_market(ah_line), (("home", f"{p}AHH"), ("away", f"{p}AHA"))))
            for market, columns in groups:
                raw = [row.get(column) for _, column in columns]
                if not any((value or "").strip() for value in raw):
                    continue
                prices = [_price(value) for value in raw]
                if any(price is None for price in prices):
                    skipped += 1
                    continue
                snapshots.extend(
                    OddsSnapshot(match_id, book.name, market, selection, price, captured, closing)  # type: ignore[arg-type]
                    for (selection, _), price in zip(columns, prices, strict=True)
                )
    return snapshots, skipped
