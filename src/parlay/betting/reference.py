"""Fair prices from a sharp reference: Pinnacle, Betfair Exchange or the market consensus.

Liquid sharp prices are the best publicly available probability forecasts
(Štrumbelj, 2014); the average price across many bookmakers is a good
substitute (Kaunitz, Zhong & Kriener, 2017). A bet has value when a
bookmaker offers more than the fair price implied by the reference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from parlay.data.odds_columns import EXCHANGE, EXCHANGES, MARKET_AVG, PSEUDO_BOOKMAKERS
from parlay.prediction.markets import de_vig

REFERENCE_POLICIES = ("auto", "pinnacle", "exchange", "consensus")
SHARP_BOOKS = {"pinnacle": "Pinnacle", "exchange": EXCHANGE}
MIN_CONSENSUS_BOOKS = 3

Quotes = Mapping[str, Mapping[str, float]]  # bookmaker -> selection -> decimal odds


@dataclass(frozen=True, slots=True)
class FairPrice:
    probabilities: dict[str, float]
    source: str
    books: int

    def odds(self, selection: str) -> float:
        return 1.0 / self.probabilities[selection]


def _complete(prices: Mapping[str, float] | None, selections: Sequence[str]) -> bool:
    return prices is not None and all(prices.get(s, 0.0) > 1.0 for s in selections)


def _devig(prices: Mapping[str, float], selections: Sequence[str], method: str) -> dict[str, float]:
    return de_vig({s: 1.0 / prices[s] for s in selections}, method=method)


def fair_price(
    quotes: Quotes, selections: Sequence[str], policy: str = "auto", *, method: str = "shin"
) -> FairPrice | None:
    """De-vigged probabilities for one market, or ``None`` when the reference is unavailable.

    ``auto`` tries Pinnacle, then Betfair Exchange, then the consensus of
    regular bookmakers (at least three with a complete market, else the
    football-data market average).
    """
    if policy not in REFERENCE_POLICIES:
        raise ValueError(f"policy must be one of {REFERENCE_POLICIES}")
    order = ("pinnacle", "exchange", "consensus") if policy == "auto" else (policy,)
    for name in order:
        if name in SHARP_BOOKS:
            prices = quotes.get(SHARP_BOOKS[name])
            if _complete(prices, selections):
                return FairPrice(_devig(prices, selections, method), SHARP_BOOKS[name], 1)  # type: ignore[arg-type]
            continue
        books = [
            prices
            for book, prices in quotes.items()
            if book not in PSEUDO_BOOKMAKERS and book not in EXCHANGES and _complete(prices, selections)
        ]
        if len(books) >= MIN_CONSENSUS_BOOKS:
            implied = {s: sum(1.0 / b[s] for b in books) / len(books) for s in selections}
            return FairPrice(de_vig(implied, method=method), "consensus", len(books))
        if _complete(quotes.get(MARKET_AVG), selections):
            return FairPrice(_devig(quotes[MARKET_AVG], selections, method), "consensus", 0)
    return None
