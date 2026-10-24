from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from conftest import make_match
from parlay.betting import Ledger, ValueConfig, fair_price, find_value_bets, run_value_backtest, settle
from parlay.betting.value import effective_odds, shrunk_kelly
from parlay.data.database import ResearchDatabase
from parlay.data.odds_columns import (
    EXCHANGE,
    MARKET_AVG,
    MARKET_MAX,
    asian_market,
    parse_market,
    totals_market,
)
from parlay.data.schemas import OddsSnapshot

T0 = datetime(2030, 1, 5, 9, tzinfo=timezone.utc)
KICKOFF = datetime(2030, 1, 5, 15, tzinfo=timezone.utc)


def snaps(match_id, book, market, prices, *, at=T0, closing=False):
    return [OddsSnapshot(match_id, book, market, sel, odds, at, closing) for sel, odds in prices.items()]


def x12(home, draw, away):
    return {"home": home, "draw": draw, "away": away}


# --- reference prices -------------------------------------------------------


def test_reference_policies_and_fallbacks():
    quotes = {
        "Pinnacle": x12(2.0, 3.6, 4.0),
        EXCHANGE: x12(2.1, 3.7, 4.2),
        "Bet365": x12(1.95, 3.5, 3.9),
        "bwin": x12(1.9, 3.5, 4.0),
        "William Hill": x12(2.0, 3.4, 3.8),
    }
    sel = ("home", "draw", "away")
    assert fair_price(quotes, sel, "auto").source == "Pinnacle"
    assert fair_price(quotes, sel, "exchange").source == EXCHANGE
    consensus = fair_price(quotes, sel, "consensus")
    assert consensus.source == "consensus" and consensus.books == 4  # exchange excluded
    assert sum(consensus.probabilities.values()) == pytest.approx(1.0)
    no_sharp = {k: v for k, v in quotes.items() if k not in ("Pinnacle", EXCHANGE)}
    assert fair_price(no_sharp, sel, "auto").source == "consensus"
    assert fair_price(no_sharp, sel, "pinnacle") is None
    assert fair_price({MARKET_AVG: x12(2.0, 3.5, 4.0)}, sel, "consensus").books == 0
    assert fair_price({"Pinnacle": {"home": 2.0, "draw": 3.6}}, sel, "pinnacle") is None
    with pytest.raises(ValueError):
        fair_price(quotes, sel, "oracle")


# --- settlement -------------------------------------------------------------


@pytest.mark.parametrize(
    "market,selection,score,expected",
    [
        ("1x2", "home", (2, 1), "win"),
        ("1x2", "draw", (2, 1), "loss"),
        ("totals_2.5", "over", (2, 1), "win"),
        ("totals_3", "over", (2, 1), "push"),
        ("totals_2.75", "over", (2, 1), "half_win"),
        ("totals_2.25", "under", (2, 1), "loss"),
        ("totals_2.25", "over", (1, 1), "half_loss"),
        ("ah_-0.75", "home", (1, 0), "half_win"),
        ("ah_-0.75", "away", (1, 0), "half_loss"),
        ("ah_-1", "home", (1, 0), "push"),
        ("ah_0.5", "away", (0, 0), "loss"),
    ],
)
def test_settlement(market, selection, score, expected):
    assert settle(market, selection, *score).label == expected


def test_settlement_profit_and_validation():
    assert settle("totals_2.75", "over", 2, 1).profit(10, 2.0) == pytest.approx(5.0)
    assert settle("ah_-0.75", "away", 1, 0).profit(10, 2.0) == pytest.approx(-5.0)
    with pytest.raises(ValueError):
        settle("corners_9.5", "over", 1, 1)
    with pytest.raises(ValueError):
        settle("1x2", "over", 1, 1)


def test_market_keys_round_trip():
    assert parse_market(totals_market(2.5)) == ("totals", 2.5)
    assert parse_market(asian_market(-0.75)) == ("ah", -0.75)
    assert asian_market(-0.0) == "ah_0"


# --- staking and value ------------------------------------------------------


def test_shrunk_kelly_and_exchange_commission():
    plain = shrunk_kelly(0.55, 2.0, fraction=1.0, max_stake=1.0, probability_sd=0.0)
    assert plain == pytest.approx(0.10)
    shrunk = shrunk_kelly(0.55, 2.0, fraction=1.0, max_stake=1.0, probability_sd=0.05)
    assert 0 < shrunk < plain
    assert shrunk_kelly(0.45, 2.0, fraction=1.0, max_stake=1.0, probability_sd=0.0) == 0.0
    assert effective_odds(3.0, EXCHANGE, 0.05) == pytest.approx(2.9)
    assert effective_odds(3.0, "Bet365", 0.05) == 3.0


def _market_rows(match_id="m1"):
    rows = snaps(match_id, "Pinnacle", "1x2", x12(2.0, 3.6, 4.0))
    rows += snaps(match_id, "Bet365", "1x2", x12(2.25, 3.3, 3.5))  # home overpriced
    rows += snaps(match_id, "bwin", "1x2", x12(2.10, 3.3, 3.6))
    rows += snaps(match_id, EXCHANGE, "1x2", x12(2.30, 3.7, 4.2))
    rows += snaps(match_id, MARKET_MAX, "1x2", x12(2.60, 3.8, 4.4))
    rows += snaps(match_id, "Pinnacle", "totals_2.5", {"over": 1.95, "under": 1.95})
    rows += snaps(match_id, "Bet365", "totals_2.5", {"over": 2.08, "under": 1.80})
    return rows


def test_find_value_bets_uses_best_bettable_price():
    match = make_match("m1", date(2030, 1, 5), kickoff_at=KICKOFF)

    def home_bet(commission):
        bets = find_value_bets([match], _market_rows(), ValueConfig(exchange_commission=commission), as_of=T0)
        return next(b for b in bets if b.market == "1x2" and b.selection == "home")

    assert home_bet(0.05).bookmaker == "Bet365"  # exchange 2.30 nets 2.235 after 5% commission < 2.25
    free = home_bet(0.0)
    assert free.bookmaker == EXCHANGE and free.effective_odds == pytest.approx(2.30)


def test_value_bet_filters():
    match = make_match("m1", date(2030, 1, 5), kickoff_at=KICKOFF)
    rows = _market_rows()
    bet365_only = find_value_bets([match], rows, ValueConfig(bookmakers=("Bet365",)), as_of=T0)
    assert {b.bookmaker for b in bet365_only} == {"Bet365"}
    assert {(b.market, b.selection) for b in bet365_only} == {("1x2", "home"), ("totals_2.5", "over")}
    home = next(b for b in bet365_only if b.selection == "home")
    assert home.expected_value == pytest.approx(2.25 * home.fair_probability - 1)
    assert home.reference == "Pinnacle" and home.key.startswith("vb:")
    assert "A vs B" in home.description
    assert not find_value_bets([match], rows, ValueConfig(markets=("ah",)), as_of=T0)
    assert not find_value_bets([match], rows, ValueConfig(bookmakers=("Bet365",), min_ev=0.2), as_of=T0)
    with_max = find_value_bets([match], rows, ValueConfig(include_market_max=True, max_ev=0.5), as_of=T0)
    assert any(b.bookmaker == MARKET_MAX for b in with_max)
    # prices captured after the cutoff and closing prices are ignored
    assert not find_value_bets([match], rows, as_of=T0 - timedelta(minutes=1))
    closing_only = [replace(r, is_closing=True) for r in rows]
    assert not find_value_bets([match], closing_only, as_of=T0)


def test_value_config_validation():
    for kwargs in (
        {"reference": "x"},
        {"markets": ("corners",)},
        {"max_stake": 0},
        {"exchange_commission": 1},
    ):
        with pytest.raises(ValueError):
            ValueConfig(**kwargs)


# --- ledger -----------------------------------------------------------------


def test_ledger_records_attaches_clv_settles_and_reports():
    db = ResearchDatabase()
    match = make_match("m1", date(2030, 1, 5), kickoff_at=KICKOFF, home_goals=None, away_goals=None)
    db.insert_matches([match])
    db.insert_odds(_market_rows())
    ledger = Ledger(db)
    bets = find_value_bets([match], db.load_odds(), ValueConfig(bookmakers=("Bet365",)), as_of=T0)
    ledger.save_candidates(bets)
    home = next(b for b in bets if b.selection == "home")
    stored = ledger.candidate(home.key)
    stored["key"] = home.key
    assert ledger.record_value_bet(stored, kind="real", stake=10.0, odds=2.2) is not None
    assert ledger.record_value_bet(stored, kind="real", stake=10.0) is None  # duplicate candidate
    ledger.record(
        kind="paper",
        match_id="m1",
        market="totals_2.5",
        selection="over",
        bookmaker="Bet365",
        odds=2.08,
        stake=5,
    )
    with pytest.raises(ValueError):
        ledger.record(
            kind="paper", match_id="m1", market="1x2", selection="over", bookmaker="X", odds=2.0, stake=1
        )

    assert ledger.update() == {"clv": 0, "settled": 0}  # nothing known yet
    db.insert_odds(snaps("m1", "Pinnacle", "1x2", x12(1.9, 3.7, 4.4), at=KICKOFF, closing=True))
    db.insert_matches([make_match("m1", date(2030, 1, 5), kickoff_at=KICKOFF, home_goals=2, away_goals=0)])
    assert ledger.update() == {"clv": 1, "settled": 2}
    real = ledger.bets(kind="real")[0]
    assert real.result == "win" and real.pnl == pytest.approx(12.0)
    assert real.closing_source == "Pinnacle"
    assert real.clv == pytest.approx(2.2 * real.closing_probability - 1)
    report = ledger.report()
    assert report["bets"] == 2 and report["settled"] == 2 and report["clv_n"] == 1
    assert "Not enough" in report["verdict"]
    assert set(report["by_market"]) == {"1x2", "totals"}


def test_scheduled_match_sync_never_erases_a_result():
    db = ResearchDatabase()
    db.insert_matches([make_match("m1", home_goals=2, away_goals=1)])
    db.insert_future_matches([make_match("m1", home_goals=None, away_goals=None)])
    assert db.load_matches()[0].home_goals == 2


# --- backtest ---------------------------------------------------------------


def test_value_backtest_settles_and_scores_against_closing():
    matches, odds = [], []
    for i in range(6):
        day = date(2024, 1, 1) + timedelta(days=i)
        pre = datetime.combine(day, datetime.min.time(), timezone.utc)
        m = make_match(f"g{i}", day, home_goals=i % 3, away_goals=1)
        matches.append(m)
        odds += snaps(m.match_id, "Pinnacle", "1x2", x12(2.0, 3.6, 4.0), at=pre)
        odds += snaps(m.match_id, "Bet365", "1x2", x12(2.3, 3.3, 3.5), at=pre)
        odds += snaps(
            m.match_id, "Pinnacle", "1x2", x12(1.95, 3.7, 4.2), at=pre + timedelta(hours=15), closing=True
        )
    result = run_value_backtest(matches, odds, ValueConfig(bookmakers=("Bet365",)))
    assert result.metrics["bets"] == 6
    assert all(s.bet.selection == "home" for s in result.bets)
    assert result.metrics["mean_clv"] == pytest.approx(2.3 * result.bets[0].closing_probability - 1)
    assert result.metrics["flat_profit"] == pytest.approx(sum(1.3 if i % 3 > 1 else -1 for i in range(6)))
    assert set(result.breakdown) == {"market", "reference", "bookmaker", "competition", "season"}
    assert run_value_backtest(
        matches, odds, ValueConfig(bookmakers=("Bet365",)), start=date(2025, 1, 1)
    ).metrics == {"bets": 0.0}


def test_exchange_commissions_and_consensus_excludes_exchanges():
    from parlay.betting.value import commission_for

    assert commission_for("Matchbook") == 0.02 and commission_for(EXCHANGE) == 0.05
    assert commission_for("Bet365", 0.1) == 0.0 and commission_for("Matchbook", 0.0) == 0.0
    assert effective_odds(3.0, "Matchbook") == pytest.approx(2.96)
    quotes = {b: x12(2.0, 3.5, 4.0) for b in ("Bet365", "bwin", "Unibet")} | {"Matchbook": x12(9.0, 9.0, 9.0)}
    assert fair_price(quotes, ("home", "draw", "away"), "consensus").books == 3


def test_latest_quotes_tolerates_split_timestamps_but_drops_stale_sides():
    from parlay.betting.value import latest_quotes

    t = datetime(2030, 1, 5, 9, tzinfo=timezone.utc)
    rows = [
        OddsSnapshot("m", "1xBet", "totals_2.5", "over", 1.9, t),
        OddsSnapshot("m", "1xBet", "totals_2.5", "under", 1.95, t + timedelta(microseconds=12)),
        OddsSnapshot("m", "Bet365", "totals_2.5", "over", 2.0, t - timedelta(days=1)),  # stale side
        OddsSnapshot("m", "Bet365", "totals_2.5", "under", 1.8, t),
    ]
    quotes = latest_quotes(rows, t + timedelta(hours=1))["totals_2.5"]
    assert quotes["1xBet"] == {"over": 1.9, "under": 1.95}
    assert quotes["Bet365"] == {"under": 1.8}
