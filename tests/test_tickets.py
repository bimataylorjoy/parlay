import math
from datetime import date, datetime, timedelta, timezone

import pytest

from parlay.betting import Ledger, ValueConfig, find_value_bets, run_value_backtest
from parlay.betting.backtest import run_parlay_backtest
from parlay.betting.ledger import settle_ticket
from parlay.betting.tickets import Leg, build_ticket, favourite_leg, price_leg, value_tickets
from parlay.cli import main
from parlay.data.database import ResearchDatabase
from parlay.data.loaders import match_id_for
from parlay.data.schemas import Match, OddsSnapshot

T0 = datetime(2030, 1, 5, 9, tzinfo=timezone.utc)


def leg(match_id, odds, p, market="1x2", selection="home"):
    return Leg(
        match_id,
        "EPL",
        date(2030, 1, 5),
        None,
        f"H{match_id}",
        f"A{match_id}",
        market,
        selection,
        odds,
        p,
        "Pinnacle",
    )


def test_ticket_maths_compounds_leg_values():
    ticket = build_ticket([leg("a", 2.1, 0.5), leg("b", 3.0, 0.35)], "1xBet")
    assert ticket.combined_odds == pytest.approx(6.3)
    assert ticket.win_probability == pytest.approx(0.175)
    assert ticket.expected_value == pytest.approx(6.3 * 0.175 - 1)
    assert ticket.expected_value == pytest.approx((1 + 0.05) * (1 + 0.05) - 1)
    boosted = build_ticket(ticket.legs, "1xBet", boost=0.10)
    assert boosted.combined_odds == pytest.approx(6.93)
    assert build_ticket(ticket.legs, "1xBet", ticket_odds=7.0).combined_odds == 7.0
    assert ticket.stake_fraction() > 0
    assert ticket.key.startswith("tk:") and ticket.key == build_ticket(ticket.legs[::-1], "1xBet").key


def test_negative_legs_are_flagged_and_kill_the_ticket():
    ticket = build_ticket([leg("a", 1.9, 0.5), leg("b", 1.9, 0.5), leg("c", 1.9, 0.5)], "1xBet")
    assert ticket.expected_value == pytest.approx(0.95**3 - 1)
    assert len(ticket.warnings) == 3 and ticket.stake_fraction() == 0.0


def test_more_legs_shrink_the_kelly_stake():
    two = build_ticket([leg("a", 2.2, 0.5), leg("b", 2.2, 0.5)], "1xBet")
    assert two.stake_fraction(leg_sd=0.05) < two.stake_fraction(leg_sd=0.0)


@pytest.mark.parametrize(
    "legs,bookmaker,message",
    [
        ([leg("a", 2.0, 0.5)], "1xBet", "at least two"),
        ([leg("a", 2.0, 0.5), leg("a", 3.0, 0.3, "totals_2.5", "over")], "1xBet", "same-match"),
        ([leg("a", 2.0, 0.5), leg("b", 2.0, 0.5)], "Betfair Exchange", "exchange"),
    ],
)
def test_invalid_tickets(legs, bookmaker, message):
    with pytest.raises(ValueError, match=message):
        build_ticket(legs, bookmaker)


def test_unpriced_leg_makes_ev_unknown():
    ticket = build_ticket([leg("a", 2.0, 0.5), leg("b", 2.0, None)], "1xBet")
    assert ticket.expected_value is None and ticket.stake_fraction() == 0.0 and ticket.warnings


def _db_with_matches(n=3):
    db = ResearchDatabase()
    matches, odds = [], []
    for i in range(n):
        kickoff = T0 + timedelta(hours=6 + i)
        match_id = match_id_for(kickoff.date(), f"H{i}", f"A{i}", "EPL")
        matches.append(
            Match(
                match_id, kickoff.date(), "EPL", "2029-30", f"H{i}", f"A{i}", None, None, kickoff_at=kickoff
            )
        )
        for book, values in {"Pinnacle": (2.0, 3.6, 4.0), "1xBet": (2.1 + 0.05 * i, 3.3, 3.5)}.items():
            for selection, price in zip(("home", "draw", "away"), values, strict=True):
                odds.append(OddsSnapshot(match_id, book, "1x2", selection, price, T0))
    db.insert_future_matches(matches)
    db.insert_odds(odds)
    return db, matches


def test_price_and_favourite_legs():
    db, matches = _db_with_matches(1)
    odds = db.load_odds()
    priced = price_leg(matches[0], "1x2", "home", odds, bookmaker="1xBet", as_of=T0)
    assert priced.odds == 2.1 and priced.reference == "Pinnacle" and priced.expected_value > 0
    assert price_leg(matches[0], "1x2", "home", odds, bookmaker="1xBet", as_of=T0, leg_odds=2.5).odds == 2.5
    assert favourite_leg(matches[0], odds, bookmaker="1xBet", as_of=T0).selection == "home"
    with pytest.raises(ValueError, match="no Bet365 price"):
        price_leg(matches[0], "1x2", "home", odds, bookmaker="Bet365", as_of=T0)


def test_value_tickets_use_one_leg_per_match_per_bookmaker():
    db, matches = _db_with_matches(3)
    bets = find_value_bets(matches, db.load_odds(), ValueConfig(bookmakers=("1xBet",)), as_of=T0)
    tickets = value_tickets(bets, max_legs=2)
    assert len(tickets) == 1 and len(tickets[0].legs) == 2
    assert tickets[0].expected_value > max(b.expected_value for b in bets)


@pytest.mark.parametrize(
    "results,label,profit",
    [
        (["win", "win"], "win", 10 * (2.0 * 3.0) - 10),
        (["win", "loss"], "loss", -10),
        (["win", "push"], "win", 10 * 2.0 - 10),
        (["half_win", "win"], "win", 10 * 1.5 * 3.0 - 10),
        (["half_loss", "win"], "win", 10 * 0.5 * 3.0 - 10),
        (["push", "push"], "void", 0.0),
    ],
)
def test_ticket_settlement(results, label, profit):
    legs = [{"result": r, "odds": o} for r, o in zip(results, (2.0, 3.0), strict=True)]
    assert settle_ticket(6.0, 10.0, legs) == (label, pytest.approx(profit))


def test_ticket_settlement_carries_a_boost():
    legs = [{"result": "win", "odds": 2.0}, {"result": "win", "odds": 3.0}]
    assert settle_ticket(6.6, 10.0, legs)[1] == pytest.approx(56.0)


def test_ledger_records_updates_and_reports_tickets():
    db, matches = _db_with_matches(2)
    ledger = Ledger(db)
    ticket = build_ticket(
        [price_leg(m, "1x2", "home", db.load_odds(), bookmaker="1xBet", as_of=T0) for m in matches], "1xBet"
    )
    ticket_id = ledger.record_ticket(ticket, kind="paper", stake=10.0)
    assert ticket_id and ledger.record_ticket(ticket, kind="paper", stake=10.0) is None
    for m in matches:
        db.insert_odds([OddsSnapshot(m.match_id, "Pinnacle", "1x2", s, p, m.kickoff_at, True)
                        for s, p in zip(("home", "draw", "away"), (1.9, 3.7, 4.4), strict=True)])  # fmt: skip
    db.insert_matches(
        [
            Match(
                matches[0].match_id,
                matches[0].date,
                "EPL",
                "x",
                "H0",
                "A0",
                1,
                0,
                kickoff_at=matches[0].kickoff_at,
            )
        ]
    )
    assert ledger.update_tickets() == {"clv": 1, "settled": 0}  # second leg not played yet
    db.insert_matches(
        [
            Match(
                matches[1].match_id,
                matches[1].date,
                "EPL",
                "x",
                "H1",
                "A1",
                2,
                1,
                kickoff_at=matches[1].kickoff_at,
            )
        ]
    )
    assert ledger.update_tickets() == {"clv": 0, "settled": 1}
    stored = ledger.tickets()[0]
    assert stored["result"] == "win"
    assert stored["pnl"] == pytest.approx(10 * ticket.combined_odds - 10)
    assert stored["clv"] == pytest.approx(ticket.combined_odds * stored["closing_probability"] - 1)
    report = ledger.report()["parlays"]
    assert report["tickets"] == 1 and report["won"] == 1 and report["clv_n"] == 1


def test_parlay_backtest_combines_same_day_value_bets():
    matches, odds = [], []
    for day in range(3):
        for i in range(2):
            d = date(2024, 1, 1) + timedelta(days=day)
            pre = datetime.combine(d, datetime.min.time(), timezone.utc)
            m = Match(f"d{day}m{i}", d, "EPL", "2023-24", f"H{i}", f"A{i}", 1, 0)
            matches.append(m)
            odds += [OddsSnapshot(m.match_id, b, "1x2", s, p, pre)
                     for b, vals in {"Pinnacle": (2.0, 3.6, 4.0), "Bet365": (2.3, 3.3, 3.5)}.items()
                     for s, p in zip(("home", "draw", "away"), vals, strict=True)]  # fmt: skip
    singles = run_value_backtest(matches, odds, ValueConfig(bookmakers=("Bet365",)))
    parlays = run_parlay_backtest(singles)
    m = parlays.metrics
    assert m["tickets"] == 3 and m["mean_legs"] == 2
    assert m["hit_rate"] == 1.0 and m["yield"] == pytest.approx(2.3 * 2.3 - 1)
    assert m["mean_expected_value"] == pytest.approx(
        math.prod([1 + singles.bets[0].bet.expected_value] * 2) - 1
    )


def test_ticket_cli(tmp_path, capsys):
    path = tmp_path / "data.sqlite"
    db, matches = _db_with_matches(3)
    disk = ResearchDatabase(str(path))
    disk.insert_future_matches(matches)
    disk.insert_odds(db.load_odds())
    disk.close()
    as_of = ["--as-of", T0.isoformat(), "--database", str(path)]
    assert main(["ticket", "--favourites", *as_of]) == 0
    out = capsys.readouterr().out
    assert "3 legs" in out and "positive expected value" in out
    assert (
        main(
            ["ticket", "--leg", "H0 vs A0:1x2:2", "--leg", "H1 vs A1:1x2:X@3.4", *as_of, "--record", "paper"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Do not bet" in out and "Recorded paper parlay #1" in out
    with pytest.raises(SystemExit, match="same-match"):
        main(["ticket", "--leg", "H0 vs A0:1x2:1", "--leg", "H0 vs A0:1x2:2", *as_of])
    with pytest.raises(SystemExit, match="no upcoming match"):
        main(["ticket", "--leg", "Foo vs Bar:1x2:1", "--leg", "H0 vs A0:1x2:2", *as_of])
    assert main(["value", "--parlay", "--paper", "--books", "1xBet", *as_of]) == 0
    assert "Mix parlays from value bets" in capsys.readouterr().out
    assert main(["bets", "list", "--database", str(path)]) == 0
    assert "Parlay #1" in capsys.readouterr().out
    assert main(["bets", "report", "--database", str(path)]) == 0
    assert "Parlays 2" in capsys.readouterr().out
