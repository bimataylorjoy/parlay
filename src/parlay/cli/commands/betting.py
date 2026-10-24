"""``parlay odds | value | bets | backtest-value``: price-based value betting with a CLV ledger."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from parlay.betting import Ledger, ValueConfig, find_value_bets, run_value_backtest
from parlay.betting.reference import REFERENCE_POLICIES
from parlay.betting.value import MARKET_KINDS, ValueBet, commission_for, describe_pick
from parlay.cli._common import DEFAULT_SNAPSHOT_DIR, add_database, iso_date
from parlay.cli.render import table
from parlay.data.database import ResearchDatabase
from parlay.data.odds_sync import sync_football_data_fixtures, sync_the_odds_api
from parlay.evaluation.serialization import write_json, write_rows

LOW_CONFIDENCE = (
    "Warning: reference '{ref}' is not Pinnacle. In the 2024-26 backtest, bets priced against the "
    "Betfair Exchange snapshot or the consensus had negative or zero closing-line value; "
    "treat them as low confidence."
)


def _utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an ISO datetime, got {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _books(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(book.strip() for book in value.split(",") if book.strip())


def _add_value_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("value")
    group.add_argument(
        "--reference",
        choices=REFERENCE_POLICIES,
        default="pinnacle",
        help="fair-price source (default: pinnacle)",
    )
    group.add_argument(
        "--books",
        default=os.environ.get("PARLAY_BOOKS"),
        help="comma-separated bookmakers you can bet with (default: $PARLAY_BOOKS, else all), "
        'e.g. "Bet365,William Hill"',
    )
    group.add_argument("--markets", nargs="+", choices=MARKET_KINDS, default=list(MARKET_KINDS))
    group.add_argument("--min-ev", type=float, default=0.02, help="minimum expected value (default: 0.02)")
    group.add_argument(
        "--max-ev", type=float, default=0.30, help="ignore larger 'edges' as stale prices (default: 0.30)"
    )
    group.add_argument("--min-odds", type=float, default=1.15)
    group.add_argument("--max-odds", type=float, default=10.0)
    group.add_argument(
        "--commission",
        type=float,
        default=None,
        help="exchange commission (default: Betfair 0.05, Matchbook 0.02)",
    )
    group.add_argument("--kelly-fraction", type=float, default=0.25)
    group.add_argument("--max-stake", type=float, default=0.02, help="bankroll cap per bet (default: 0.02)")
    group.add_argument(
        "--probability-sd", type=float, default=0.02, help="uncertainty of fair prices for Kelly shrinkage"
    )
    group.add_argument(
        "--include-market-max", action="store_true", help="allow football-data's best price as a venue"
    )


def _value_config(args: argparse.Namespace) -> ValueConfig:
    return ValueConfig(
        reference=args.reference,
        bookmakers=_books(args.books),
        markets=tuple(args.markets),
        min_ev=args.min_ev,
        max_ev=args.max_ev,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        exchange_commission=args.commission,
        kelly_fraction=args.kelly_fraction,
        max_stake=args.max_stake,
        probability_sd=args.probability_sd,
        include_market_max=args.include_market_max,
    )


def register(commands: argparse._SubParsersAction) -> None:
    daily = commands.add_parser(
        "daily",
        help="daily routine: results, odds sync, CLV/settlement, value bets (paper), report",
    )
    add_database(daily)
    daily.add_argument("--league", nargs="+", default=["E0", "E1"], help="football-data division codes")
    daily.add_argument("--books", default=os.environ.get("PARLAY_BOOKS"), help="bookmakers you can bet with")
    daily.add_argument("--min-ev", type=float, default=0.02)
    daily.add_argument("--bankroll", type=float, default=1000.0)
    daily.add_argument("--days", type=int, default=3, help="value bets for matches within this many days")
    daily.add_argument("--skip-results", action="store_true", help="do not download current-season results")
    daily.add_argument("--no-paper", action="store_true", help="do not log value bets as paper bets")
    daily.set_defaults(handler=run_daily)

    ticket = commands.add_parser(
        "ticket",
        help="price a mix parlay (accumulator) against Pinnacle fair probabilities",
        description="Legs: value-bet keys (vb:...) and/or --leg 'Home vs Away:MARKET:SELECTION' "
        "(MARKET = 1x2, totals_2.5, ah_-0.75; SELECTION = home/draw/away, 1/X/2, over/under). "
        "--favourites adds the 1X2 favourite of every upcoming match.",
    )
    add_database(ticket)
    ticket.add_argument("keys", nargs="*", help="value-bet keys printed by `parlay value`")
    ticket.add_argument("--leg", action="append", default=[], help="'Home vs Away:MARKET:SELECTION[@odds]'")
    ticket.add_argument("--favourites", action="store_true", help="add the favourite of every upcoming match")
    ticket.add_argument("--days", type=int, default=3, help="window for --favourites (default: 3 days)")
    ticket.add_argument("--max-legs", type=int, default=10, help="cap on --favourites legs (default: 10)")
    ticket.add_argument("--book", default="1xBet", help="bookmaker of the ticket (default: 1xBet)")
    ticket.add_argument("--reference", choices=REFERENCE_POLICIES, default="pinnacle")
    ticket.add_argument(
        "--boost", type=float, default=0.0, help="accumulator odds boost, e.g. 0.10 for +10%%"
    )
    ticket.add_argument(
        "--ticket-odds", type=float, help="total odds shown on the betslip (overrides the product)"
    )
    ticket.add_argument("--bankroll", type=float, default=1000.0)
    ticket.add_argument("--kelly-fraction", type=float, default=0.25)
    ticket.add_argument("--max-stake", type=float, default=0.02)
    ticket.add_argument("--record", choices=("paper", "real"), help="store the ticket in the ledger")
    ticket.add_argument("--stake", type=float, help="stake to record (default: the Kelly suggestion)")
    ticket.add_argument("--as-of", type=_utc_datetime, help="price as of this UTC time (default: now)")
    ticket.set_defaults(handler=run_ticket)

    watch = commands.add_parser(
        "watch",
        help="re-price every kickoff slot ~50 min before kickoff (after lineups) and report value bets",
    )
    add_database(watch)
    _add_value_options(watch)
    watch.add_argument("--competition", nargs="+", default=["EPL", "Championship"])
    watch.add_argument("--lead", type=int, default=50, help="minutes before kickoff to sync (default: 50)")
    watch.add_argument("--min-lead", type=int, default=10, help="skip slots closer than this to kickoff")
    watch.add_argument("--regions", default="eu")
    watch.add_argument("--api-markets", default="h2h,totals")
    watch.add_argument("--credit-reserve", type=int, default=20, help="stop syncing below this many credits")
    watch.add_argument("--bankroll", type=float, default=1000.0)
    watch.add_argument("--no-paper", action="store_true", help="do not log value bets as paper bets")
    watch.add_argument("--notify-empty", action="store_true", help="also notify slots without value bets")
    watch.add_argument(
        "--once", action="store_true", help="process due slots and exit (for cron every 5 min)"
    )
    watch.add_argument(
        "--next", action="store_true", help="show upcoming slots and credit estimate, then exit"
    )
    watch.add_argument("--until", type=_utc_datetime, help="stop the loop at this UTC time")
    watch.set_defaults(handler=run_watch)

    odds = commands.add_parser("odds", help="download upcoming fixtures and prices")
    odds_commands = odds.add_subparsers(dest="odds_command", required=True, metavar="<action>")
    sync = odds_commands.add_parser("sync", help="store the latest prices for upcoming matches")
    add_database(sync)
    sync.add_argument(
        "--source",
        choices=("football-data", "the-odds-api"),
        default="football-data",
        help="football-data fixtures.csv (free, no key) or The Odds API (free key, includes Pinnacle)",
    )
    sync.add_argument("--league", nargs="+", default=["E0", "E1"], help="football-data division codes")
    sync.add_argument(
        "--competition", nargs="+", default=["EPL", "Championship"], help="The Odds API competitions"
    )
    sync.add_argument("--api-key", default=None, help="The Odds API key (prefer $THE_ODDS_API_KEY)")
    sync.add_argument(
        "--regions", default="eu", help="The Odds API regions; 'eu' includes Pinnacle (default: eu)"
    )
    sync.add_argument(
        "--api-markets", default="h2h,totals", help="h2h,totals[,spreads]; cost = markets x regions"
    )
    sync.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    sync.set_defaults(handler=run_odds_sync)

    value = commands.add_parser("value", help="list value bets for upcoming matches from stored prices")
    add_database(value)
    _add_value_options(value)
    value.add_argument("--days", type=int, default=7, help="look this many days ahead (default: 7)")
    value.add_argument(
        "--bankroll", type=float, default=1000.0, help="bankroll used to size stakes (default: 1000)"
    )
    value.add_argument("--paper", action="store_true", help="record every value bet as a paper bet")
    value.add_argument(
        "--as-of", type=_utc_datetime, help="evaluate as of this UTC time (replay); default: now"
    )
    value.add_argument("--output", type=Path, help="write the value bets to CSV")
    value.add_argument(
        "--parlay", action="store_true", help="also build mix-parlay tickets from the value bets"
    )
    value.add_argument("--max-legs", type=int, default=3, help="legs per suggested ticket (default: 3)")
    value.set_defaults(handler=run_value)

    bets = commands.add_parser("bets", help="bet ledger: add, update with closing prices/results, report")
    bets_commands = bets.add_subparsers(dest="bets_command", required=True, metavar="<action>")
    add = bets_commands.add_parser("add", help="record a bet (by value-bet key or manually)")
    add_database(add)
    add.add_argument("key", nargs="?", help="value-bet key printed by `parlay value`, e.g. vb:1a2b3c4d")
    add.add_argument("--stake", type=float, required=True)
    add.add_argument("--odds", type=float, help="price actually taken (default: the quoted price)")
    add.add_argument("--paper", action="store_true", help="record as a paper bet (default: real)")
    add.add_argument(
        "--commission", type=float, default=None, help="exchange commission (default: per exchange)"
    )
    manual = add.add_argument_group("manual entry (without a key)")
    manual.add_argument("--match-id")
    manual.add_argument("--market", help="1x2, totals_2.5, ah_-0.75, ...")
    manual.add_argument("--selection", help="home/draw/away, over/under")
    manual.add_argument("--bookmaker")
    add.set_defaults(handler=run_bets_add)
    update = bets_commands.add_parser(
        "update", help="attach closing prices (CLV) and settle finished matches"
    )
    add_database(update)
    update.set_defaults(handler=run_bets_update)
    report = bets_commands.add_parser("report", help="profit, yield and closing-line value")
    add_database(report)
    report.add_argument("--kind", choices=("paper", "real"))
    report.set_defaults(handler=run_bets_report)
    listing = bets_commands.add_parser("list", help="list recorded bets")
    add_database(listing)
    listing.add_argument("--kind", choices=("paper", "real"))
    listing.add_argument("--open", action="store_true", help="only unsettled bets")
    listing.set_defaults(handler=run_bets_list)

    anchor = commands.add_parser(
        "anchor-test",
        help="does the goal model (or other features) add information to the Pinnacle price?",
    )
    add_database(anchor)
    anchor.add_argument(
        "predictions", type=Path, help="predictions.csv from `parlay backtest` (out-of-sample)"
    )
    anchor.add_argument("--l2", type=float, default=1.0, help="L2 penalty on the adjustment weights")
    anchor.add_argument("--min-ev", type=float, default=0.02)
    anchor.add_argument(
        "--books", default=os.environ.get("PARLAY_BOOKS"), help="bookmakers used in the bet simulation"
    )
    anchor.set_defaults(handler=run_anchor_test)

    backtest = commands.add_parser("backtest-value", help="replay the value engine on historical prices")
    add_database(backtest)
    _add_value_options(backtest)
    backtest.add_argument("--from", dest="start", type=iso_date, help="first match date")
    backtest.add_argument("--to", dest="end", type=iso_date, help="last match date")
    backtest.add_argument("--output-dir", type=Path, default=Path("outputs/value_backtest"))
    backtest.add_argument(
        "--parlay", action="store_true", help="also test daily mix parlays from the value bets"
    )
    backtest.add_argument("--max-legs", type=int, default=3, help="legs per parlay ticket (default: 3)")
    backtest.set_defaults(handler=run_backtest_value)


def _open_database(path: str) -> ResearchDatabase:
    if not Path(path).exists():
        raise SystemExit(f"Database {path} not found; run `parlay fetch` first")
    return ResearchDatabase(path)


def run_odds_sync(args: argparse.Namespace) -> int:
    now = datetime.now(timezone.utc)
    database = ResearchDatabase(args.database)
    try:
        if args.source == "football-data":
            result = sync_football_data_fixtures(
                database, args.league, now=now, snapshot_dir=args.snapshot_dir
            )
            note = (
                "football-data fixtures.csv has no Pinnacle prices; "
                "also sync --source the-odds-api for a sharp reference."
            )
        else:
            api_key = args.api_key or os.environ.get("THE_ODDS_API_KEY")
            if not api_key:
                raise SystemExit("Set THE_ODDS_API_KEY (free at https://the-odds-api.com) or pass --api-key")
            result = sync_the_odds_api(
                database, api_key, args.competition, now=now, regions=args.regions, markets=args.api_markets
            )
            note = f"The Odds API credits remaining: {result.credits_remaining}"
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        database.close()
    print(
        f"Stored {result.matches} upcoming matches and {result.prices} prices "
        f"from {len(result.bookmakers)} bookmakers."
    )
    if result.bookmakers:
        print("Bookmakers: " + ", ".join(result.bookmakers))
    print(note)
    return 0


def _value_rows(bets: list[ValueBet], bankroll: float) -> list[list[str]]:
    rows = []
    for bet in bets:
        kickoff = bet.kickoff_at.strftime("%a %d %b %H:%M") if bet.kickoff_at else bet.match_date.isoformat()
        rows.append(
            [
                bet.key,
                kickoff,
                bet.description,
                bet.bookmaker,
                f"{bet.odds:.2f}",
                f"{bet.fair_odds:.2f}",
                f"{bet.expected_value:+.1%}",
                f"{bet.stake_fraction:.2%} ({bet.stake_fraction * bankroll:.2f})",
                bet.reference,
            ]
        )
    return rows


def run_value(args: argparse.Namespace) -> int:
    database = _open_database(args.database)
    now = args.as_of or datetime.now(timezone.utc)
    horizon = now + timedelta(days=args.days)

    def upcoming(match) -> bool:
        start = match.kickoff_at or datetime.combine(match.date, datetime.max.time(), timezone.utc)
        return match.home_goals is None and now < start <= horizon

    matches = [m for m in database.load_matches() if upcoming(m)]
    odds = [o for m in matches for o in database.load_odds(match_id=m.match_id)]
    config = _value_config(args)
    bets = find_value_bets(matches, odds, config, as_of=now)
    ledger = Ledger(database)
    ledger.save_candidates(bets)
    print(f"{len(matches)} upcoming matches with {len(odds)} stored prices; {len(bets)} value bets.\n")
    if not odds:
        print("No prices stored yet: run `parlay odds sync` (and `--source the-odds-api` for Pinnacle).")
    if bets:
        print(
            table(
                ["key", "kickoff (UTC)", "bet", "book", "odds", "fair", "EV", "stake", "reference"],
                _value_rows(bets, args.bankroll),
                "lllrrrrrl",
            )
        )
        print("\nRecord a bet you placed:  parlay bets add <key> --stake <amount> [--odds <price taken>]")
    if config.reference != "pinnacle" or any(bet.reference != "Pinnacle" for bet in bets):
        print("\n" + LOW_CONFIDENCE.format(ref=config.reference))
    if args.paper:
        recorded = 0
        for bet in bets:
            if ledger.has_open_bet("paper", bet.match_id, bet.market, bet.selection):
                continue  # one paper bet per selection: re-runs after price moves must not double count
            recorded += (
                ledger.record_value_bet(
                    bet,
                    kind="paper",
                    stake=max(bet.stake_fraction * args.bankroll, 0.01),
                    commission=commission_for(bet.bookmaker, config.exchange_commission),
                )
                is not None
            )
        print(f"\nRecorded {recorded} new paper bets.")
    if args.parlay:
        _print_value_tickets(ledger, bets, args, config)
    if args.output:
        write_rows([{"key": bet.key, **asdict(bet)} for bet in bets], args.output)
    database.close()
    return 0


def run_bets_add(args: argparse.Namespace) -> int:
    database = _open_database(args.database)
    ledger = Ledger(database)
    kind = "paper" if args.paper else "real"
    if args.key:
        candidate = ledger.candidate(args.key)
        candidate["key"] = args.key
        commission = commission_for(candidate["bookmaker"], args.commission)
        bet_id = ledger.record_value_bet(
            candidate, kind=kind, stake=args.stake, odds=args.odds, commission=commission
        )
    else:
        missing = [
            name
            for name in ("match_id", "market", "selection", "bookmaker", "odds")
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(f"without a key, provide --{' --'.join(m.replace('_', '-') for m in missing)}")
        match = next((m for m in database.load_matches() if m.match_id == args.match_id), None)
        commission = commission_for(args.bookmaker, args.commission)
        bet_id = ledger.record(
            kind=kind,
            match_id=args.match_id,
            market=args.market,
            selection=args.selection,
            bookmaker=args.bookmaker,
            odds=args.odds,
            stake=args.stake,
            commission=commission,
            match=match,
        )
    database.close()
    print(f"Recorded {kind} bet #{bet_id}." if bet_id else "This bet was already recorded.")
    return 0


def run_bets_update(args: argparse.Namespace) -> int:
    database = _open_database(args.database)
    ledger = Ledger(database)
    counts = ledger.update()
    ticket_counts = ledger.update_tickets()
    database.close()
    print(f"Closing prices attached to {counts['clv']} bets; {counts['settled']} bets settled.")
    if ticket_counts["clv"] or ticket_counts["settled"]:
        print(f"Parlays: CLV for {ticket_counts['clv']} tickets; {ticket_counts['settled']} tickets settled.")
    print(
        "Results arrive via `parlay fetch --to-season <current>`; "
        "closing prices via the same files or an odds sync."
    )
    return 0


def run_bets_report(args: argparse.Namespace) -> int:
    database = _open_database(args.database)
    report = Ledger(database).report(kind=args.kind)
    database.close()
    parlays = report["parlays"]
    if parlays["tickets"]:
        print(
            f"Parlays {parlays['tickets']} ({parlays['open']} open, {parlays['won']} won) · "
            f"staked {parlays['staked']:.2f} · profit {parlays['profit']:+.2f} · "
            f"yield {parlays['yield']:+.2%} ± {parlays['yield_se']:.2%} · "
            f"CLV {parlays['mean_clv']:+.2%} ± {parlays['clv_se']:.2%} (n={parlays['clv_n']})\n"
        )
    if not report["bets"]:
        print("No single bets recorded yet.")
        return 0
    print(
        f"Bets {report['bets']} ({report['open']} open) · staked {report['staked']:.2f} · "
        f"profit {report['profit']:+.2f} · yield {report['yield']:+.2%} ± {report['yield_se']:.2%}"
    )
    print(
        f"Closing-line value: {report['mean_clv']:+.2%} ± {report['clv_se']:.2%} over {report['clv_n']} bets"
    )
    rows = [
        [
            kind,
            f"{v['bets']:.0f}",
            f"{v['settled']:.0f}",
            f"{v['yield']:+.2%}",
            f"{v['mean_clv']:+.2%} ± {v['clv_se']:.2%}",
        ]
        for kind, v in report["by_market"].items()
    ]
    print(table(["market", "bets", "settled", "yield", "CLV"], rows))
    print("\n" + report["verdict"])
    return 0


def run_bets_list(args: argparse.Namespace) -> int:
    database = _open_database(args.database)
    ledger = Ledger(database)
    rows = ledger.bets(kind=args.kind, open_only=args.open)
    tickets = ledger.tickets(kind=args.kind, open_only=args.open)
    database.close()
    for t in tickets:
        clv = f"{t['clv']:+.1%}" if t["clv"] is not None else "-"
        pnl = f"{t['pnl']:+.2f}" if t["pnl"] is not None else "-"
        print(
            f"Parlay #{t['ticket_id']} {t['kind']} {t['bookmaker']} @ {t['odds']:.2f} stake {t['stake']:.2f} "
            f"CLV {clv} · {t['result'] or 'open'} {pnl}"
        )
        for leg in t["leg_rows"]:
            pick = describe_pick(leg["home_team"], leg["away_team"], leg["market"], leg["selection"])
            print(f"   {leg['leg_no']}. {pick} @ {leg['odds']:.2f} · {leg['result'] or 'open'}")
    if tickets:
        print()
    if not rows:
        print("No single bets.")
        return 0
    print(
        table(
            ["#", "kind", "match", "market", "pick", "book", "odds", "stake", "CLV", "result", "P/L"],
            [
                [
                    str(b.bet_id),
                    b.kind,
                    f"{b.home_team} vs {b.away_team}" if b.home_team else b.match_id,
                    b.market,
                    b.selection,
                    b.bookmaker,
                    f"{b.odds:.2f}",
                    f"{b.stake:.2f}",
                    f"{b.clv:+.1%}" if b.clv is not None else "-",
                    b.result or "open",
                    f"{b.pnl:+.2f}" if b.pnl is not None else "-",
                ]
                for b in rows
            ],
            "lllllrrrrlr",
        )
    )
    return 0


def run_backtest_value(args: argparse.Namespace) -> int:
    database = _open_database(args.database)
    matches, odds = database.load_matches(), database.load_odds()
    database.close()
    config = _value_config(args)
    result = run_value_backtest(matches, odds, config, start=args.start, end=args.end)
    m = result.metrics
    if not m.get("bets"):
        print("No value bets found for this configuration and period.")
        completed = {mt.match_id for mt in matches if mt.home_goals is not None}
        known = {o.bookmaker for o in odds if o.match_id in completed}
        venues = set(config.bookmakers or ()) - {"Pinnacle"}  # the reference is never a betting venue
        if config.bookmakers and not venues & known:
            print(
                f"None of your bookmakers ({', '.join(config.bookmakers)}) appear in the historical prices "
                f"(available: {', '.join(sorted(known - {'market_max', 'market_avg'}))}). "
                'Run with --books "" to test every bookmaker.'
            )
        return 0
    print(
        f"{m['bets']:.0f} bets on {m['matches']:.0f} matches · mean odds {m['mean_odds']:.2f} · "
        f"expected EV {m['mean_expected_value']:+.2%}"
    )
    print(f"Flat 1-unit yield {m['yield']:+.2%} ± {m['yield_se']:.2%} · profit {m['flat_profit']:+.1f} units")
    print(f"Closing-line value {m['mean_clv']:+.2%} ± {m['clv_se']:.2%} (n={m['clv_n']:.0f})")
    print(f"Fractional-Kelly bankroll multiple {m['kelly_bankroll_multiple']:.3f}\n")
    for grouping in ("market", "reference", "season"):
        rows = [
            [
                label,
                f"{v['bets']:.0f}",
                f"{v['yield']:+.2%} ± {v['yield_se']:.2%}",
                f"{v['mean_clv']:+.2%} ± {v['clv_se']:.2%}",
            ]
            for label, v in result.breakdown[grouping].items()
            if v.get("bets")
        ]
        print(table([grouping, "bets", "yield", "CLV"], rows))
        print()
    write_rows(
        [
            {
                **asdict(s.bet),
                "result": s.result,
                "flat_profit": s.flat_profit,
                "closing_probability": s.closing_probability,
                "clv": s.clv,
            }
            for s in result.bets
        ],
        args.output_dir / "bets.csv",
    )
    report: dict[str, object] = {"metrics": m, "breakdown": result.breakdown, "config": asdict(config)}
    if args.parlay:
        report["parlays"] = _parlay_backtest_report(result, args, config)
    write_json(report, args.output_dir / "metrics.json")
    print(f"Artifacts written to {args.output_dir}/")
    if config.reference != "pinnacle":
        print(LOW_CONFIDENCE.format(ref=config.reference))
    return 0


def run_daily(args: argparse.Namespace) -> int:
    """Chain the daily commands; The Odds API is used when THE_ODDS_API_KEY is set."""
    from parlay.cli import main as cli_main
    from parlay.data.loaders import season_for
    from parlay.data.sources import FOOTBALL_DATA_LEAGUES as LEAGUES

    database = ["--database", args.database]
    season = int(season_for(datetime.now(timezone.utc).date())[:4])
    steps: list[tuple[str, list[str]]] = []
    if not args.skip_results:
        steps.append(
            (
                "Results and closing prices",
                [
                    "fetch",
                    "--league",
                    *args.league,
                    "--from-season",
                    str(season),
                    "--to-season",
                    str(season),
                    *database,
                ],
            )
        )
    steps.append(
        ("Prices: football-data fixtures.csv", ["odds", "sync", "--league", *args.league, *database])
    )
    competitions = [LEAGUES[code] for code in args.league if code in LEAGUES]
    if os.environ.get("THE_ODDS_API_KEY"):
        steps.append(
            (
                "Prices: The Odds API (Pinnacle)",
                ["odds", "sync", "--source", "the-odds-api", "--competition", *competitions, *database],
            )
        )
    else:
        print("THE_ODDS_API_KEY is not set: no Pinnacle reference, so few or no value bets will be found.")
    steps.append(("Closing-line value and settlement", ["bets", "update", *database]))
    value = [
        "value",
        "--days",
        str(args.days),
        "--min-ev",
        str(args.min_ev),
        "--bankroll",
        str(args.bankroll),
    ]
    if args.books:
        value += ["--books", args.books]
    if not args.no_paper:
        value.append("--paper")
    steps.append(("Value bets", [*value, *database]))
    steps.append(("Paper-trading report", ["bets", "report", "--kind", "paper", *database]))
    for title, argv in steps:
        print(f"\n=== {title} ===")
        try:
            cli_main(argv)
        except (OSError, SystemExit) as exc:
            if isinstance(exc, SystemExit) and exc.code in (0, None):
                continue
            print(f"step failed: {exc}")
    return 0


def run_anchor_test(args: argparse.Namespace) -> int:
    import csv

    from parlay.betting.anchored import FEATURES, build_rows, evaluate_anchored

    if not args.predictions.exists():
        raise SystemExit(f"{args.predictions} not found; run `parlay backtest --output-dir ...` first")
    with args.predictions.open(encoding="utf-8") as handle:
        model = {
            row["match_id"]: (float(row["home_win"]), float(row["draw"]), float(row["away_win"]))
            for row in csv.DictReader(handle)
        }
    database = _open_database(args.database)
    rows = build_rows(database.load_matches(), database.load_odds(), model, bettable=_books(args.books))
    database.close()
    report = evaluate_anchored(rows, l2=args.l2, min_ev=args.min_ev)
    if not report.pooled:
        print("Not enough seasons with Pinnacle prices and model forecasts.")
        return 0
    print(f"{len(rows)} matches with a Pinnacle pre-match price and an out-of-sample model forecast.\n")
    print(
        table(
            ["season", "n", "market LL", "anchored LL", "gain", *FEATURES],
            [
                [
                    str(r["season"]),
                    f"{r['n']:.0f}",
                    f"{r['market_log_loss']:.4f}",
                    f"{r['anchored_log_loss']:.4f}",
                    f"{r['improvement']:+.4f} ± {r['improvement_se']:.4f}",
                    *(f"{report.weights[str(r['season'])][f]:+.3f}" for f in FEATURES),
                ]
                for r in report.seasons
            ],
        )
    )
    p = report.pooled
    print(
        f"\nPooled out-of-sample: market {p['market_log_loss']:.4f} vs anchored {p['anchored_log_loss']:.4f} "
        f"(gain {p['improvement']:+.4f} ± {p['improvement_se']:.4f}, n={p['n']:.0f})"
    )
    for prefix in ("market", "anchored"):
        bets, yld, yse = p[f"{prefix}_bets"], p[f"{prefix}_yield"], p[f"{prefix}_yield_se"]
        clv, cse = p[f"{prefix}_mean_clv"], p[f"{prefix}_clv_se"]
        print(
            f"Bets priced with {prefix:<8}: {bets:.0f} bets, "
            f"yield {yld:+.2%} ± {yse:.2%}, CLV {clv:+.2%} ± {cse:.2%}"
        )
    significant = p["improvement"] > 2 * p["improvement_se"]
    better_clv = p["anchored_mean_clv"] > p["market_mean_clv"]
    print(
        "\nVerdict: "
        + (
            "the adjustment adds information (significant log-loss gain and higher CLV)."
            if significant and better_clv
            else "no reliable information beyond the Pinnacle price; keep pricing bets against the market."
        )
    )
    return 0


def run_watch(args: argparse.Namespace) -> int:
    import time

    from parlay.betting.watch import WatchConfig, Watcher, format_slot_message
    from parlay.data.odds_sync import credit_cost
    from parlay.notify import notifier_from_env

    config = WatchConfig(
        competitions=tuple(args.competition),
        lead_minutes=args.lead,
        min_lead_minutes=args.min_lead,
        regions=args.regions,
        markets=args.api_markets,
        credit_reserve=args.credit_reserve,
        bankroll=args.bankroll,
        paper=not args.no_paper,
        notify_empty=args.notify_empty,
        value=_value_config(args),
    )
    api_key = os.environ.get("THE_ODDS_API_KEY", "")
    if not api_key and not args.next:
        raise SystemExit("Set THE_ODDS_API_KEY: the watcher needs Pinnacle prices from The Odds API")
    database = _open_database(args.database)
    notifier = notifier_from_env()
    watcher = Watcher(database, api_key, config, notifier=notifier)
    now = datetime.now(timezone.utc)

    if args.next:
        slots = [s for s in watcher.slots(now) if s.kickoff <= now + timedelta(days=7)]
        rows = [
            [
                f"{s.kickoff:%a %d %b %H:%M}",
                f"{s.trigger(config.lead_minutes):%a %H:%M}",
                ", ".join(s.competitions),
                str(len(s.matches)),
                str(credit_cost(config.markets, config.regions, len(s.competitions))),
            ]
            for s in slots
        ]
        print(table(["kickoff (UTC)", "sync at", "competitions", "matches", "credits"], rows, "lllrr"))
        total = sum(int(r[-1]) for r in rows)
        remaining = watcher.last_known_credits()
        print(
            f"\n{len(rows)} slots in the next 7 days ≈ {total} credits "
            f"(last known remaining: {remaining if remaining is not None else 'unknown'})."
        )
        database.close()
        return 0

    print(
        f"Watching {', '.join(config.competitions)}: sync {config.lead_minutes} min before each kickoff; "
        f"notifications: {'Telegram' if notifier else 'terminal only'}."
    )
    if not watcher.slots(now):
        print(
            "No upcoming kickoffs stored; run `parlay odds sync --source the-odds-api` "
            "(or `parlay daily`) first."
        )
    try:
        while True:
            now = datetime.now(timezone.utc)
            for report in watcher.run_due(now):
                print(f"\n[{now:%Y-%m-%d %H:%M} UTC] {report.status}")
                print(format_slot_message(report, config.bankroll))
            if args.once or (args.until and now >= args.until):
                break
            upcoming = watcher.next_trigger(now)
            wait = 900.0 if upcoming is None else (upcoming - now).total_seconds()
            if args.until:
                wait = min(wait, max(0.0, (args.until - now).total_seconds()))
            time.sleep(max(30.0, min(wait, 900.0)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        database.close()
    return 0


SELECTION_ALIASES = {"1": "home", "x": "draw", "2": "away", "o": "over", "u": "under"}


def _find_match(text: str, matches: list):
    from parlay.data.teams import canonical_team_name

    for match in matches:
        if match.match_id == text:
            return match
    if " vs " not in text.lower():
        raise SystemExit(f"cannot read match {text!r}; write it as 'Home vs Away'")
    index = text.lower().index(" vs ")
    home = canonical_team_name(text[:index]).casefold()
    away = canonical_team_name(text[index + 4 :]).casefold()
    found = [m for m in matches if m.home_team.casefold() == home and m.away_team.casefold() == away]
    if not found:
        raise SystemExit(f"no upcoming match {text!r} with stored prices; run `parlay odds sync` first")
    return min(found, key=lambda m: m.date)


def _ticket_rows(ticket) -> list[list[str]]:
    rows = []
    for number, leg in enumerate(ticket.legs, start=1):
        fair = f"{1 / leg.fair_probability:.2f}" if leg.fair_probability else "-"
        ev = f"{leg.expected_value:+.1%}" if leg.expected_value is not None else "-"
        kickoff = f"{leg.kickoff_at:%a %H:%M}" if leg.kickoff_at else leg.match_date.isoformat()
        rows.append([str(number), kickoff, leg.description, f"{leg.odds:.2f}", fair, ev])
    return rows


def _print_ticket(ticket, *, bankroll: float, kelly_fraction: float, max_stake: float) -> float:
    print(table(["#", "kickoff", "leg", "odds", "fair", "EV"], _ticket_rows(ticket), "lllrrr"))
    p = ticket.win_probability
    stake = ticket.stake_fraction(kelly_fraction=kelly_fraction, max_stake=max_stake)
    boost = f" (incl. {ticket.boost:+.0%} boost)" if ticket.boost else ""
    print(f"\nTicket at {ticket.bookmaker}: {len(ticket.legs)} legs · odds {ticket.combined_odds:.2f}{boost}")
    if p is None:
        print("Fair probability unknown (a leg has no reference price): EV cannot be computed.")
    else:
        print(
            f"Fair odds {1 / p:.2f} · win probability {p:.1%} · EV {ticket.expected_value:+.1%} · "
            f"Kelly stake {stake:.2%} of bankroll ≈ {stake * bankroll:.2f}"
        )
    for warning in ticket.warnings:
        print(f"  ! {warning}")
    if p is not None and (ticket.expected_value or 0) <= 0:
        print("Verdict: negative expected value — the bookmaker margin compounds with every leg. Do not bet.")
    elif p is not None:
        print("Verdict: positive expected value, but the ticket loses most of the time; keep stakes small.")
    return stake


def run_ticket(args: argparse.Namespace) -> int:
    from parlay.betting.tickets import build_ticket, favourite_leg, price_leg

    database = _open_database(args.database)
    ledger = Ledger(database)
    now = args.as_of or datetime.now(timezone.utc)
    upcoming = [
        m
        for m in database.load_matches()
        if m.home_goals is None and (m.kickoff_at is None or m.kickoff_at > now)
    ]
    legs = []
    try:
        for key in args.keys:
            data = ledger.candidate(key)
            match = next((m for m in upcoming if m.match_id == data["match_id"]), None)
            if match is None:
                raise SystemExit(f"{key}: the match has started or is unknown")
            legs.append(
                price_leg(
                    match,
                    data["market"],
                    data["selection"],
                    database.load_odds(match_id=match.match_id),
                    bookmaker=args.book,
                    as_of=now,
                    reference=args.reference,
                    leg_odds=float(data["odds"]) if data["bookmaker"] == args.book else None,
                )
            )
        for spec in args.leg:
            body, _, price = spec.partition("@")
            parts = body.rsplit(":", 2)
            if len(parts) != 3:
                raise SystemExit(f"cannot read leg {spec!r}; expected 'Home vs Away:MARKET:SELECTION'")
            match = _find_match(parts[0].strip(), upcoming)
            selection = SELECTION_ALIASES.get(parts[2].strip().lower(), parts[2].strip().lower())
            legs.append(
                price_leg(
                    match,
                    parts[1].strip().lower(),
                    selection,
                    database.load_odds(match_id=match.match_id),
                    bookmaker=args.book,
                    as_of=now,
                    reference=args.reference,
                    leg_odds=float(price) if price else None,
                )
            )
        if args.favourites:
            horizon = now + timedelta(days=args.days)
            taken = {leg.match_id for leg in legs}
            for match in sorted(upcoming, key=lambda m: (m.kickoff_at or now, m.match_id)):
                if len(legs) >= args.max_legs:
                    break
                if match.match_id in taken or (match.kickoff_at and match.kickoff_at > horizon):
                    continue
                try:
                    legs.append(
                        favourite_leg(
                            match,
                            database.load_odds(match_id=match.match_id),
                            bookmaker=args.book,
                            as_of=now,
                            reference=args.reference,
                        )
                    )
                except ValueError:
                    continue
        ticket = build_ticket(legs, args.book, boost=args.boost, ticket_odds=args.ticket_odds)
    except (ValueError, KeyError) as exc:
        database.close()
        raise SystemExit(str(exc)) from exc
    stake = _print_ticket(
        ticket, bankroll=args.bankroll, kelly_fraction=args.kelly_fraction, max_stake=args.max_stake
    )
    if args.record:
        amount = args.stake or max(stake * args.bankroll, 0.01)
        ticket_id = ledger.record_ticket(ticket, kind=args.record, stake=amount)
        print(
            f"\nRecorded {args.record} parlay #{ticket_id}."
            if ticket_id
            else "\nThis ticket is already open."
        )
    database.close()
    return 0


def _print_value_tickets(
    ledger: Ledger, bets: list[ValueBet], args: argparse.Namespace, config: ValueConfig
) -> None:
    from parlay.betting.tickets import value_tickets

    tickets = value_tickets(bets, max_legs=args.max_legs)
    print("\n=== Mix parlays from value bets (one leg per match, one bookmaker per ticket) ===")
    if not tickets:
        print("Fewer than two value bets on different matches at one bookmaker: no ticket.")
        return
    for ticket in tickets:
        print()
        stake = _print_ticket(
            ticket, bankroll=args.bankroll, kelly_fraction=config.kelly_fraction, max_stake=config.max_stake
        )
        print(f"Ticket key {ticket.key}")
        if args.paper:
            recorded = ledger.record_ticket(ticket, kind="paper", stake=max(stake * args.bankroll, 0.01))
            if recorded:
                print(f"Recorded paper parlay #{recorded}.")


def _parlay_backtest_report(result, args: argparse.Namespace, config: ValueConfig) -> dict[str, float]:
    from parlay.betting.backtest import run_parlay_backtest

    parlays = run_parlay_backtest(
        result, max_legs=args.max_legs, kelly_fraction=config.kelly_fraction, max_stake=config.max_stake
    )
    pm = parlays.metrics
    if not pm["tickets"]:
        print("No day had two value bets on different matches at one bookmaker: no parlay tickets.\n")
        return pm
    print(
        f"Mix parlays (up to {args.max_legs} legs, one ticket per day and bookmaker): "
        f"{pm['tickets']:.0f} tickets, {pm['mean_legs']:.1f} legs, odds {pm['mean_odds']:.1f}, "
        f"win probability {pm['mean_win_probability']:.1%}"
    )
    print(
        f"  expected EV {pm['mean_expected_value']:+.1%} · realized {pm['yield']:+.1%} "
        f"± {pm['yield_se']:.1%} · hit rate {pm['hit_rate']:.0%} · "
        f"longest losing streak {pm['longest_losing_streak']:.0f}"
    )
    print(
        f"  CLV {pm['mean_clv']:+.1%} ± {pm['clv_se']:.1%} · Kelly bankroll multiple "
        f"{pm['kelly_bankroll_multiple']:.3f} · same legs as singles {pm['singles_yield']:+.1%} "
        f"± {pm['singles_yield_se']:.1%}\n"
    )
    write_rows(parlays.tickets, args.output_dir / "parlays.csv")
    return pm
