"""``parlay odds | value | bets | backtest-value``: price-based value betting with a CLV ledger."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from parlay.betting import Ledger, ValueConfig, find_value_bets, run_value_backtest
from parlay.betting.reference import REFERENCE_POLICIES
from parlay.betting.value import MARKET_KINDS, ValueBet
from parlay.cli._common import DEFAULT_SNAPSHOT_DIR, add_database, iso_date
from parlay.cli.render import table
from parlay.data.database import ResearchDatabase
from parlay.data.loaders import load_football_data_fixtures
from parlay.data.odds_columns import EXCHANGE
from parlay.data.sources import FOOTBALL_DATA_LEAGUES, acquire_csv
from parlay.evaluation.serialization import write_json, write_rows

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
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
    group.add_argument("--commission", type=float, default=0.05, help="exchange commission (default: 0.05)")
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
        "--commission", type=float, default=None, help="exchange commission (default: 0.05 on Betfair)"
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

    backtest = commands.add_parser("backtest-value", help="replay the value engine on historical prices")
    add_database(backtest)
    _add_value_options(backtest)
    backtest.add_argument("--from", dest="start", type=iso_date, help="first match date")
    backtest.add_argument("--to", dest="end", type=iso_date, help="last match date")
    backtest.add_argument("--output-dir", type=Path, default=Path("outputs/value_backtest"))
    backtest.set_defaults(handler=run_backtest_value)


def _open_database(path: str) -> ResearchDatabase:
    if not Path(path).exists():
        raise SystemExit(f"Database {path} not found; run `parlay fetch` first")
    return ResearchDatabase(path)


def run_odds_sync(args: argparse.Namespace) -> int:
    now = datetime.now(timezone.utc)
    database = ResearchDatabase(args.database)
    if args.source == "football-data":
        unknown = [code for code in args.league if code not in FOOTBALL_DATA_LEAGUES]
        if unknown:
            raise SystemExit(f"unknown division codes {unknown}; known: {', '.join(FOOTBALL_DATA_LEAGUES)}")
        snapshot, _ = acquire_csv(FIXTURES_URL, args.snapshot_dir)
        leagues = {code: FOOTBALL_DATA_LEAGUES[code] for code in args.league}
        matches, odds = load_football_data_fixtures(snapshot, captured_at=now, leagues=leagues)
        note = (
            "football-data fixtures.csv has no Pinnacle prices; "
            "also sync --source the-odds-api for a sharp reference."
        )
    else:
        from parlay.data.the_odds_api import SPORT_KEYS, fetch_odds, parse_events

        api_key = args.api_key or os.environ.get("THE_ODDS_API_KEY")
        if not api_key:
            raise SystemExit("Set THE_ODDS_API_KEY (free at https://the-odds-api.com) or pass --api-key")
        matches, odds = [], []
        quota = None
        for competition in args.competition:
            if competition not in SPORT_KEYS:
                raise SystemExit(f"unknown competition {competition!r}; known: {', '.join(SPORT_KEYS)}")
            events, quota = fetch_odds(
                api_key, SPORT_KEYS[competition], regions=args.regions, markets=args.api_markets
            )
            upcoming = [
                e for e in events if datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) > now
            ]
            found_matches, found_odds = parse_events(upcoming, competition, retrieved_at=now)
            matches += found_matches
            odds += found_odds
        note = f"The Odds API credits remaining: {quota.remaining}" if quota else ""
    database.insert_future_matches(matches)
    database.insert_odds(odds)
    database.close()
    books = sorted({o.bookmaker for o in odds})
    print(f"Stored {len(matches)} upcoming matches and {len(odds)} prices from {len(books)} bookmakers.")
    if books:
        print("Bookmakers: " + ", ".join(books))
    if note:
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
        recorded = sum(
            ledger.record_value_bet(
                bet,
                kind="paper",
                stake=max(bet.stake_fraction * args.bankroll, 0.01),
                commission=config.exchange_commission if bet.bookmaker == EXCHANGE else 0.0,
            )
            is not None
            for bet in bets
        )
        print(f"\nRecorded {recorded} new paper bets.")
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
        commission = (
            args.commission
            if args.commission is not None
            else (0.05 if candidate["bookmaker"] == EXCHANGE else 0.0)
        )
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
        commission = (
            args.commission if args.commission is not None else (0.05 if args.bookmaker == EXCHANGE else 0.0)
        )
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
    counts = Ledger(database).update()
    database.close()
    print(f"Closing prices attached to {counts['clv']} bets; {counts['settled']} bets settled.")
    print(
        "Results arrive via `parlay fetch --to-season <current>`; "
        "closing prices via the same files or an odds sync."
    )
    return 0


def run_bets_report(args: argparse.Namespace) -> int:
    database = _open_database(args.database)
    report = Ledger(database).report(kind=args.kind)
    database.close()
    if not report["bets"]:
        print("No bets recorded yet.")
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
    rows = Ledger(database).bets(kind=args.kind, open_only=args.open)
    database.close()
    if not rows:
        print("No bets.")
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
    write_json(
        {"metrics": m, "breakdown": result.breakdown, "config": asdict(config)},
        args.output_dir / "metrics.json",
    )
    print(f"Artifacts written to {args.output_dir}/")
    if config.reference != "pinnacle":
        print(LOW_CONFIDENCE.format(ref=config.reference))
    return 0
