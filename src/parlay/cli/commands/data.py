"""``parlay ingest`` and ``parlay fetch``: build the research database."""

from __future__ import annotations

import argparse
from pathlib import Path

from parlay.cli._common import DEFAULT_SNAPSHOT_DIR, add_database
from parlay.data.database import ResearchDatabase
from parlay.data.ingestion import ingest_csv_files
from parlay.data.sources import FOOTBALL_DATA_LEAGUES, acquire_csv, football_data_url


def register(commands: argparse._SubParsersAction) -> None:
    ingest = commands.add_parser("ingest", help="ingest football-data CSV files (paths or URLs)")
    ingest.add_argument("csv", nargs="+", help="CSV paths or http(s) URLs")
    add_database(ingest)
    ingest.add_argument("--competition", default="EPL", help="competition label (default: EPL)")
    ingest.add_argument("--season", default=None, help="season label (default: derived from match dates)")
    ingest.add_argument(
        "--manifest", default="data/snapshots/manifest.json", help="where to write the manifest"
    )
    ingest.add_argument("--feature-window", type=int, default=5)
    ingest.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    ingest.set_defaults(handler=run_ingest)

    fetch = commands.add_parser("fetch", help="download and ingest football-data.co.uk seasons")
    fetch.add_argument(
        "--league",
        nargs="+",
        default=["E0", "E1"],
        help=f"football-data division codes (default: E0 E1). Known: {', '.join(FOOTBALL_DATA_LEAGUES)}",
    )
    fetch.add_argument(
        "--from-season", type=int, default=2019, help="first season start year (default: 2019)"
    )
    fetch.add_argument(
        "--to-season", type=int, required=True, help="last season start year, e.g. 2026 for 2026-27"
    )
    add_database(fetch)
    fetch.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    fetch.add_argument("--manifest-dir", default="data/snapshots")
    fetch.set_defaults(handler=run_fetch)


def run_ingest(args: argparse.Namespace) -> int:
    snapshots = [acquire_csv(source, args.snapshot_dir)[0] for source in args.csv]
    database = ResearchDatabase(args.database)
    manifest = ingest_csv_files(
        snapshots,
        database,
        competition=args.competition,
        season=args.season,
        feature_window=args.feature_window,
        manifest_path=args.manifest,
    )
    database.close()
    print(
        f"Ingested {manifest['match_count']} matches and {manifest['odds_count']} odds snapshots "
        f"into {args.database}"
    )
    return 0


def run_fetch(args: argparse.Namespace) -> int:
    if args.to_season < args.from_season:
        raise SystemExit("--to-season must not be before --from-season")
    database = ResearchDatabase(args.database)
    total = 0
    for league in args.league:
        competition = FOOTBALL_DATA_LEAGUES.get(league, league)
        for start in range(args.from_season, args.to_season + 1):
            url = football_data_url(league, start)
            try:
                snapshot, _ = acquire_csv(url, args.snapshot_dir)
            except OSError as exc:
                print(f"  {league} {start}-{(start + 1) % 100:02d}: download failed ({exc})")
                continue
            manifest = ingest_csv_files(
                [snapshot],
                database,
                competition=competition,
                manifest_path=Path(args.manifest_dir) / f"{league}_{start}.json",
            )
            total += int(manifest["match_count"])  # type: ignore[call-overload]
            season = f"{start}-{(start + 1) % 100:02d}"
            print(f"  {league} {season}: {manifest['match_count']} matches ({competition})")
    database.close()
    print(f"Ingested {total} matches into {args.database}")
    return 0
