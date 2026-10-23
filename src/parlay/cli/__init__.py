"""Command-line interface: ``parlay <command> --help``."""

from __future__ import annotations

import argparse
import logging

from parlay import __version__
from parlay.cli.commands import betting, data, evaluate, predict, sportmonks, telegram
from parlay.config import load_env_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parlay",
        description="Time-safe football probability research: fit, backtest and price matches.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show warnings from data loading")
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    data.register(commands)
    evaluate.register(commands)
    predict.register(commands)
    betting.register(commands)
    telegram.register(commands)
    sportmonks.register(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.verbose else logging.ERROR, format="%(levelname)s: %(message)s"
    )
    return int(args.handler(args))


__all__ = ["build_parser", "main"]
