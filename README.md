# Parlay Research

Modular research framework for football score probabilities. The first slice
contains the data contract, validation, independent-Poisson score matrix, and
derived market probabilities. Model fitting and temporal backtesting will use
the same interfaces in subsequent milestones.

## Development

```bash
python -m pytest
```

The project is research-only. It does not place bets or provide financial
advice.

## Operational Workflow

Ingest raw CSV files once, then run backtests from the resulting database:

```bash
PYTHONPATH=src python3 -m parlay.cli ingest data/raw/E0.csv \
  --database data/parlay.sqlite \
  --manifest data/snapshots/manifest.json \
  --competition EPL --season 2023-24

PYTHONPATH=src python3 -m parlay.cli backtest \
  --database data/parlay.sqlite \
  --model dixon_coles \
  --initial-train-days 730 --test-days 30 \
  --output-dir outputs/backtest/epl-dixon-coles
```

When `--database` exists, the backtest command reads matches and odds from
SQLite. CSV arguments are only a fallback for an unpopulated database.

## Data Layout

The database layer uses SQLite by default and keeps separate tables for:

- `teams`: canonical team identity and aliases are resolved before ingestion.
- `matches`: one row per match, including the final result when known.
- `odds_snapshots`: append-only odds observations with `captured_at`.
- `feature_snapshots`: serialized feature vectors with their own `as_of` time.

Backtests use expanding time windows. A prediction for a match may only use
matches, odds, and feature snapshots available before that prediction time.

## Contributors

* **bimataylorjoy** — [@bimataylorjoy](https://github.com/bimataylorjoy) (maintainer, multi-market engine, Championship ingestion, CLI dashboard)
