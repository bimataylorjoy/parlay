# Parlay — time-safe football probability research

<p align="center">
  <a href="https://github.com/bimataylorjoy/parlay/actions/workflows/ci.yml"><img src="https://github.com/bimataylorjoy/parlay/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="python">
  <img src="https://img.shields.io/badge/version-0.2.0-orange" alt="version">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license">
</p>

Parlay fits attack/defence models to historical football results, turns them
into a full score distribution for any fixture, prices every common market
(1X2, totals and Asian handicaps including quarter lines, BTTS, correct score,
corners) and evaluates everything with **walk-forward backtests that cannot see
the future**.

It also includes a **price-based value-betting workflow**: compare bookmaker
prices with the de-vigged Pinnacle price, log bets in a ledger, and measure
closing-line value. That — not the goal model — is where the backtests show
an edge ([docs/value-betting.md](docs/value-betting.md)).

> Research software. It does not place bets. The goal model is well calibrated
> but carries no information beyond the market; treat its prices as a second
> opinion. Never stake money you cannot afford to lose.

---

## Contents

- [Highlights](#highlights)
- [Benchmarks](#benchmarks)
- [Value betting](#value-betting)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Fixture CSV format](#fixture-csv-format)
- [CLI reference](#cli-reference)
- [Python API](#python-api)
- [Project layout](#project-layout)
- [Development](#development)
- [Upgrading from 0.1](#upgrading-from-01)
- [Limitations](#limitations)

## Highlights

- **One score matrix, every market.** Poisson, Dixon-Coles and Negative
  Binomial goal models share one parameterization; markets are sums over
  `P(home = i, away = j)`, so prices are mutually consistent.
- **Fast, verified estimation.** Penalized MLE with analytic, vectorized
  gradients (checked against finite differences in the test suite), time
  decay, an L2 prior that shrinks thinly-observed teams, and a
  shots-on-target blend that reduces noise in team ratings.
- **Leakage-safe evaluation.** Results are only used once they were knowable;
  prices must be captured before kickoff − 60 min; closing prices are used
  only as a benchmark and for closing-line value (CLV).
- **Honest scoring.** Log loss, Brier, RPS, Over 2.5 and BTTS log loss, all
  compared with the de-vigged bookmaker and the Pinnacle closing line on the
  same matches. Hyperparameters are tuned with an untouched holdout.
- **Practical pricing.** Shin/power/multiplicative de-vig, push-aware EV and
  Kelly, fair odds, quarter lines, Indonesian/Malay/HK/American odds input.

## Benchmarks

EPL + Championship, football-data.co.uk, walk-forward with weekly refits.
Hyperparameters were chosen on matches up to 30 June 2024; the numbers below
are from the **untouched holdout, 1 July 2024 – October 2026** (2,062 matches).
Lower is better.

| 1X2 | log loss | RPS |
|---|---|---|
| Dixon-Coles + SOT blend (default) | **1.0380** | 0.2139 |
| Poisson + SOT blend | 1.0383 | 0.2139 |
| Poisson, goals only | 1.0407 | 0.2148 |
| Bet365 pre-match, Shin de-vig (same matches) | 1.0211 | — |
| Pinnacle closing, de-vig (1,410 matches; model on the same set: 1.0290) | 1.0089 | — |

| Over/Under 2.5 | log loss |
|---|---|
| Model | 0.6887 |
| Bet365 pre-match, de-vig | 0.6845 |

| Corners O/U 10.5 | log loss |
|---|---|
| Corner model | 0.6852 |
| Constant base rate (fitted on the holdout itself) | 0.6875 |

What this means:

- The model is well calibrated (temperature scaling fitted on earlier
  forecasts leaves the score unchanged, T ≈ 0.98), but it knows only results,
  shots and corners. Bookmakers also price injuries, lineups and news, and the
  closing line is sharper still.
- A naive "bet when the model disagrees with the market" strategy (edge ≥ 3%,
  EV ≥ 2%) loses money in the holdout (yield −13% ± 4%) with **negative CLV
  (−8%)**: large disagreements are mostly missing information, not edge.
- For reference, version 0.1 scored **1.0764** on the 2020–2026 walk-forward
  because its Dixon-Coles optimizer never moved from its starting point (all
  team strengths were zero). With identical settings the fixed engine scores
  1.0298.

Reproduce with `parlay fetch --from-season 2018 --to-season 2026` followed by
`parlay backtest --from 2024-07-01` (and `--model poisson`, `--sot-blend 0`, ...).

## Value betting

```bash
export THE_ODDS_API_KEY=...                       # free key: 500 credits/month, region eu includes Pinnacle
export PARLAY_BOOKS="Bet365,William Hill"         # bookmakers you can use
parlay odds sync                                   # football-data fixtures.csv (free, no key)
parlay odds sync --source the-odds-api             # Pinnacle reference
parlay daily                                       # all of the below in one go
parlay watch                                       # re-price each kickoff slot after lineups are out
parlay value --paper                               # value bets vs the fair Pinnacle price
parlay bets add vb:1a2b3c4d --stake 10 --odds 2.12
parlay bets update && parlay bets report           # closing-line value, yield, verdict
parlay backtest-value --from 2019-07-01            # same engine on historical prices
```

Historical test (July 2019 – June 2026, regular bookmakers vs the Pinnacle
reference, EV ≥ 2%): 331 bets, **closing-line value +3.3% ± 0.6%**, positive
in every season; flat yield +3.3% ± 11.4% (not yet significant). Against the
Betfair Exchange snapshot or the bookmaker consensus the CLV is ≤ 0, so the
default reference is Pinnacle. Details, data sources and the daily workflow:
[docs/value-betting.md](docs/value-betting.md).

## Installation

```bash
git clone https://github.com/bimataylorjoy/parlay.git
cd parlay
python -m venv .venv && source .venv/bin/activate
pip install -e .            # numpy + scipy only
pip install -e ".[dev]"     # + pytest, ruff, mypy, pre-commit
pip install -e ".[bayes]"   # + PyMC / ArviZ for --estimator bayesian_hmc
```

Requires Python ≥ 3.10.

## Quick start

```bash
# 1. Download and ingest EPL (E0) and Championship (E1), seasons 2019-20 .. 2026-27
parlay fetch --league E0 E1 --from-season 2019 --to-season 2026

# 2. Walk-forward backtest; artifacts go to outputs/backtest/
parlay backtest --from 2024-07-01
parlay backtest --competition EPL --model poisson --sot-blend 0

# 3. Compare models on identical folds, or tune with an untouched holdout
parlay compare --from 2024-07-01
parlay tune --model dixon_coles --holdout-days 365

# 4. Price upcoming fixtures (see examples/)
parlay predict examples/fixtures_template.csv --output outputs/predictions.csv
```

`parlay predict` prints a summary table and, per fixture, expected goals,
1X2 with fair odds, BTTS, totals (with push for integer/quarter lines), Asian
handicaps, the most likely scores, corners, and every priced selection with
fair odds, EV and edge over the de-vigged market:

```
2026-09-05  Manchester City vs Coventry City
  xG 2.60 - 0.79   1X2  75.8%  15.5%   8.6%   fair 1.32 / 6.44 / 11.59
  BTTS yes  50.7%   win-to-nil H  41.8% A   3.7%   double chance 1X  91.4% X2  24.2%
  Goals  1.5: O  85.5%  2.25: O  65.7% (push 10%)  2.5: O  65.7%  2.75: O  54.7% (push 11%)  3.5: O  43.8%
  Correct score  2-0 11.5%  3-0 9.9%  2-1 9.0%  1-0 8.4%  3-1 7.8%
  Corners xC 6.3 - 4.0  8.5: O  65.0%  9.5: O  54.4%  10.5: O  44.0%  11.5: O  34.5%
  Priced selections:
    btts:no                      odds   2.30  fair   2.03  EV +13.3% edge +8.8%  <== value
    totals:over 2.75             odds   1.70  fair   1.63  EV  +4.0% edge +5.3%  <== value
    asian_handicap:away -1.75    odds   1.98  fair   1.97  EV  +0.3% edge +1.5%
    totals:over 2.5              odds   1.50  fair   1.52  EV  -1.5% edge +2.3%
    ...
```

(The prices in `examples/fixtures_template.csv` are illustrative, so the
"value" flags above are not real recommendations.)

Value bets require EV ≥ `--min-ev` and edge ≥ `--min-edge`; stakes are
fractional Kelly capped by `--max-stake`. Fixtures flagged `REJECT` by the
anomaly layer (extreme disagreement plus thin history) get no recommendation.

## Fixture CSV format

Required: `Date` (`DD/MM/YYYY`, `DD/MM/YY` or `YYYY-MM-DD`), `HomeTeam`,
`AwayTeam`. Full names are mapped to football-data names
(`Manchester City` → `Man City`, see `src/parlay/data/teams.py`). Lines
starting with `#` are ignored. Optional price columns (prefix `B365` by
default, `--odds-prefix` for another bookmaker):

| market | columns |
|---|---|
| 1X2 | `B365H`, `B365D`, `B365A` |
| goal totals, any line | `B365>2.5`, `B365<2.5`, `B365>2.75`, ... |
| BTTS | `B365BTTS_Y`, `B365BTTS_N` |
| double chance | `B3651X`, `B365X2`, `B36512` |
| Asian handicap | `AHh` (home line, e.g. `-0.75`), `B365AHH`, `B365AHA` |
| corner totals | `B365CRN>9.5`, `B365CRN<9.5` (legacy `B365C>9.5` for lines ≥ 5) |

Prices in other formats: `--odds-format indo|malay|hk|american|fractional`
(e.g. Indonesian `-1.25` → decimal `1.80`).

## CLI reference

Run `parlay <command> --help` for every option.

| command | purpose |
|---|---|
| `fetch` | download football-data seasons (`--league E0 E1 SP1 ...`) and ingest them |
| `ingest` | ingest CSV files or URLs, with SHA-256 snapshots and a manifest |
| `backtest` | walk-forward backtest of one model; writes `predictions.csv`, `metrics.json`, `calibration.json` |
| `compare` | all goal models on identical folds |
| `tune` | half-life × L2 grid; selects on a development period, scores once on a holdout |
| `calibrate` | temperature scaling fitted on earlier forecasts of a backtest |
| `predict` | price fixtures from a CSV and list value bets |
| `odds sync` | store upcoming prices from football-data `fixtures.csv` or The Odds API |
| `value` | value bets against the de-vigged Pinnacle price; `--paper` logs them |
| `bets add / update / report / list` | bet ledger with closing-line value and settlement |
| `backtest-value` | replay the value engine on historical football-data prices |
| `anchor-test` | test whether the goal model or other features add information to the Pinnacle price |
| `daily` | results, odds syncs, CLV/settlement, value bets (paper) and report in one go |
| `watch` | re-price each kickoff slot ~50 min before kickoff (after lineups); optional Telegram alerts |
| `sync-sportmonks` / `predict-sportmonks` | fixtures, O/U 2.5 prices and team news from Sportmonks (`SPORTMONKS_API_TOKEN`) |

Shared model options: `--model {poisson,dixon_coles,negative_binomial}`,
`--estimator {mle,bayesian_hmc}`, `--half-life-days` (or `none`), `--l2-reg`,
`--sot-blend`. Walk-forward options: `--initial-train-days`, `--test-days`
(refit cadence, default 7), `--competition` (repeatable; scores only that
competition but trains on all), `--from` / `--to`, `--bookmaker`.

## Python API

```python
from datetime import date
from parlay.data.database import ResearchDatabase
from parlay.models import fit_team_strength, fit_corner_strength
from parlay.prediction import Fixture, PredictConfig, predict_fixture
from parlay.prediction.markets import totals_probabilities, asian_handicap

db = ResearchDatabase("data/parlay.sqlite")
history = [m for m in db.load_matches() if m.home_goals is not None]
odds = db.load_odds()
cutoff = date(2026, 10, 24)
train = [m for m in history if m.date < cutoff]

model = fit_team_strength(train, model="dixon_coles", as_of=cutoff)
matrix = model.score_matrix("Arsenal", "Chelsea")
print(totals_probabilities(matrix, 2.75))   # {'over': ..., 'push': ..., 'under': ...}
print(asian_handicap(matrix, -0.25))        # home win / push / loss

fixture = Fixture(cutoff, "Arsenal", "Chelsea", odds={("1x2", "home", None): 2.05,
                                                      ("1x2", "draw", None): 3.5,
                                                      ("1x2", "away", None): 3.6})
prediction = predict_fixture(fixture, model, fit_corner_strength(train, as_of=cutoff), PredictConfig())
print(prediction.value_bets, prediction.diagnostics)
```

Backtests are configured with a dataclass:

```python
from parlay.evaluation import BacktestConfig, run_backtest
result = run_backtest(history, odds, BacktestConfig(model="poisson", competitions=("EPL",)))
print(result.metrics["log_loss"], result.metrics["market_log_loss"])
```

The modelling details — likelihoods, priors, Dixon-Coles constraints, quarter
lines, Kelly with pushes, the information-set rules — are in
[docs/methodology.md](docs/methodology.md).

## Project layout

```
src/parlay/
  data/         schemas, CSV loader, team names, SQLite store, information sets, Sportmonks
  features/     rolling pre-match features, provenance, player ratings (research)
  models/       MLE engine, Poisson / Dixon-Coles / NB score models, team strength, corners, PyMC
  prediction/   market maths, de-vig, Kelly, odds formats, fixture pricing, parlays
  betting/      sharp reference prices, value finder, bet ledger with CLV, settlement, value backtest
  evaluation/   walk-forward backtest, scoring rules, calibration, tuning, regimes, anomalies
  cli/          `parlay` command (one module per command group) and text rendering
tests/          unit, statistical (parameter recovery, gradient checks), leakage and CLI tests
examples/       fixture CSV templates
docs/           methodology, value betting, and the original P0-P3 upgrade specification
```

## Development

```bash
make install     # pip install -e ".[dev]"
make check       # ruff + mypy + pytest
pre-commit install
```

CI runs lint, type checks and the test suite on Python 3.10–3.12.

## Upgrading from 0.1

- **Rebuild the database.** Match IDs no longer include the season label, so
  re-ingesting into an old database would duplicate matches; the loader now
  also stores every bookmaker, Asian handicap and closing price, and converts
  football-data kickoff times from UK time to UTC:
  `rm data/parlay.sqlite && parlay fetch --from-season 2018 --to-season 2026`.
- Removed options: `--sot-weight` (it never affected the model),
  `--early-season-shrinkage` (it shrank every prediction 75% toward average),
  `--estimator heuristic`, and the experimental `models.dynamic`.
  Shrinkage now comes from the L2 prior (`--l2-reg`); shots on target from
  `--sot-blend`.
- `run_backtest(matches, odds, BacktestConfig(...))` replaces the long keyword
  lists; `run_backtest_full` and `tune_half_life_nested` were folded into
  `run_backtest` and `nested_tuning`.
- `prediction.markets.totals_probabilities` replaces
  `totals_settlement_probabilities`, `corner_totals_probabilities` and
  `models.poisson.totals_probability`.
- `cli.py` became the `parlay.cli` package; `python -m parlay` still works.
- Corner odds columns in fixture CSVs: prefer `B365CRN>x` (football-data uses
  `B365C>2.5` for closing goal totals).

## Limitations

- Only results, shots on target and corners inform the ratings: no lineups,
  injuries, suspensions, managers or transfers.
- Teams with fewer than ~10 matches in the data are scored noticeably worse
  (holdout log loss 1.19 on 36 such matches vs 1.04) and are flagged in predictions.
- The corner model beats a constant base rate only slightly; corner prices
  are the least reliable output.
- Sportmonks team news and lineups are stored with availability timestamps
  but are not used to adjust rates yet.

## Contributors

- **bimataylorjoy** — [@bimataylorjoy](https://github.com/bimataylorjoy), maintainer.

Issues and pull requests are welcome.

## Disclaimer

This is a research framework, not a tipster service. Probabilities are model
estimates with real uncertainty, and the backtests above show no exploitable
edge against the market. Never stake money you cannot afford to lose.
