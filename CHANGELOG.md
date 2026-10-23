# Changelog

## Unreleased

### Added
- `parlay.betting`: fair prices from Pinnacle / Betfair Exchange / consensus,
  value finder over the bookmakers you use (exchange commission, Kelly with
  uncertainty shrinkage), bet ledger with closing-line value and settlement
  of 1X2, totals and Asian handicaps (including quarter lines), and a
  historical value backtest.
- Free odds feeds: football-data `fixtures.csv` and The Odds API (Pinnacle in
  region `eu`). CLI: `odds sync`, `value`, `bets add|update|report|list`,
  `backtest-value`.
- `parlay telegram`: button-driven bot for beginners in Indonesian, restricted
  to your chat id: value-bet cards (pick in plain words, odds, win
  probability, EV in money, stake in money), "placed"/simulate/add-to-parlay
  buttons, parlay slip with ticket EV, bankroll setting, plain-language
  report, schedule in local time, daily maintenance (results, odds,
  settlement, outcome message) and watcher alerts as cards. `--setup`
  detects and saves the chat id.
- Secrets are read from `~/.config/parlay/env` (or `$PARLAY_ENV_FILE`)
  without overriding variables already set; tests never read it.
- Mix parlays: `parlay ticket` (accumulator EV against Pinnacle, boosts,
  per-leg warnings, shrunk Kelly), `value --parlay` (tickets from value bets
  only), parlay tickets in the ledger (leg-by-leg settlement incl. half
  win/loss, ticket CLV) and `backtest-value --parlay`.
- `parlay watch`: lineup-window re-pricing of each kickoff slot (The Odds API
  synced only for the competitions in the slot, once per slot, with a credit
  reserve), paper logging and optional Telegram notifications.
- `parlay daily` routine and `parlay anchor-test` (market-anchored adjustment,
  walk-forward by season). Result on 2022-26: the goal model and simple
  features add no information beyond Pinnacle (model weight negative every
  season), so bets stay priced against the plain Pinnacle price.
- Loader stores every football-data bookmaker, best/average prices, Betfair
  Exchange, Asian handicap lines and closing prices.

### Fixed
- football-data kickoff times are UK local time; they are now converted to UTC.
- Syncing a scheduled match no longer erases a stored result.
- Model backtests fall back to Betfair Exchange closing prices when Pinnacle
  is missing (football-data dropped Pinnacle in 2026-27).

## 0.2.0 — 2026-10-23

### Fixed
- Dixon-Coles MLE never left its starting point (numerical gradient returned
  NaN), so all team strengths were zero while the fit reported convergence.
  Walk-forward 1X2 log loss with identical settings: 1.0764 → 1.0298.
- `predict` shrank every prediction 75% toward league average
  (`--early-season-shrinkage 0.75` applied to all matches).
- `--sot-weight` was accepted but never used.
- Corner model used different log-rate bounds in fitting and prediction and a
  fixed dispersion; it was worse than a constant base rate.
- The CSV loader ignored team-name normalization and aborted a whole season on
  one malformed price.
- Backtests on a multi-league database scored every league regardless of
  `--competition`; closing prices could be selected as pre-match prices.
- `calibrate` failed on real data when many matches shared a kickoff time.

### Added
- Vectorized MLE engine with analytic gradients (one engine for goals and corners).
- Shots-on-target strength blend (`--sot-blend`).
- `parlay fetch` to download football-data seasons; Asian handicap, double
  chance, any goal/corner line and non-decimal odds formats in `predict`.
- RPS, O/U 2.5 and BTTS scoring, market and closing-line benchmarks on
  identical samples, closing-line value, joint half-life × L2 nested tuning.
- CI, ruff, mypy, pre-commit, statistical and leakage tests.

### Changed (breaking)
- Match IDs no longer include the season label — rebuild the database.
- `BacktestConfig` replaces the long keyword lists; `cli.py` became a package.
- Removed `--sot-weight`, `--early-season-shrinkage`, the heuristic estimator
  and `models.dynamic`.
