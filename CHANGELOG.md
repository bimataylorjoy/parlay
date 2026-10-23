# Changelog

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
