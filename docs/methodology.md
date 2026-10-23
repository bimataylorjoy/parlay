# Methodology

This document describes what the models assume, how they are estimated, how
markets are priced from them, and how they are evaluated without leaking
future information.

## 1. Goal model

For a match between home team *h* and away team *a*:

```
log λ_home = μ + γ·venue + attack[h] − defense[a]
log λ_away = μ           + attack[a] − defense[h]
```

* `μ` — league intercept, `γ` — home advantage (`venue = 0` on neutral grounds).
* `attack` / `defense` — team parameters with sum-to-zero constraints
  (the last team is minus the sum of the others).
* Log-rates are bounded to `[−3, 3]` in fitting **and** prediction
  (`models/numerics.py`).

Three count distributions share this parameterization:

| model | score distribution |
|---|---|
| `poisson` | independent Poisson |
| `dixon_coles` | Poisson × τ correction on 0-0, 0-1, 1-0, 1-1 (Dixon & Coles, 1997) |
| `negative_binomial` | independent NB2, `Var = μ + μ²/φ`, φ estimated jointly |

Dixon-Coles τ:

```
τ(0,0) = 1 − λμρ    τ(0,1) = 1 + λρ    τ(1,0) = 1 + μρ    τ(1,1) = 1 − ρ
```

with ρ restricted to `[−0.3, 0.3]`.

## 2. Estimation

`models/mle.py` maximizes the weighted, penalized log-likelihood

```
Σ_i w_i · [log p(y_home,i | λ_home,i) + log p(y_away,i | λ_away,i) + log τ_i]
  − ½ · l2 · (Σ attack² + Σ defense²)
```

* **Time decay** `w_i = 0.5^(age_i / half_life)` — recent matches count more.
* **L2 prior** — equivalent to a `N(0, 1/l2)` prior on every team parameter
  (MAP estimation). Teams with little weighted data are shrunk toward the
  league average automatically, which matters most for promoted teams.
* The gradient is analytic and vectorized; L-BFGS-B needs one objective
  evaluation per iteration. A finite-difference gradient check is part of the
  test suite (`tests/test_mle.py`).
* EPL and Championship are fitted **jointly**: promoted and relegated teams
  connect the two leagues, so a promoted team enters the Premier League with a
  strength estimated on the same scale.

`estimator="bayesian_hmc"` (optional `bayes` extra) samples the Poisson model
with PyMC NUTS using exchangeable priors; predictions then average score
matrices over posterior draws (posterior predictive), not plug-in means.

## 3. Corner model

`models/corners.py` uses the same engine on corner counts with an NB2
likelihood whose dispersion is estimated (corners are over-dispersed). It is
fitted and evaluated independently of the goal model.

## 4. From score matrix to markets

The score matrix `M[i, j] = P(home = i, away = j)` is the single source of
truth (`prediction/markets.py`):

* 1X2: lower triangle / diagonal / upper triangle.
* Totals and Asian handicaps on integer, half and **quarter** lines. A quarter
  line is a stake split across the two adjacent lines, so its win/push/loss
  probabilities are the average of those two lines and expected value is exact.
* BTTS, win-to-nil, double chance, correct score, team totals.
* Parlays: joint probability of all legs from the same matrix
  (`prediction/parlay.py`); legs on one match are dependent.

The grid is sized adaptively so the truncated tail mass is below `1e-6`
(never smaller than 11×11) and renormalized.

## 5. Prices and value

* **De-vig**: Shin (default), power, or multiplicative.
* **Edge** = model probability − de-vigged market probability (both
  conditional on the bet not being void).
* **EV** = `p_win·(odds − 1) − p_loss`; pushes return the stake.
* **Kelly** with push: `f* = (b·p_win − p_loss) / (b·(p_win + p_loss))`,
  multiplied by the Kelly fraction and capped by `--max-stake`.
* **Fair odds** = break-even price `1 + p_loss / p_win`.

`evaluation/anomaly.py` treats large model-market disagreement as something to
investigate (missing team news, promoted teams), not as an automatic edge.

## 6. Evaluation without leakage

* **Walk-forward folds** (`evaluation/temporal.py`): the model is refitted
  every `--test-days` on matches whose results were *knowable* before the
  first forecast of the fold (`data/information.py`: kickoff + competition
  result lag; end of day when kickoff is unknown).
* **Forecast time** = kickoff − 60 minutes. Pre-match prices must be captured
  at or before it; closing prices are never used as inputs, only as a
  benchmark and for closing-line value (CLV).
* **Scores**: log loss, multiclass Brier, ranked probability score (RPS) for
  1X2; binary log loss for Over 2.5 and BTTS. Market benchmarks are always
  computed on the identical subset of matches.
* **Tuning** (`evaluation/tuning.py`): hyperparameters are selected on a
  development period and scored once on an untouched holdout. Grid scores
  on the development period are optimistic by construction.
* **Calibration**: temperature scaling fitted only on earlier forecasts.

## 7. Known limitations

* The models only see results (and corners). They know nothing about injuries,
  suspensions, rotation, managers or transfers, which the market prices in.
* Closing markets remain sharper than the model (see the README benchmark).
  Positive backtest yield without positive CLV should be treated as noise.
* Team news and lineups from Sportmonks are stored with availability
  timestamps but are not yet used to adjust rates; player-impact adjustments
  require validated player ratings and an out-of-sample ablation.
