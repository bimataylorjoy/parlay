# parlay

[![CI](https://github.com/bimataylorjoy/parlay/actions/workflows/ci.yml/badge.svg)](https://github.com/bimataylorjoy/parlay/actions/workflows/ci.yml)

**Score-distribution models and market-relative evaluation for association
football, with a closing-line-value framework for price-based betting.**

## Abstract

We fit penalized maximum-likelihood rate models (Poisson, Dixon–Coles,
negative binomial) with exponential time decay to English Premier League
and Championship results, and derive prices for 1X2, goal totals, Asian
handicaps (including quarter lines), both-teams-to-score, correct score and
corner totals from a single joint score distribution. All models are
evaluated walk-forward under an explicit information-set constraint, with
hyperparameters selected before a held-out period of 2,062 matches
(July 2024 – October 2026). On that period the best model attains a 1X2 log
loss of 1.0380, against 1.0211 for de-vigged Bet365 pre-match prices and
1.0089 for de-vigged Pinnacle closing prices. A market-anchored
specification fitted walk-forward by season assigns the model a negative
weight in every season; the model therefore carries no information beyond
the sharp price. Price discrepancies between bookmakers do carry
information: over July 2019 – June 2026, bets on regular bookmakers priced
at least 2% above the de-vigged Pinnacle price (331 bets) achieved a
closing-line value of +3.3% ± 0.6%, positive in each of seven seasons, while
the flat-stake yield (+3.3% ± 11.4%) is not yet distinguishable from zero.
The repository contains the estimation code, the evaluation harness, and
the tooling used to apply the price-based rule prospectively.

## 1. Data

| source | content | use |
|---|---|---|
| football-data.co.uk season files | results, shots on target, corners; pre-match and closing prices of 6–12 bookmakers, Betfair Exchange, market maximum and average | training, evaluation, historical value backtest |
| football-data.co.uk `fixtures.csv` | upcoming fixtures with the same bookmakers (no Pinnacle from 2026-27) | prospective prices |
| The Odds API v4, region `eu` | ~25 bookmakers including Pinnacle and Betfair Exchange; main lines only | prospective sharp reference |

Kickoff times in football-data files are UK local time and are converted to
UTC. Every ingested file is stored as a SHA-256-addressed snapshot with a
manifest. Team names are normalized to football-data conventions
(`src/parlay/data/teams.py`).

## 2. Models

Let $\lambda_{ij} = \exp(\mu + \eta + \alpha_i - \delta_j)$ and
$\mu_{ij} = \exp(\mu + \alpha_j - \delta_i)$ be the expected goals of home
team $i$ and away team $j$, with attack $\alpha$, defence $\delta$ and home
advantage $\eta$. Three likelihoods share this parameterization:

* **Poisson** (Maher, 1982): independent Poisson margins.
* **Dixon–Coles** (Dixon & Coles, 1997): Poisson margins with the
  low-score dependence correction $\tau_{\lambda,\mu,\rho}(x, y)$ for
  $x, y \le 1$.
* **Negative binomial (NB2)**: Poisson margins replaced by NB2 margins with a
  shared dispersion parameter, for overdispersed counts.

Parameters maximize the time-weighted log-likelihood
$\sum_m w_m \ell_m(\theta) - \tfrac{\kappa}{2}\lVert(\alpha, \delta)\rVert^2$
with $w_m = 2^{-\Delta t_m / h}$ (half-life $h$, default 365 days) and an L2
prior ($\kappa = 2.5$) that shrinks thinly observed teams toward the mean.
Optimization uses L-BFGS-B with analytic, vectorized gradients; the test
suite checks them against finite differences and verifies parameter
recovery on simulated leagues. Ratings fitted on goals are blended with
ratings fitted on shots on target (weight 0.5), which reduces sampling noise
in the strength estimates. Corners use the same machinery with separate
hyperparameters (half-life 270 days, $\kappa = 150$). An optional Bayesian
estimator (PyMC, HMC) is provided for comparison.

Every market is a functional of the score matrix
$P(\text{home} = x, \text{away} = y)$, so the derived prices are mutually
consistent. Integer and quarter lines are settled with push and half
win/half loss outcomes. Market prices are de-vigged with Shin's (1993)
method by default; power and multiplicative normalizations are available.
Full derivations are given in [docs/methodology.md](docs/methodology.md).

## 3. Evaluation protocol

* **Information sets.** A result is usable only after it was knowable. A
  price is usable only if it was captured at least 60 minutes before
  kickoff. Closing prices appear only as benchmarks and in the computation
  of closing-line value (CLV), never as inputs. Leakage tests enforce these
  rules.
* **Walk-forward.** Models are refitted weekly on all data available at the
  refit time and score only the following week.
* **Holdout.** Half-life and L2 strength were selected by grid search on
  matches up to 30 June 2024. The period 1 July 2024 – October 2026 was
  scored once with the selected values.
* **Scoring rules.** Log loss, Brier score and ranked probability score
  (Epstein, 1969; Constantinou & Fenton, 2012) for 1X2; log loss for
  Over/Under 2.5, BTTS and corner totals. Market benchmarks are computed on
  the same matches.
* **Calibration.** Temperature scaling (Guo et al., 2017) is fitted on
  earlier forecasts only and applied to later ones.

## 4. Results

### 4.1 Forecast accuracy (holdout, 2,062 matches)

| 1X2 | log loss | RPS |
|---|---|---|
| Dixon–Coles + shots-on-target blend (default) | **1.0380** | 0.2139 |
| Poisson + shots-on-target blend | 1.0383 | 0.2139 |
| Poisson, goals only | 1.0407 | 0.2148 |
| Bet365 pre-match, Shin de-vig | 1.0211 | — |
| Pinnacle closing, de-vig (1,410 matches; model on the same subset: 1.0290) | 1.0089 | — |

| market | model | benchmark |
|---|---|---|
| Over/Under 2.5, log loss | 0.6887 | 0.6845 (Bet365 pre-match, de-vig) |
| Corners O/U 10.5, log loss | 0.6852 | 0.6875 (constant base rate fitted on the holdout) |

The model is well calibrated: the fitted temperature is $T \approx 0.98$,
and applying it leaves the score unchanged. The gap to the market is
consistent with the model's information set, which contains only results,
shots and corners, whereas bookmakers also price lineups, injuries and news.
Teams with fewer than about ten matches in the data are forecast markedly
worse (log loss 1.19 on 36 such matches).

### 4.2 Information beyond the market

The market-anchored specification (Egidi, Pauli & Torelli, 2018; cf.
Hubáček, Šourek & Železný, 2019) corrects the de-vigged Pinnacle
probabilities rather than forecasting from scratch:

$$\text{score}_k = \log p^{\text{Pinnacle}}_k + \textstyle\sum_j w_j x_{jk}, \qquad p = \operatorname{softmax}(\text{score}).$$

The features $x_{jk}$ are: the model's log-ratio to Pinnacle, Bet365's
log-ratio to Pinnacle, rest-day difference, promoted-team indicator, and a
recalibration term. Weights are fitted walk-forward by season with an L2
penalty. Out of sample, 2022–2026 (3,260 matches):

| specification | log loss | bets | CLV |
|---|---|---|---|
| Pinnacle price | **1.0023** | 209 | **+2.1% ± 0.7%** |
| anchored, all features | 1.0029 | 458 | −1.1% ± 0.4% |
| anchored, Bet365 deviation only | 1.0024 | 199 | +2.0% ± 0.8% |
| anchored, rest days + promoted teams | 1.0028 | 340 | −0.1% ± 0.5% |

The weight on the goal model lies between −0.08 and −0.16 in every season.
A rule that bets on large model–market disagreements loses on the holdout
(yield −13% ± 4%, CLV −8%). This matches the evidence that betting odds are
the most accurate public forecasts (Štrumbelj, 2014). We conclude that the
model's disagreements with the sharp price are noise. A new feature may
influence bets only if it improves held-out log loss and CLV in this test.

### 4.3 Price-based value

Bets are selected when a regular bookmaker's price exceeds the de-vigged
Pinnacle price by at least 2% EV, with one bet per match and market,
following the price-discrepancy approach of Kaunitz, Zhong & Kriener (2017).
Over July 2019 – June 2026:

| selection | bets | flat yield | CLV |
|---|---|---|---|
| all tracked bookmakers, Pinnacle reference | 331 | +3.3% ± 11.4% | **+3.3% ± 0.6%** |
| Bet365 only | 127 | +14.7% ± 19.6% | +2.7% ± 0.9% |
| EV ≥ 5% | 77 | +29.5% ± 29.0% | +6.3% ± 1.3% |
| bookmaker consensus as reference | 162 | −12.9% ± 12.5% | +0.7% ± 0.8% |

CLV is positive in each season from 2019-20 to 2025-26 (+1.6% to +6.3%).
Over 2024–26, CLV is +2.6% ± 0.8% with Pinnacle as the reference, against
−1.4% ± 0.9% with the Betfair Exchange snapshot and −0.3% ± 1.1% with the
bookmaker consensus. Pinnacle is therefore the only admissible reference.
Accumulators are priced as $\prod_l (1 + \text{EV}_l) - 1$. Because the
bookmaker margin compounds per leg, they are positive only when composed of
individually positive legs (33 historical tickets: CLV +11.9% ± 3.2%, hit
rate 9%). Stakes follow fractional Kelly, shrunk for uncertainty in the fair
probability (Baker & McHale, 2013). See
[docs/value-betting.md](docs/value-betting.md).

## 5. Limitations

* The goal model observes only results, shots on target and corners. It
  has no information on lineups, injuries, suspensions, managerial changes or
  transfers.
* football-data prices are captured once per round. The historical backtest
  therefore under-represents opportunities and over-represents how long they
  last.
* Profit has not been shown to differ from zero; at observed volumes this
  requires thousands of bets. CLV is the primary diagnostic.
* The Odds API exposes only each bookmaker's main line. Totals and handicap
  prices at lines Pinnacle does not quote cannot be referenced.
* Bookmakers restrict accounts that consistently beat the closing price.
* The corner model improves only slightly on a constant base rate.

## 6. Reproduction

```bash
git clone https://github.com/bimataylorjoy/parlay.git && cd parlay
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                                   # ".[bayes]" adds PyMC/ArviZ

parlay fetch --league E0 E1 --from-season 2018 --to-season 2026
parlay backtest --from 2024-07-01                         # §4.1; --model poisson, --sot-blend 0
parlay compare  --from 2024-07-01                         # all models, identical folds
parlay tune --model dixon_coles --holdout-days 365        # hyperparameter selection
parlay calibrate outputs/backtest/predictions.csv         # temperature scaling
parlay anchor-test outputs/backtest/predictions.csv       # §4.2
parlay backtest-value --from 2019-07-01 --books ""        # §4.3; add --parlay for accumulators
```

Backtests write `predictions.csv`, `metrics.json` and `calibration.json` to
`outputs/backtest/`. Requires Python ≥ 3.10. The core dependencies are numpy
and scipy.

## 7. Software

### Command-line interface

| command | purpose |
|---|---|
| `fetch`, `ingest` | download and ingest football-data files with snapshots and manifests |
| `backtest`, `compare`, `tune`, `calibrate` | walk-forward evaluation (§3–4.1) |
| `anchor-test` | market-anchored information test (§4.2) |
| `backtest-value` | historical replay of the value rule, singles or accumulators (§4.3) |
| `predict` | price fixtures from a CSV (format: `examples/fixtures_template.csv`) |
| `odds sync` | store prospective prices from `fixtures.csv` or The Odds API |
| `value`, `ticket` | current value bets and accumulator pricing against Pinnacle |
| `bets add / update / report / list` | bet ledger: settlement, CLV, sequential verdict |
| `watch` | re-price each kickoff slot ~50 minutes before kickoff, after lineups are published |
| `daily` | results, prices, settlement and value bets in one run |
| `telegram` | Telegram interface to the value, ledger and watch functions |

`parlay <command> --help` lists all options. API keys are read from the
environment or from `~/.config/parlay/env`. The prospective workflow is
described in [docs/value-betting.md](docs/value-betting.md).

### Python

```python
from datetime import date
from parlay.data.database import ResearchDatabase
from parlay.models import fit_team_strength
from parlay.prediction.markets import asian_handicap, totals_probabilities

db = ResearchDatabase("data/parlay.sqlite")
cutoff = date(2026, 10, 24)
train = [m for m in db.load_matches() if m.home_goals is not None and m.date < cutoff]

model = fit_team_strength(train, model="dixon_coles", as_of=cutoff)
matrix = model.score_matrix("Arsenal", "Chelsea")
totals_probabilities(matrix, 2.75)   # {'over': ..., 'push': ..., 'under': ...}
asian_handicap(matrix, -0.25)
```

### Layout

```
src/parlay/
  data/         schemas, loaders, odds feeds, SQLite store, information sets
  models/       MLE engine, score models, team strength, corners, PyMC
  prediction/   market functionals, de-vig, Kelly, odds formats, fixture pricing
  evaluation/   walk-forward backtest, scoring rules, calibration, tuning
  betting/      reference prices, value finder, ledger, settlement, accumulators, watcher
  cli/, telegram/
tests/          unit, statistical (gradient checks, parameter recovery), leakage, CLI
docs/           methodology, value betting
```

`make check` runs ruff, mypy and pytest. CI runs them on Python 3.10–3.12.
Changes between versions, including migration from 0.1, are listed in
[CHANGELOG.md](CHANGELOG.md).

## References

* Baker, R. D. & McHale, I. G. (2013). Optimal betting under parameter uncertainty: improving the Kelly criterion. *Decision Analysis*, 10(3), 189–199.
* Constantinou, A. C. & Fenton, N. E. (2012). Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models. *Journal of Quantitative Analysis in Sports*, 8(1).
* Dixon, M. J. & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C*, 46(2), 265–280.
* Egidi, L., Pauli, F. & Torelli, N. (2018). Combining historical data and bookmakers' odds in modelling football scores. *Statistical Modelling*, 18(5–6), 436–459.
* Epstein, E. S. (1969). A scoring system for probability forecasts of ranked categories. *Journal of Applied Meteorology*, 8(6), 985–987.
* Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of ICML*.
* Hubáček, O., Šourek, G. & Železný, F. (2019). Exploiting sports-betting market using machine learning. *International Journal of Forecasting*, 35(2), 783–796.
* Kaunitz, L., Zhong, S. & Kriener, J. (2017). Beating the bookies with their own numbers — and how the online sports betting market is rigged. arXiv:1710.02824.
* Maher, M. J. (1982). Modelling association football scores. *Statistica Neerlandica*, 36(3), 109–118.
* Shin, H. S. (1993). Measuring the incidence of insider trading in a market for state-contingent claims. *The Economic Journal*, 103(420), 1141–1153.
* Štrumbelj, E. (2014). On determining probability forecasts from betting odds. *International Journal of Forecasting*, 30(4), 934–943.

## License

MIT. This is research software: the results above describe historical
data and do not guarantee future returns.
