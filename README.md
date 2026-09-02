# Parlay Research — Time-Safe Football Probability Engine

<p align="center">
  <a href="https://github.com/bimataylorjoy/parlay/actions"><img src="https://img.shields.io/badge/tests-82%20passed-brightgreen" alt="tests"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="python"></a>
  <a href="https://github.com/bimataylorjoy/parlay/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/version-0.1.0-orange" alt="version"></a>
  <a href="https://github.com/bimataylorjoy/parlay/blob/main/README.md"><img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license"></a>
  <a href="https://github.com/bimataylorjoy/parlay"><img src="https://img.shields.io/badge/coverage-EPL%20%2B%20Championship-blueviolet" alt="coverage"></a>
</p>

<p align="center">
  <strong>Research-grade, anti-leakage framework for football score probabilities — from Poisson to Dixon-Coles to Corners, with fair-odds, Kelly and full backtesting.</strong><br/>
  <em>Research-only. Does not place bets or provide financial advice.</em>
</p>

---

## Table of Contents

- [Why Parlay](#why-parlay)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Markets & Maths](#markets--maths)
- [Outputs](#outputs)
- [Data Layout](#data-layout)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Development & Testing](#development--testing)
- [Contributors](#contributors)
- [License](#license)

---

## Why Parlay

Most public betting models leak future information, overfit to goals, and only support `Over 2.5`. Parlay is built for **sharpness over automation**:

* **Strict temporal correctness** — every feature, weight and odd is gated by `as_of < fixture_date`.
* **Academic-grade models** — Independent Poisson, Dixon-Coles low-score correction, Negative-Binomial overdispersion, Bayesian HMC, plus a dedicated **CornerStrengthModel**.
* **Full market coverage** — 1X2, Double Chance, BTTS, Win-to-Nil, **arbitrary & quarter O/U** (2.25/2.75 split-stake), Correct Score, Corners O/U & 1X2.
* **Market intelligence** — Shin/Power/Multiplicative de-vig, EV, fractional Kelly, CLV vs Pinnacle closing.
* **Not a bot** — optimized for **manual, personal use**: you compare `fair_odds = 1/p` vs your bookmaker.

> **Accuracy (expanding window, 40 folds, 1504 tests):** Poisson `log_loss 1.064 brier 0.644` | Pinnacle closing `0.98/0.58`. Hit-rate most-likely 1X2 **54.1%** vs naive home 44.0%.

---

## Features

| Layer | What you get |
|---|---|
| **Data** | `football-data.co.uk` E0 (EPL) + E1 (Championship) ingestion, SHA256 snapshots, SQLite time-safe DB, canonical team aliases |
| **Features** | Rolling `window=5` goals/corners/shots/win_rate + `rest_days` (congestion) |
| **Models** | Poisson, Dixon-Coles `ρ∈[-0.3,0.3]`, NB2 `Var=μ+μ²/φ`, Bayesian HMC, Corners Poisson/NB, time-decay `w=0.5^{age/half_life}`, SOT blending `g_eff=(1-α)g+α·sot·0.31` |
| **Markets** | `P(i,j)` 11×11 → 1X2, `P(over)=Σ M[total>line]`, BTTS, Asian Handicap quarter, Correct Score `P(i-j)=M[i,j]`, grouped, Corners 21×21 |
| **Value** | `edge = p_model - p_market`, `EV = p·odds-1`, `Kelly = EV/(odds-1)·fraction` cap `0.5%`, `fair=1/p` |
| **Eval** | Expanding window, log loss/Brier, temperature scaling, flat-stake, grid tune, compare |
| **CLI** | `ingest`, `backtest`, `compare`, `tune`, `predict`, `sync-sportmonks`, `predict-sportmonks`, `calibrate` |

---

## Architecture

```
CSV (E0/E1) + Sportmonks API  ─┐
                               ├─→ [Schemas+Validation+Normalization] ─→ [Loaders+Sources] ─→ [SQLite]
                               │                                              ↓
                               │                                   [Features: rolling pre-match]
                               │                                              ↓
                               └─→ [TeamStrengthModel / CornerStrengthModel] ←── MLE (L-BFGS-B, L2) / Heuristic / Bayesian
                                                                    ↓
                                                            Score Matrix P(i,j)
                                                                    ↓
                                   ┌──────────────┬──────────┬──────┴──────┬────────────┬──────────┐
                                   1X2        Over/Under  BTTS/WinToNil  CorrectScore  Corners
                                   └──────────────┴──────────┴─────────────┴────────────┴──────────┘
                                                                    ↓
                                                        De-vig (Shin) → EV/Kelly → Dashboard
                                                                    ↓
                                                        Backtest / Calibration / Tuning
```

* **Score matrix** is the single source of truth — every market is a sum over `M`.
* **Time-safe by design** — `history[team]` only uses `date < fixture_date`.

---

## Installation

```bash
git clone https://github.com/bimataylorjoy/parlay.git
cd parlay

# minimal (numpy+scipy)
pip install -e .

# with Bayesian HMC (optional)
pip install -e ".[dev]"  # plus pymc if needed: pip install pymc

# verify
PYTHONPATH=src python3 -m pytest -q  # 82 passed
```

**Requirements:** Python `>=3.10`, `numpy>=1.24`, `scipy>=1.10`.

---

## Quick Start

### 1. Ingest EPL + Championship

```bash
# EPL 8 seasons (already in data/snapshots)
PYTHONPATH=src python3 -m parlay.cli ingest \
  data/raw/snapshots/*E0.csv --competition EPL --season multi --database data/parlay.sqlite

# Championship for promoted teams (Coventry, Hull, Wrexham, Charlton, QPR, WBA...)
PYTHONPATH=src python3 -m parlay.cli ingest \
  data/raw/snapshots/*E1.csv --competition Championship --season multi
```

### 2. Predict next GW (EPL GW3 + Championship GW4 early-morning)

```bash
# EPL GW3 (Man City vs Coventry is now correct)
cat upcoming_fixtures.csv
# Date,HomeTeam,AwayTeam,B365H,B365D,B365A,B365>2.5,B365<2.5
# 05/09/2026,Manchester City,Coventry City,1.28,5.50,11.00,1.50,2.50

PYTHONPATH=src python3 -m parlay.cli predict upcoming_fixtures.csv \
  --model dixon_coles --estimator mle --half-life-days 365 --sot-weight 0.3

# Championship dini hari 01:45 WIB (4 laga)
PYTHONPATH=src python3 -m parlay.cli predict upcoming_fixtures_championship.csv \
  --model dixon_coles --estimator mle --output outputs/predictions_championship_early_simple.csv

# Full 95-col CSV for Excel
PYTHONPATH=src python3 -m parlay.cli predict upcoming_fixtures.csv \
  --model dixon_coles --estimator mle --output outputs/predictions_gw3_full.csv
```

**Sample dashboard** `src/parlay/cli.py:450`:

```
05/09/2026 | Manchester City vs Coventry City | 82.3% 13.7%  4.1% | 50.1% 49.9% | Home @1.28 EV +5.3%
  └─ xG: Man City 2.33 - 0.35 Coventry | BTTS Yes 26.4% fair 3.78 | Home Win to Nil 63.8%
     O/U 2.5 → Over 50.1% fair 2.00 | O/U 2.75 → Over 39.1% fair 2.56 push 11%
     Correct Score Top5: 2-0 18.7% (5.35) | 1-0 16.0% (6.24) | 3-0 14.5%
     xCorners: Man City 9.6-2.5 Coventry total 12.0 | Corners O/U 10.5 Over 60.9% fair 1.64
```

### 3. Backtest & Compare

```bash
PYTHONPATH=src python3 -m parlay.cli backtest --model dixon_coles --estimator mle \
  --half-life-days 730 --output-dir outputs/backtest/epl-dixon-coles
# → outputs/backtest/epl-dixon-coles/metrics.json, predictions.csv, calibration.json

PYTHONPATH=src python3 -m parlay.cli compare --estimator mle
# → outputs/compare/model_comparison.json  (poisson vs dixon_coles vs NB vs market)

PYTHONPATH=src python3 -m parlay.cli tune --model poisson
# grid search half_life
```

---

## CLI Reference

| Command | Key Args | Description |
|---|---|---|
| `ingest <csv...>` | `--competition`, `--season`, `--database`, `--manifest` | SHA256 snapshot + SQLite upsert |
| `backtest` | `--model`, `--estimator`, `--half-life-days`, `--sot-weight`, `--initial-train-days 730`, `--test-days 30` | Expanding window, writes `predictions.csv` |
| `compare` | `--estimator`, `--half-life-days` | All 3 models on identical folds |
| `tune` | `--model` | Half-life grid search |
| `predict <fixtures.csv>` | `--model`, `--estimator`, `--kelly-fraction 0.25`, `--max-stake 0.005`, `--min-ev 0.02`, `--output` | Manual dashboard + CSV |
| `sync-sportmonks` | `--start-date`, `--end-date`, `--token` | Fetch fixtures+totals (EPL `8`, Championship `9`) `src/parlay/data/sportmonks.py:15` |
| `calibrate <predictions.csv>` | — | Temperature scaling `T` |

**Fixture CSV input** `src/parlay/data/normalization.py:46` — `Date,HomeTeam,AwayTeam` required; optional `B365H,D,A, B365>1.5/<1.5, B365>2.5, B365BTTS_Y/N, B365C>9.5` etc. Full template: `upcoming_fixtures_template.csv:1`.

---

## Markets & Maths

**Expected goals** `src/parlay/models/team_strength.py:31`:
```
λ_home = exp( μ + γ + α_home - β_away )
λ_away = exp( μ + α_away - β_home )
```
`μ` intercept, `γ` home advantage, `α` attack, `β` defense, sum-to-zero.

**Time decay** `src/parlay/models/team_strength.py:132`: `w = 0.5^{age / half_life}`. **SOT blend**: `g_eff = (1-α)g + α·sot·0.31`.

**Poisson** `src/parlay/models/poisson.py:8`: `P(k;λ)=λ^k e^{-λ}/k!` → `M = outer(P_home,P_away)/sum`.

**Dixon-Coles** `src/parlay/models/dixon_coles.py:27`: `τ(0,0)=1-λμρ, τ(0,1)=1+λρ, τ(1,0)=1+μρ, τ(1,1)=1-ρ` with bounds `ρ≤1/(λμ)`, `ρ≥-1/λ`.

**NB2** `src/parlay/models/negative_binomial.py:8`: `Var=μ+μ²/φ`, `φ = μ̄²/(Var-μ)` MoM.

**MLE** `src/parlay/models/mle.py:79`: `NLL = -Σ w·(g·logλ-λ + low·logτ) + 0.5·l2·Σ(α²+β²)`, `L-BFGS-B` bounds `α,β∈[-3,3]`.

**Corners** `src/parlay/models/corners.py:24` separate `λ_c` with `Var=μ+μ²/φ`, `21×21` matrix.

**Markets** `src/parlay/prediction/markets.py:231`, `src/parlay/models/poisson.py:28`: `P_home=Σ tril(M)`, `P_over(line)=Σ M[total>line]`, `P_btts=Σ M[1:,1:]`, `P_cs[i-j]=M[i,j]`, quarter `2.75 = 0.5·2.5+0.5·3.0`.

**De-vig** `src/parlay/prediction/markets.py:47`: Multiplicative, Power `Σ p^k=1`, Shin `p_i(z)=(√(z²+4(1-z)β_i²/Σβ)-z)/2(1-z)` (default, corrects favourite-longshot bias).

**Value** `src/parlay/prediction/markets.py:128`: `edge = p_model - p_market`, `EV = p·odds-1`, `Kelly = EV/(odds-1)·fraction` cap `0.5%`, `fair=1/p`.

---

## Outputs

`predict --output` writes 95 columns (simple 25-col variant: `outputs/predictions_gw3_simple.csv:1`):

* **1X2**: `home_win_prob, fair_home, double_chance_1x`
* **O/U**: `over_1_5/2_5/2_75/3_5_prob + fair`
* **BTTS**: `btts_yes_prob, home_win_to_nil`
* **Correct Score**: `cs_2_0_prob, fair_cs_2_0, cs_other, top_scorelines`
* **Expectation**: `exp_home_goals, exp_total_corners`
* **Corners**: `corners_over_10_5_prob, fair_corners_over_10_5, corners_home_most`
* **Audit**: `B365H, recommendation, recommendation_ev, stake_fraction`

Example `outputs/predictions_championship_early_simple.csv:1` — 4 laga dini hari `01:45 WIB`:
```
QPR vs Cardiff → QPR 48.4% fair 2.07 vs 2.40 EV +16% | Under 10.5 corners 63% fair 1.58
Burnley vs Middlesbrough → Burnley 61.3% fair 1.63 vs 2.00 EV +22% | BTTS No 67% fair 1.49
```

---

## Data Layout

```
data/parlay.sqlite      # SQLite (teams, matches, odds_snapshots, feature_snapshots, predictions)
data/raw/E0_*.csv        # football-data.co.uk EPL (keep)
data/raw/snapshots/      # immutable SHA snapshots (ignored in git)
data/snapshots/*.json    # ingestion manifests
outputs/                 # backtest / predictions (ignored, regenerate)
```

Ingestion is **append-only** with `ON CONFLICT(match_id) DO UPDATE` `src/parlay/data/database.py:126`.

---

## Evaluation

* **Temporal** `src/parlay/evaluation/temporal.py`: `expanding_window(initial_train 730d, test 30d)`.
* **Metrics** `src/parlay/evaluation/metrics.py:22`: `log_loss=-ln(p_actual)`, `Brier=Σ(p-1_{actual})²`.
* **Calibration** `src/parlay/evaluation/calibration.py:10`: `p^{1/T}/Σ p^{1/T}`.
* **Tuning** `src/parlay/evaluation/tuning.py`: half-life grid search.
* **Strategy** `evaluate_flat_stake(min_edge 0.03, min_ev 0.02)` + Pinnacle closing baseline `src/parlay/evaluation/backtest.py:166`.

---

## Project Structure

```
src/parlay/
  data/         schemas, normalization, validation, loaders, sources, database, ingestion, sportmonks
  models/       poisson, dixon_coles, negative_binomial, mle, bayesian, team_strength, corners
  features/     historical (rolling pre-match)
  evaluation/   backtest, metrics, market, calibration, temporal, compare, tuning, serialization
  prediction/   markets (de-vig, Kelly, Asian, Correct Score)
tests/          82 tests (unit + integration + CLI)  →  pytest
scripts/        run_epl_totals, analyze_asian_totals
```

---

## Development & Testing

```bash
PYTHONPATH=src python3 -m pytest -q          # 82 passed
PYTHONPATH=src python3 -m pytest tests/test_markets.py -v
PYTHONPATH=src python3 -m parlay.cli --help
```

* **No Docker/CI required** — pure Python, SQLite, `numpy/scipy`.
* **Config** via CLI flags; `half_life_days`, `sot_weight`, `l2_reg` tune sharpness, not automation.

---

## Contributors

* **bimataylorjoy** — [@bimataylorjoy](https://github.com/bimataylorjoy) (maintainer, multi-market engine, Championship ingestion, CLI dashboard) `pyproject.toml:12`

Contributions welcome — open an issue or PR. For betting syndicate-grade accuracy, see `docs/ROADMAP.md` (planned: player-level xG, rest-day congestion, live in-play).

---

## License

MIT — see `LICENSE` (research-only, no warranty).

---

## Disclaimer

This is a **research framework**, not a tipster service. Probabilities are estimates from historical goals/corners. The market (Pinnacle closing `log_loss 0.98`) is still sharper than the model (`1.06`). Use `fair odds` as a **screening filter** with strict bankroll (≤0.5% per bet) and always compare to your bookmaker's live price.

