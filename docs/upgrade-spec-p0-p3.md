# Parlay — Statistical Correctness & Information Quality Upgrade

Work directly inside the existing `parlay` repository.

This is a football probabilistic modelling and market research engine covering:

- Poisson team strength models
- Dixon-Coles
- Negative Binomial
- Bayesian HMC
- historical features
- walk-forward backtesting
- calibration
- de-vig
- EV
- Kelly
- market probability comparison
- markets such as 1X2, Over/Under, BTTS, Asian totals, Correct Score, and Corners

Do not rewrite the repository from scratch.

First inspect the current repository and treat the current implementation as the source of truth. Preserve what is already correct and make the minimum coherent architectural changes necessary.

The priority hierarchy is:

```text
Information Quality
        ↓
Temporal Correctness
        ↓
Statistical Model
        ↓
Uncertainty Quantification
        ↓
Probability Distribution
        ↓
Calibration
        ↓
Market Comparison
        ↓
Research Decision
```

Do not improve the project by merely adding more ML models. Do not add XGBoost, Random Forest, LSTM, Transformer, or generic AI prediction layers unless there is a mathematically justified reason.

Prioritize better information and statistical correctness over model complexity.

---

## 0. Inspect Before Modifying

Before editing:

1. Inspect the complete repository structure.
2. Identify every model implementation.
3. Trace the data flow from ingestion to prediction.
4. Trace temporal semantics used by:
   - database queries
   - feature generation
   - training
   - backtesting
   - calibration
5. Inspect the existing tests.
6. Identify public APIs that should remain backward compatible.
7. Run the existing test suite.
8. Create an internal implementation plan based on the actual repository.

Do not blindly apply assumptions from this specification if the current repository already has a better implementation.

---

## 1. Create a Canonical Information Set

Introduce one canonical abstraction defining what information was available at prediction time.

Conceptually:

```python
InformationSet(
    as_of: datetime,
    competition: str | None = None,
)
```

or an equivalent design appropriate for the repository.

Every historical query should answer:

> Was this information knowable before `as_of`?

Avoid scattered conditions such as:

```python
match.date <= as_of
```

across unrelated modules.

Centralize temporal eligibility.

Distinguish clearly:

```text
fixture kickoff time
        ≠
result availability time
        ≠
forecast timestamp
```

A match should only be usable if its result was available before the forecast timestamp.

If exact result timestamps are unavailable, implement and document a conservative fallback such as:

```text
known_at = kickoff_at + estimated_match_duration
```

Do not silently assume all matches on the same calendar date were known.

Ensure all feature generation satisfies:

```text
feature_timestamp <= forecast_timestamp
```

---

## 2. Upgrade Temporal Backtesting

Audit the current walk-forward implementation.

Support two explicit evaluation modes.

### Fixed-origin

```text
TRAIN
│
├──── frozen model
│
└──────── TEST WINDOW
```

Useful for research where the model remains frozen.

### Rolling-origin

Preferred production-like evaluation:

```text
Day 1
Train on all known history
Predict Day 1

Day 2
Update with Day 1 results
Predict Day 2

Day 3
Update with Day 1 + Day 2 results
Predict Day 3
```

Fixture-level chronological updates are acceptable when computationally practical.

The API must explicitly expose the selected mode.

Add metadata such as:

```python
{
    "evaluation_mode": "...",
    "forecast_timestamp_policy": "...",
    "model_update_frequency": "...",
}
```

Do not silently describe fixed-origin evaluation as a dynamic production simulation.

---

## 3. Fix Dixon-Coles Rho Estimation

Audit the Dixon-Coles likelihood carefully.

If the current implementation clips a global `rho` independently for each match, remove that behavior.

A global parameter must remain globally consistent.

Do not perform:

```text
global rho
↓
clip differently per match
↓
evaluate likelihood
```

This silently changes the parameter meaning and creates a piecewise objective.

Prefer determining a globally valid optimization domain from the training data:

```python
rho_lower = max(match_lower_bounds)
rho_upper = min(match_upper_bounds)
```

when mathematically appropriate.

Alternatively use a numerically stable parameterization that guarantees valid Dixon-Coles correction factors without mutating `rho` per observation.

Requirements:

- no per-match silent parameter mutation
- no negative probability corrections
- documented rho domain
- unit tests around boundary conditions
- likelihood continuity tests
- optimizer stability tests

Training likelihood and prediction correction must use identical mathematical semantics.

---

## 4. Unify Training and Prediction Domains

Audit all clipping and bounds.

Eliminate situations where:

```text
training model domain != prediction model domain
```

For example, training and prediction must not clip `log(lambda)` to different ranges.

Prefer:

- stable numerical implementations
- regularization
- parameter bounds
- safe exponentiation

over arbitrary clipping.

If clipping remains necessary:

1. document why
2. use identical semantics during fitting and inference
3. add consistency tests

---

## 5. Implement True Bayesian Posterior Predictive Inference

Do not collapse the posterior into parameter means and then claim uncertainty propagation.

Avoid:

```text
posterior
↓
parameter means
↓
single lambda
↓
single score matrix
```

Implement posterior predictive inference:

```text
Posterior draws
      │
      ├── θ₁ → λ₁ → ScoreMatrix₁
      ├── θ₂ → λ₂ → ScoreMatrix₂
      ├── θ₃ → λ₃ → ScoreMatrix₃
      │
      ▼
Aggregate posterior predictive distribution
```

Mathematically approximate:

```text
P(Y_new | D)
=
∫ P(Y_new | θ) P(θ | D) dθ
```

using Monte Carlo posterior samples.

Support a richer result conceptually similar to:

```python
PredictionDistribution(
    mean_probability=...,
    uncertainty=...,
    posterior_samples=...,
)
```

Do not expose unnecessary massive sample arrays by default. Provide configurable posterior draw subsampling.

For Bayesian predictions expose uncertainty where meaningful:

```text
lambda_home:
- mean
- credible interval

lambda_away:
- mean
- credible interval

market probabilities:
- posterior mean
- credible interval
```

Clearly distinguish parameter uncertainty from event randomness.

---

## 6. Fix Bayesian Prior Symmetry

Audit identifiability constraints.

Do not use an asymmetric prior construction such as:

```python
attack_last = -sum(other_attacks)
```

when priors are assigned only to the first `n-1` teams.

All teams should remain exchangeable under the prior.

Prefer:

```python
attack_raw ~ Normal(0, sigma, shape=n_teams)

attack = attack_raw - mean(attack_raw)
```

Likewise for defense.

Use non-centered parameterization if sampling diagnostics justify it.

Expose Bayesian diagnostics:

- R-hat
- effective sample size
- divergences
- sampler warnings

Do not silently accept HMC fits with poor convergence.

Diagnostics should be available from the fitted model result.

---

## 7. Implement Proper Negative Binomial Estimation

Audit the current Negative Binomial implementation.

If it currently follows:

```text
Fit Poisson
↓
calculate residuals
↓
estimate dispersion separately
↓
call it Negative Binomial
```

upgrade it.

Estimate mean parameters and dispersion coherently under a proper joint NB likelihood.

Conceptually optimize:

```text
theta =
{
    attack,
    defense,
    intercept,
    home_advantage,
    dispersion
}
```

Use a numerically stable constrained parameterization such as:

```python
dispersion = exp(log_dispersion)
```

or another justified transform.

Document:

- exact NB parameterization
- NB1 vs NB2
- variance relationship
- dispersion interpretation

Add synthetic tests for:

- overdispersed data
- dispersion recovery
- Poisson limiting behavior

---

## 8. Remove Fractional Effective Goals as Generative Targets

Audit any logic similar to:

```text
effective_goals =
(1 - alpha) * goals
+
alpha * SOT * conversion
```

Do not feed fractional pseudo-goals into a Poisson count likelihood as though they were observed counts.

Replace this with a statistically coherent design.

Priority:

### Option A
Use reliable, timestamp-safe xG/xGA data.

### Option B
Use SOT-derived information as explanatory covariates:

```text
log(lambda)
=
team_strength_terms
+
beta_sot * recent_sot_signal
```

### Option C
Introduce a latent attacking quality model if architecture justifies it.

Do not hardcode a universal conversion constant without empirical estimation.

Any conversion rate must be:

- estimated from historical data
- league-aware when appropriate
- time-safe
- shrinkage-regularized

Improve information quality without corrupting the observation likelihood.

---

## 9. Build a Better Information and Feature Architecture

Prioritize information quality over model proliferation.

Design the feature architecture so additional data sources can be integrated cleanly.

Potential feature groups:

### Match performance

- goals
- xG
- xGA
- shots
- shots on target
- non-penalty xG

### Context

- rest days
- fixture congestion
- home/away
- travel distance when reliable
- competition effects
- season phase

### Team dynamics

- time-decayed form
- attack trend
- defensive trend
- promoted/relegated team uncertainty

### Availability

Only when reliable and timestamp-safe:

- lineup strength
- key player availability
- confirmed lineups

Do not add features merely because data exists.

Where practical introduce feature provenance:

```python
FeatureValue(
    value=...,
    source=...,
    computed_at=...,
    available_at=...,
)
```

At minimum document availability assumptions for every data source.

---

## 10. Add Dynamic Team Strength as a Research Option

Do not replace the existing stable baseline without benchmarking.

Introduce a research-capable dynamic alternative.

Conceptually:

```text
Attack[i, t]
~
Normal(
    Attack[i, t-1],
    sigma_attack_evolution
)

Defense[i, t]
~
Normal(
    Defense[i, t-1],
    sigma_defense_evolution
)
```

Possible approaches:

- state-space model
- Bayesian random walk
- dynamic latent strength model
- exponential time-decay baseline

Keep simpler models as benchmarks.

The architecture should allow answering:

> Does additional dynamic complexity improve out-of-sample performance?

Do not assume complexity is improvement.

---

## 11. Implement Adaptive Score Matrix Truncation

Audit score matrix truncation.

Do not use a fixed `max_goals` and blindly renormalize when significant tail probability exists.

Implement adaptive truncation.

Conceptually:

```python
max_goals = smallest k
such that
P(X > k) < epsilon
```

Use configurable epsilon.

Track metadata such as:

```python
tail_mass_home
tail_mass_away
joint_tail_mass_estimate
truncation_epsilon
```

Avoid silently normalizing away meaningful probability mass.

This is especially important for:

- high lambda predictions
- corners
- posterior predictive mixtures

Ensure numerical stability under extreme rates.

---

## 12. Audit and Correct the Corners Model Independently

Treat corners as a separate modelling problem.

Do not blindly transfer goals assumptions.

Audit:

- likelihood
- dispersion
- support truncation
- parameter constraints
- clipping
- market mapping

If the model claims Negative Binomial but fits Poisson parameters and only applies NB during prediction, either implement a joint NB likelihood or correct the terminology.

Remove arbitrary hard floors on lambda unless empirically justified.

Corners should have independent diagnostics and backtests.

Do not assume improvements in goals modelling transfer automatically to corners.

---

## 13. Implement Temporally Safe Calibration

Calibration must be temporally isolated.

Do not:

```text
generate predictions over all backtest history
↓
find optimal temperature on all predictions
↓
report calibrated performance on the same predictions
```

Implement explicit protocols:

```text
TRAIN WINDOW
      ↓
fit model

CALIBRATION WINDOW
      ↓
fit calibration parameters

TEST WINDOW
      ↓
final evaluation
```

For rolling evaluation:

```text
Past
├── model training
├── calibration fitting
└── current test
```

Support calibration methods only when justified:

- temperature scaling
- isotonic regression
- beta calibration

Evaluate:

- reliability diagrams
- Expected Calibration Error
- Brier score
- Log loss

Report calibration separately for:

- 1X2
- Over/Under
- BTTS
- corners

Do not rely only on global calibration metrics.

---

## 14. Implement Nested Temporal Hyperparameter Tuning

Audit all hyperparameter selection.

Parameters may include:

- half-life
- regularization
- feature weights
- model variants
- dispersion settings
- calibration parameters

Do not select these on the same untouched evaluation period used for final performance claims.

Preferred protocol:

```text
Development period
    ↓
inner walk-forward tuning
    ↓
lock configuration
    ↓
final untouched holdout
```

Use nested walk-forward evaluation when practical.

Backtest metadata should identify:

```text
tuned_on
evaluated_on
hyperparameters
selection_metric
```

Prevent accidental repeated peeking at final holdout performance.

---

## 15. Upgrade to an Uncertainty-First Prediction API

Do not return only:

```python
{
    "over_2_5": 0.578
}
```

Support richer prediction output:

```python
PredictionResult(
    probabilities=...,
    expected_rates=...,
    uncertainty=...,
    model_metadata=...,
    calibration_metadata=...,
)
```

The API should preserve simple access to probabilities while making uncertainty and assumptions available.

---

## 16. Keep Probability Modelling Separate from Market Decisions

Maintain a strict separation:

```text
Probability Model
        ↓
Produces probabilistic beliefs

Market Layer
        ↓
Interprets bookmaker prices

Decision Research Layer
        ↓
Compares them
```

Do not contaminate the probability model with bookmaker odds unless implementing an explicitly documented market-informed model.

Maintain or improve de-vig methods such as:

- multiplicative
- power
- Shin

Attach method metadata.

When comparing:

```text
Model Probability
vs
Market Fair Probability
```

do not treat raw implied probabilities as fair probabilities without accounting for margin.

---

## 17. Treat Extreme Model-Market Disagreement as an Anomaly Signal

Do not automatically interpret a large model edge as an opportunity.

Add an uncertainty/anomaly layer.

Conceptually:

```python
DecisionDiagnostics(
    model_probability=0.79,
    market_probability=0.51,
    disagreement=0.28,
    uncertainty="high",
    calibration_regime="poor",
    anomaly_flags=[
        "extreme_model_market_disagreement",
        "outside_historical_calibration_regime",
    ],
)
```

Potential checks:

- probability outside historical calibration range
- model-market disagreement z-score
- feature distribution shift
- extreme lambda
- insufficient historical sample
- promoted/new team uncertainty
- wide posterior uncertainty
- data missingness

Support research states:

```text
RESEARCH_SIGNAL
WATCH
PASS
REJECT
```

Do not automatically convert disagreement into betting advice.

---

## 18. Add Regime Analysis Infrastructure

Move beyond the simplistic objective:

> Find one globally best football prediction model.

Support conditional evaluation by regimes such as:

- league
- season phase
- home/away strength imbalance
- favorite/underdog
- high-total vs low-total matches
- early season
- promoted teams
- rest asymmetry
- market type

For each regime evaluate:

- sample size
- log loss
- Brier score
- calibration
- market comparison

Do not declare performance differences from tiny samples.

Include minimum sample thresholds and confidence intervals where appropriate.

---

## 19. Add Statistical Diagnostics

Every model should expose diagnostics appropriate to its methodology.

### Frequentist

- optimizer success
- gradient information where available
- parameter identifiability
- convergence warnings

### Bayesian

- R-hat
- ESS
- divergences
- posterior predictive checks

### Distribution models

- mean vs variance diagnostics
- overdispersion checks
- residual diagnostics

### Evaluation

- log loss
- Brier
- calibration
- sharpness
- confidence intervals via bootstrap where appropriate

Never silently hide convergence failures.

A failed fit must propagate an explicit warning or failure state.

---

## 20. Add Comprehensive Tests

Do not consider the work complete without tests.

Add tests for:

### Temporal integrity

- future matches never enter history
- same-day fixture ordering respects timestamps
- unavailable results cannot enter the information set

### Dixon-Coles

- rho validity
- no negative correction probabilities
- globally consistent rho
- likelihood continuity around bounds

### Bayesian

- exchangeable prior behavior
- posterior predictive probabilities sum correctly
- uncertainty propagation differs from posterior-mean plug-in when expected

### Negative Binomial

- synthetic overdispersion
- dispersion recovery
- Poisson limit behavior

### Score matrices

- probabilities sum approximately to one
- adaptive truncation controls tail mass
- extreme lambda remains numerically stable

### Calibration

- calibration and test windows are temporally disjoint
- calibration fitting cannot see future test outcomes

### Backtesting

- fixed-origin semantics
- rolling-origin semantics
- metadata accurately reflects evaluation mode

Use synthetic data where necessary.

Do not depend on network APIs in unit tests.

---

## 21. Maintain Practical Performance

Do not make the repository unusably slow.

Potential controls:

- model caching keyed by information cutoff
- configurable update frequency
- posterior draw subsampling
- parallel-safe execution where appropriate
- deterministic random seeds
- separation between research mode and fast baseline mode

Avoid obvious unnecessary repeated fitting.

---

## 22. Update Documentation Honestly

Update README and technical documentation.

Do not overclaim.

Document precisely:

1. model assumptions
2. information availability assumptions
3. temporal evaluation protocol
4. calibration protocol
5. hyperparameter selection protocol
6. uncertainty semantics
7. known limitations

Only claim:

> Bayesian uncertainty quantification

if posterior predictive uncertainty is genuinely propagated.

Only claim:

> leakage-safe

with exact timestamp assumptions documented.

Add a concise architecture diagram.

---

## 23. Preserve Backward Compatibility

Do not break existing users unnecessarily.

If public APIs need to change:

1. provide migration paths where practical
2. document breaking changes
3. prefer explicit configuration

For example:

```python
BacktestConfig(
    mode="rolling_origin",
    calibration_mode="separate_window",
)
```

Do not silently change semantics.

---

## Implementation Priority

Follow this dependency order unless repository inspection strongly suggests otherwise.

### P0 — Mathematical correctness

1. Audit and fix Dixon-Coles rho
2. Unify training/prediction domains
3. Fix Bayesian prior symmetry
4. Implement posterior predictive inference
5. Adaptive score matrix truncation

### P1 — Evaluation correctness

6. InformationSet abstraction
7. Timestamp-safe data eligibility
8. Explicit fixed vs rolling-origin backtests
9. Calibration temporal separation
10. Nested temporal hyperparameter tuning

### P2 — Information and model quality

11. Remove fractional effective goals
12. Improve feature/covariate architecture
13. Proper Negative Binomial fitting
14. Independently improve corners model
15. Add dynamic strength research model

### P3 — Research intelligence

16. Uncertainty-first prediction API
17. Anomaly detection
18. Regime analysis
19. Statistical diagnostics

---

## Critical Constraints

You must not:

- rewrite the repository unnecessarily
- delete existing models without benchmarking replacements
- add ML models for cosmetic sophistication
- introduce future information leakage
- tune on the final holdout period
- silently ignore optimizer failures
- silently clamp global parameters differently per observation
- call posterior means full Bayesian uncertainty
- feed fractional pseudo-counts into Poisson likelihood without explicit quasi-likelihood justification
- claim an edge merely because model probability differs from bookmaker odds
- replace simple baselines without demonstrating out-of-sample improvement

---

## Definition of Done

The task is complete only when:

### Code

- [ ] Existing architecture inspected first
- [ ] Mathematical P0 issues fixed
- [ ] Temporal information semantics centralized
- [ ] Backtest modes explicitly defined
- [ ] Bayesian posterior predictive inference implemented
- [ ] Bayesian priors are exchangeable
- [ ] Dixon-Coles rho is globally coherent
- [ ] NB methodology is statistically correct or honestly described
- [ ] Score truncation is adaptive
- [ ] Calibration is temporally isolated
- [ ] Hyperparameter tuning is temporally valid

### Data and information

- [ ] Feature timestamps documented
- [ ] No known future leakage path remains
- [ ] Fractional effective-goal targets removed or explicitly justified
- [ ] Information provenance architecture improved

### Evaluation

- [ ] Fixed-origin and rolling-origin semantics tested
- [ ] Final evaluation can remain untouched by tuning
- [ ] Calibration metrics reported per market
- [ ] Regime analysis infrastructure exists

### Reliability

- [ ] New unit tests added
- [ ] Existing tests pass unless intentionally migrated
- [ ] Numerical stability tested
- [ ] Failed optimization is visible

### Documentation

- [ ] README updated
- [ ] Model assumptions documented
- [ ] Known limitations documented
- [ ] No exaggerated scientific claims

---

## Final Report Required

After implementation, provide a detailed engineering report containing:

### 1. Original repository audit

Summarize what was discovered before changes.

### 2. Changes implemented

For every major change explain:

```text
Problem
↓
Why it was statistically or architecturally incorrect
↓
Implementation change
↓
Files changed
↓
Expected impact
```

### 3. Mathematical changes

Explain equations and assumptions for:

- Dixon-Coles
- Bayesian posterior predictive inference
- Negative Binomial
- adaptive truncation
- calibration

### 4. Temporal integrity report

Explicitly answer:

```text
What information is available at prediction time?
What timestamp assumptions remain?
What leakage risks still exist?
```

### 5. Before vs after

Compare:

- architecture
- statistical validity
- temporal correctness
- uncertainty handling
- evaluation rigor

Do not fabricate performance improvements.

If a change does not improve out-of-sample performance, report that honestly.

### 6. Remaining limitations

Rank unresolved issues:

```text
P0
P1
P2
Research ideas
```

### 7. Test report

Include:

- tests added
- tests passed
- tests migrated
- known failures

---

## Guiding Philosophy

Evolve this repository into:

> A scientifically defensible probabilistic football forecasting and market research framework.

Do not turn it into an AI football betting tip generator.

The primary output should not be a "high confidence pick".

The primary output should be a well-calibrated probability distribution with transparent uncertainty, valid temporal evaluation, explicit assumptions, and the ability to recognize when the model should distrust itself.

Inspect first.

Preserve what is correct.

Fix what is mathematically wrong.

Prefer evidence over sophistication.

Benchmark every additional complexity against simple baselines.
