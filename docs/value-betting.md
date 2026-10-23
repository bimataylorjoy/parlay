# Value betting against sharp prices

## Why price-based, not model-based

On EPL + Championship 2020–2026 the goal model carries **no information beyond
the market**: in a logarithmic opinion pool fitted on 2020–24 and scored on
2024–26, the optimal weight on the model next to Bet365 prices is −0.09 (and
−0.06 next to the Pinnacle closing line). Betting the model's disagreements
with the market produced closing-line value (CLV) of −7% over 3,987 bets.

The literature points the same way. Odds are the most accurate public
forecasts (Štrumbelj, 2014); documented edges come from **price
discrepancies** between bookmakers (Kaunitz, Zhong & Kriener, 2017), from
information the market has not priced yet (lineups and player ratings,
Arntzen & Hvattum, 2021), from slow reactions to regime changes (Winkelmann,
Deutscher & Ötting, 2021), or from models trained to be *decorrelated* from
the market (Hubáček, Šourek & Železný, 2019). Profitable statistical models
report small edges, e.g. about +0.8% per bet on Over/Under (Wheatcroft, 2020).

So `parlay value` prices every bet against a sharp reference — the Pinnacle
price, de-vigged with Shin's method — and recommends bookmakers that offer
more than that fair price.

## Historical evidence (`parlay backtest-value`)

football-data.co.uk prices, regular bookmakers only (not the "best price"
column), one bet per match and market, EV ≥ 2%, July 2019 – June 2026:

| | bets | flat yield | CLV vs closing |
|---|---|---|---|
| All tracked bookmakers, Pinnacle reference | 331 | +3.3% ± 11.4% | **+3.3% ± 0.6%** |
| Bet365 only | 127 | +14.7% ± 19.6% | **+2.7% ± 0.9%** |
| Consensus reference instead of Pinnacle | 162 | −12.9% ± 12.5% | +0.7% ± 0.8% |
| EV ≥ 5% | 77 | +29.5% ± 29.0% | **+6.3% ± 1.3%** |

CLV is positive in every season from 2019-20 to 2025-26 (+1.6% to +6.3%).
Profit is not yet statistically distinguishable from zero — at these
volumes it takes thousands of bets — which is exactly why the ledger tracks
CLV.

Reference choice matters. In 2024–26, when both are available:

| reference | regular bookmakers: CLV |
|---|---|
| Pinnacle | +2.6% ± 0.8% |
| Betfair Exchange (football-data snapshot) | −1.4% ± 0.9% |
| Consensus of bookmakers | −0.3% ± 1.1% |

football-data stopped publishing Pinnacle prices in 2026-27, so the live
reference comes from The Odds API (region `eu`, free plan).

Caveats: football-data captures pre-match prices once (before the round), so
real opportunities are more numerous but shorter-lived; tracked bookmakers
are a small subset of what you may have access to; bookmakers limit accounts
that consistently beat the closing line.

## Free data sources

| source | cost | what it gives |
|---|---|---|
| football-data.co.uk season files (`parlay fetch`) | free | results, pre-match and **closing** prices of ~6–12 bookmakers, Betfair Exchange, best/average price |
| football-data.co.uk `fixtures.csv` (`parlay odds sync`) | free, no key | upcoming matches with the same bookmakers; updated around Tuesday and Friday; **no Pinnacle** |
| The Odds API (`parlay odds sync --source the-odds-api`) | free key, 500 credits/month | ~25 bookmakers in region `eu` including **Pinnacle** (from its public site, possibly delayed) and Betfair Exchange; cost = markets × regions per call |

With `--api-markets h2h,totals` and region `eu`, one sync of EPL +
Championship costs 4 credits: about four syncs a day on the free plan.

## Daily workflow

```bash
export THE_ODDS_API_KEY=...                       # free key from the-odds-api.com
export PARLAY_BOOKS="Bet365,William Hill,1xBet"   # bookmakers you can actually use (names as printed by `odds sync`)

parlay fetch --league E0 E1 --from-season 2026 --to-season 2026   # results + closing prices (after each round)
parlay odds sync                                   # soft-book prices from fixtures.csv
parlay odds sync --source the-odds-api             # Pinnacle reference + more bookmakers
parlay value --paper                               # value bets; also logged as paper bets

parlay bets add vb:1a2b3c4d --stake 10 --odds 2.12 # a bet you actually placed (price you got)
parlay bets update                                 # attach closing prices and settle results
parlay bets report                                 # yield, CLV ± standard error, verdict
```

`parlay value` sizes stakes with fractional Kelly shrunk for the uncertainty
of the fair price (Baker & McHale, 2013), capped by `--max-stake`. Exchange
prices are reduced by `--commission`. Edges above `--max-ev` (30%) are
discarded as stale or erroneous prices.

## Reading the report

* **CLV** = taken price × fair closing probability − 1. Its standard error
  shrinks much faster than that of profit, so it is the first thing to watch.
* `bets report` prints a verdict: *STOP* once CLV is significantly negative
  after 50 bets, *evidence of edge* once it is significantly positive after
  200 bets.
* Paper-trade first. Move to real stakes only after CLV stays positive over a
  few hundred bets, and keep stakes small: even a genuine 2–3% edge produces
  long losing runs.

## Next steps

1. Use more odds snapshots per day close to kickoff, so the closing proxy
   and prices are fresher.
2. Market-anchored model: `logit p = logit p_market + f(features)` with
   features the market prices slowly (confirmed lineups, injuries, xG),
   trained with a decorrelation penalty and accepted only if its
   opinion-pool weight is positive out of sample.
