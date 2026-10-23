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

## Lineup window: `parlay watch`

Confirmed lineups appear about an hour before kickoff and the Pinnacle price
reacts within minutes; slower bookmakers may keep the old price for a while.
`parlay watch` re-prices every kickoff slot `--lead` minutes (default 50)
before kickoff: it syncs The Odds API only for the competitions playing in
that slot, looks for value in those matches, logs paper bets and notifies
you. Each slot is processed once, and syncing stops below `--credit-reserve`
remaining credits.

```bash
parlay watch --next                  # upcoming slots and their credit cost (no API calls)
parlay watch                         # run until stopped (Ctrl+C), or --until 2030-01-06T00:00
parlay watch --once                  # process due slots and exit — for cron every 5 minutes
```

A typical weekend of EPL + Championship is 6–10 slots, 2–4 credits each.

**Telegram bot (free).** Create a bot with @BotFather and put its token in
`~/.config/parlay/env` (the CLI loads this file automatically; keep it
`chmod 600` and out of git). Send `/start` to the bot, then:

```bash
parlay telegram --setup      # detects your chat id, saves it, sends a welcome message
parlay telegram              # bot + lineup-window watcher in one process
```

The bot answers only your chat id and is driven by buttons (Indonesian, for
beginners). Each value bet is a card: what to bet in plain words ("Menang
Arsenal", "Over 2,5 gol — menang jika total gol ≥ 3"), the bookmaker odds and
fair odds, win probability, EV in money per Rp1.000, a stake suggestion in
money (respecting `PARLAY_MIN_STAKE`), and buttons **✅ Saya pasang** (asks
the stake; `50rb`, `1jt`, `50rb@2,15` for a different price), **🧪 Simulasi**
and **➕ Tambah ke parlay**. The **🎟 Slip parlay** shows the ticket's odds,
win probability and EV and refuses negative tickets. **📊 Laporan** explains
profit and CLV in plain words. Every morning (06:00 local, `PARLAY_TIMEZONE`,
default Asia/Jakarta) the bot fetches results and fresh prices, settles bets
and reports the outcome; watcher alerts arrive as cards. Typed commands:
`/menu`, `/value`, `/slip`, `/laporan`, `/jadwal`, `/modal 1jt`, `/bantuan`.

**Cron** (keeps running after reboots; keep secrets out of the repository):

```bash
# ~/.config/parlay/env  (chmod 600)
export THE_ODDS_API_KEY=...
export PARLAY_BOOKS="1xBet,Pinnacle"
export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...

# crontab -e
*/5 * * * * cd ~/parlay && . ~/.config/parlay/env && .venv/bin/parlay watch --once >> ~/parlay-watch.log 2>&1
30 7 * * *  cd ~/parlay && . ~/.config/parlay/env && .venv/bin/parlay daily >> ~/parlay-daily.log 2>&1
```

## Mix parlays (accumulators)

For legs on different matches, `ticket odds = Π leg odds` and
`EV = Π (1 + EV_leg) − 1`: the bookmaker margin compounds with every leg.

* Live 1xBet 1X2 prices (Oct 2026) average **−4.0%** EV per selection against
  the Pinnacle fair price; a ticket of 10 favourites priced at 495 had fair
  odds 690, i.e. **EV −28%**. Typical legs: 2 → −8%, 4 → −15%, 10 → −33%.
* Only legs that are value bets on their own compound into a positive ticket.
  Historically (2019–26, one ticket per day and bookmaker, ≤ 3 legs): 33
  tickets, expected EV +12.5%, ticket CLV +11.9% ± 3.2%, but a 9% hit rate,
  a 19-ticket losing streak and a Kelly bankroll multiple of 1.03 — no better
  than the same legs as singles, with far more variance.
* Legs on the same match are correlated and are rejected; exchanges do not
  offer accumulators. Odds boosts (`--boost 0.10`) are the main way a ticket of
  near-fair legs becomes positive.

```bash
parlay ticket vb:0bb3a596 vb:2ce5ad1d                       # legs from `parlay value`
parlay ticket --leg "Arsenal vs Everton:1x2:1" --leg "Hull vs Brentford:totals_2.5:over@1.95"
parlay ticket --favourites --max-legs 10                     # "everything at once": see the EV damage
parlay ticket ... --boost 0.10 --record real --stake 5       # log a ticket you placed
parlay value --parlay --max-legs 3 --paper                   # suggested tickets from value bets only
parlay backtest-value --from 2019-07-01 --parlay             # historical test of the parlay strategy
```

Tickets are settled leg by leg (push, half win and half loss included), get
a CLV of `ticket odds × Π p_close − 1`, and appear in `bets list` and
`bets report`. The Kelly stake shrinks with the number of legs because the
uncertainty of `Π p` grows.

## Reading the report

* **CLV** = taken price × fair closing probability − 1. Its standard error
  shrinks much faster than that of profit, so it is the first thing to watch.
* `bets report` prints a verdict: *STOP* once CLV is significantly negative
  after 50 bets, *evidence of edge* once it is significantly positive after
  200 bets.
* Paper-trade first. Move to real stakes only after CLV stays positive over a
  few hundred bets, and keep stakes small: even a genuine 2–3% edge produces
  long losing runs.

## Does the goal model add anything? (`parlay anchor-test`)

`parlay anchor-test outputs/backtest/predictions.csv` fits a market-anchored
adjustment, `score_k = log p_pinnacle_k + Σ w_j x_jk`, walk-forward by season,
with features the market might price slowly: the goal model's log-ratio to
Pinnacle, Bet365's log-ratio to Pinnacle, rest-day difference, promoted-team
difference and a market-recalibration term. On synthetic data the test
recovers a genuinely informative feature and rejects noise
(`tests/test_anchored.py`). On EPL + Championship 2022–2026 (3,260 matches,
out of sample):

| | log loss | bets | CLV |
|---|---|---|---|
| Pinnacle price as is | 1.0023 | 209 | **+2.1% ± 0.7%** |
| Market-anchored adjustment (all features) | 1.0029 | 458 | −1.1% ± 0.4% |
| Bet365 deviation only | 1.0024 | 199 | +2.0% ± 0.8% |
| Rest days + promoted teams only | 1.0028 | 340 | −0.1% ± 0.5% |

The weight on the goal model is negative in every season (−0.08 to −0.16):
its disagreements with Pinnacle are noise. **Bets are therefore priced against
the plain Pinnacle price**, and any new feature must pass `anchor-test`
(significant log-loss gain *and* higher CLV) before it may influence bets.

## Next steps

1. New information for `anchor-test`: football-data added expected goals
   (HxG/AxG) in 2026-27, and confirmed lineups arrive ~60 minutes before
   kickoff. Both need a season or more of history before they can be tested.
