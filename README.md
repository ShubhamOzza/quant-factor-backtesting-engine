# Project I-5: Cross-Sectional Factor Backtesting Engine

A point-in-time cross-sectional equity factor research and backtesting
pipeline, implemented from scratch in a single, heavily commented Python
script. No backtesting library is used: every step of the portfolio
accounting is explicit and auditable, and a built-in sanity-check suite
verifies weights, timing, costs, and P&L identities on every run.

## What it does

1. Loads daily adjusted OHLCV data, either from a real long-format CSV or
   from a clearly labeled synthetic demonstration mode that runs offline.
2. Builds a point-in-time top-N-by-market-cap universe at every monthly
   formation date.
3. Computes 12-1 momentum and a 6-month lagged value factor (earnings
   yield, or book-to-market when supplied), winsorizes and z-scores each
   cross-sectionally, and combines them into an equal-weighted composite.
4. Forms equal-weighted decile portfolios, a dollar-neutral long-short
   portfolio (top decile minus bottom decile), and a long-only top-decile
   portfolio measured against a naive equal-weighted universe benchmark.
5. Runs an explicit daily backtest: signals are fixed at the prior close,
   trades execute at the next open, and a 10 bps one-way cost is charged
   on traded notional. Gross and net curves are both produced.
6. Produces diagnostics: rank IC versus forward 21-day returns, decile
   monotonicity, factor decay at 1/3/6/12 months, rank autocorrelation of
   the signal, and turnover per rebalance.
7. Runs robustness checks: formation-window perturbations, cost
   sensitivity, validation/test sample separation, a coarse regime
   breakdown plus a true calendar-year performance table, a negative
   control (signal shuffled cross-sectionally across five independent
   draws, which must show a statistically insignificant rank IC), and a
   trial-count note for multiple-testing awareness.
8. Enforces correctness at runtime with hard assertions: feasible weights,
   no position before its signal, finite returns, costs weakly reducing
   performance, a hand-checkable P&L snapshot, and an independent naive
   equal-weight benchmark cross-check.

## How to run

```bash
pip install -r requirements.txt

# Offline synthetic demo (default, clearly labeled as artificial data)
python Point_in_Time_Factor_Backtesting_Engine.py --mode synthetic

# Quick smoke test (small synthetic panel)
python Point_in_Time_Factor_Backtesting_Engine.py --mode synthetic --fast \
    --outdir results_smoke

# Real data
python Point_in_Time_Factor_Backtesting_Engine.py --mode csv \
    --data prices.csv --outdir results_real
```

Useful flags: `--universe-size` (default 500), `--start` / `--end`
(defaults 2010-01-01 to 2025-12-31), `--cost-bps` (default 10),
`--fund-lag-months` (default 6), `--n-deciles` (default 10; use 5 for
quintiles), `--weighting` (`equal` default, or `vol_inverse`), `--seed`,
`--fast`.

## Input data schema (CSV mode)

Long format, one row per ticker per trading day:

| column           | required | description                                             |
|------------------|----------|---------------------------------------------------------|
| date             | yes      | ISO date, YYYY-MM-DD                                    |
| ticker           | yes      | security identifier                                     |
| open, high, low  | yes      | adjusted daily prices                                   |
| close            | yes      | adjusted close (split- and dividend-adjusted)           |
| volume           | yes      | daily share volume                                      |
| mktcap           | no       | market cap; if absent, a price*volume proxy is used     |
| earnings_yield   | no       | point-in-time E/P; preferred value factor               |
| book_to_market   | no       | point-in-time B/M; used if earnings yield is absent     |

The engine applies the 6-month fundamental lag itself. If no fundamental
column is present, it falls back to a momentum-only composite and says so.

A yfinance-style workflow is supported by downloading adjusted OHLCV per
ticker, adding a `ticker` column, concatenating into long format, and
passing the file via `--data`. Market cap and fundamentals are not
available from yfinance reliably on a point-in-time basis; supply them
from a point-in-time vendor when running real research.

## File map

| file                                       | purpose                                  |
|--------------------------------------------|------------------------------------------|
| Point_in_Time_Factor_Backtesting_Engine.py | the complete engine                      |
| Word_Document_Content.md                   | academic report content                  |
| LinkedIn_Project_Portfolio_Entry.md        | portfolio entry                          |
| Plain_English_Project_Notes.md             | plain-language explanation               |
| README.md                                  | this file                                |
| requirements.txt                           | dependencies                             |
| results/                                   | output of the full synthetic demo run    |
| results_smoke/                             | output of the fast smoke-test run        |
| full_run.log / smoke_run.log               | console logs of the two demo runs        |

## Output files (written to the chosen --outdir)

| file                                | content                                     |
|-------------------------------------|---------------------------------------------|
| performance_summary.csv             | CAGR, vol, Sharpe, Sortino, max drawdown, Calmar, hit rate per book, gross and net |
| daily_returns_by_book.csv           | full daily return matrix                    |
| turnover_by_rebalance.csv           | per-execution turnover and cost detail      |
| turnover_summary.csv                | mean one-way and two-way turnover per book  |
| capacity_estimate.csv               | rough order-of-magnitude capacity estimate  |
| ic_summary.csv / rank_ic_monthly.csv| rank IC statistics and monthly series       |
| decile_monotonicity.csv             | mean forward return by decile               |
| factor_decay.csv                    | rank IC at 1/3/6/12 month horizons          |
| rank_autocorrelation.csv            | month-to-month signal persistence           |
| robustness_perturbations.csv        | formation-window variants                   |
| robustness_cost_sensitivity.csv     | performance at 0/5/10/20 bps one-way        |
| robustness_split_sample.csv         | validation/test halves                      |
| robustness_regimes.csv              | coarse multi-year regime breakdown          |
| robustness_calendar_year.csv        | true year-by-year (Jan-Dec) performance table |
| robustness_bucket_and_weighting.csv | deciles vs quintiles, equal vs vol-inverse weighting |
| robustness_rebalance_frequency.csv  | monthly vs quarterly vs semiannual turnover/cost tradeoff |
| negative_control_check.csv          | placebo (shuffled-signal) IC and Sharpe     |
| hand_check_snapshot.csv             | one execution day, hand-verifiable          |
| sanity_checks.csv                   | results of every runtime correctness check  |
| trial_count_note.txt                | multiple-testing disclosure                 |
| signals_composite.csv               | composite scores at each formation date     |
| plot_*.png                          | equity curves, drawdown, IC, deciles, decay |

## Important notes

- Synthetic mode generates artificial data with a deliberately modest
  embedded factor structure. Its outputs demonstrate that the machinery is
  correct; they are not evidence about real markets, and every plot and
  log line says so.
- The demonstration long-short Sharpe (gross ~1.7, net ~1.6) is materially
  higher than the roughly 0.4-0.8 gross Sharpe reported in the published
  12-1 momentum literature. Word_Document_Content.md Section 6.5 explains
  why honestly: the runtime correctness audit and the negative control both
  pass, so the gap is attributable to the synthetic generator's clean,
  persistent embedded signal rather than a bug. Do not read the synthetic
  headline numbers as market evidence.
- The risk-free rate is assumed to be zero in the Sharpe calculation.
- Stocks with missing daily returns are treated as flat for that day; in
  the synthetic panel this case never occurs, and in real data it should
  be rare after proper adjustment.
- The runtime sanity checks fail loudly (AssertionError) rather than
  silently producing impossible output.
