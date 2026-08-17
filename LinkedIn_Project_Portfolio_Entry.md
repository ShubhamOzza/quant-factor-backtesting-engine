# LinkedIn Project Portfolio Entry

## Title

Point-in-Time Cross-Sectional Factor Backtesting Engine: Momentum and Value Deciles with Institutional-Grade Timing, Cost, and Validation Discipline

## Description

I built a complete cross-sectional equity factor research pipeline in Python, from raw prices to decision-ready diagnostics, without using any backtesting library. Every line of the portfolio accounting is explicit and hand-checkable, which is exactly the standard a quant research desk applies before trusting a backtest.

Highlights:

- Point-in-time universe construction: a top-500-by-market-cap universe is re-derived at every monthly formation date, eliminating survivorship and look-ahead bias in membership.
- Classic factor suite: 12-1 momentum (skipping the most recent month to avoid short-term reversal) and a 6-month lagged earnings yield value factor, each winsorized and z-scored cross-sectionally, then combined into a composite signal.
- Realistic trade simulation: signals are frozen at the prior close, trades execute at the next open, and a 10 bps one-way transaction cost is charged on traded notional. Gross and net equity curves are reported side by side for decile portfolios, a dollar-neutral long-short portfolio, and a long-only variant measured against an equal-weighted benchmark.
- Full diagnostic battery: rank IC versus forward 21-day returns, decile monotonicity, factor decay at 1/3/6/12 month horizons, signal rank autocorrelation, turnover per rebalance, and a full metric suite (CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, hit rate) plus a rough order-of-magnitude capacity estimate, gross and net for every book.
- Robustness and honesty checks: formation-window perturbations, cost sensitivity from 0 to 20 bps, validation/test sample separation, a coarse regime breakdown plus a true calendar-year performance table, deciles-vs-quintiles and equal-vs-vol-inverse weighting comparisons, a monthly-vs-quarterly-vs-semiannual rebalance-frequency sensitivity showing the turnover/cost tradeoff, a negative control that shuffles the signal cross-sectionally and confirms the placebo's rank IC is statistically indistinguishable from zero, and a written trial-count disclosure to keep multiple-testing risk visible.
- Runtime correctness audit: hard assertions verify that portfolio weights are feasible, no position earns a return before its signal exists, returns are finite, costs strictly reduce net performance, and an independently recomputed equal-weight benchmark matches the engine's accounting. A single execution day is exported so the P&L identity can be verified by hand from raw prices.

The engine ships with a clearly labeled synthetic data mode so it runs offline end to end; synthetic output is used only to demonstrate that the machinery is correct, never as market evidence. The same code accepts real long-format OHLCV CSV input with optional point-in-time fundamentals.

## Skills (comma-separated)

Quantitative Research, Factor Investing, Cross-Sectional Asset Pricing, Backtesting Engineering, Point-in-Time Data Management, Look-Ahead Bias Control, Portfolio Construction, Transaction Cost Modeling, Rank IC Analysis, Turnover Analysis, Robustness Testing, Overfitting Diagnostics, Python, NumPy, Pandas, Matplotlib, Equity Markets, Momentum Factor, Value Factor, Reproducible Research, Financial Data Validation

## Media recommendations

1. Equity curves chart (plot_equity_curves.png): log-scale cumulative growth of the long-short, long-only, and benchmark books, gross versus net, as the hero image.
2. Rank IC chart (plot_rank_ic.png): monthly rank IC bars with the 12-month rolling mean; it communicates signal quality at a glance.
3. Decile monotonicity bar chart (plot_decile_monotonicity.png): the classic visual proof that a factor orders the cross-section.
4. Factor decay chart (plot_factor_decay.png): rank IC across 1/3/6/12 month horizons.
5. A short screen recording or slide of the console output where the sanity-check suite prints PASS for each correctness audit; this is a strong differentiator that signals engineering rigor.
6. A one-page architecture diagram: data to universe to signals to deciles to backtest loop to diagnostics and checks.

Note for posting: keep the synthetic-data disclaimer visible in any image or caption. Presenting demo output without that context would misrepresent the work.
