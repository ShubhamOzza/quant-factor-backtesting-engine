# A Point-in-Time Cross-Sectional Factor Backtesting Engine: Momentum and Value Decile Portfolios under Realistic Timing and Cost Assumptions

## Abstract

This project designs, implements, and validates a cross-sectional equity
factor backtesting engine built entirely from first principles, without
recourse to any backtesting library. The engine evaluates a composite of
two canonical factors, 12-1 price momentum and earnings yield lagged six
months, on a point-in-time top-500-by-market-capitalization universe with
monthly rebalancing. Portfolio construction follows the academic decile
sort tradition: equal-weighted decile portfolios, a dollar-neutral
long-short portfolio (top decile minus bottom decile), and a long-only
top-decile variant benchmarked against a naive equal-weighted universe
portfolio. Trade simulation respects a strict information boundary:
signals are computed from data available at the prior close, execution
occurs at the next open, and a one-way transaction cost of ten basis
points is charged on traded notional. The engine reports gross and net
equity curves, a full performance metric suite, rank information
coefficients against forward 21-day returns, decile monotonicity, factor
decay at 1, 3, 6, and 12 month horizons, signal rank autocorrelation, and
turnover. Robustness is examined through formation-window perturbations,
cost sensitivity, validation and test sample separation, calendar regime
breakdown, and an explicit trial-count disclosure. Correctness is enforced
by a runtime audit covering weight feasibility, position timing, return
finiteness, the cost ordering of net and gross performance, a
hand-checkable profit and loss reconstruction, and an independent
equal-weight benchmark cross-check. The engine ships with a clearly
labeled synthetic data mode so the complete pipeline is reproducible
offline; synthetic results are used exclusively to demonstrate mechanical
correctness and are never presented as market evidence.

## 1. Introduction

Cross-sectional factor investing rests on a deceptively simple premise:
sorting stocks on a characteristic today should order their future
returns. The academic literature, from Jegadeesh and Titman's momentum
portfolios to the value strategies of Fama and French, established that
such sorts can carry economically meaningful information. The practical
difficulty is that a backtest of these ideas is only credible if its
information handling is impeccable. Three failure modes recur: look-ahead
bias, in which the simulation trades on information that was not yet
available; survivorship bias, in which the test universe is defined using
outcomes known only in the future; and cost neglect, in which gross paper
returns are presented as if trading were free.

This project treats those failure modes as engineering constraints rather
than afterthoughts. The engine is written as a single, heavily commented
Python script in which every accounting step is explicit. No backtesting
library is used, because the objective is not merely to obtain an equity
curve but to obtain one whose every dollar can be traced to raw prices by
hand.

## 2. Data

### 2.1 Real data workflow

The engine accepts daily adjusted OHLCV data in long format with the
schema date, ticker, open, high, low, close, volume, and optional columns
for market capitalization, earnings yield, and book-to-market. Prices must
be split- and dividend-adjusted so that computed returns are total
returns. The target configuration is fifteen years of daily data, 2010 to
2025, for a universe large enough to support a top-500 ranking. Market
capitalization drives point-in-time universe membership. Fundamentals are
lagged six months internally to reflect publication delay; a
point-in-time vendor snapshot is the preferred source for real research.

### 2.2 Synthetic demonstration mode

For offline reproduction, the engine generates an artificial daily panel
with a realistic correlation structure: a common market factor, ten sector
factors, and idiosyncratic noise drive returns, while each stock carries a
persistent latent earnings yield trait and a slowly varying latent
momentum drift. Next-period returns load only modestly on these traits,
which is what keeps the resulting information coefficients in the
plausible range observed in the empirical literature rather than at
implausibly high values. Market capitalizations evolve with prices plus
noise so that universe membership rotates over time. All console output,
plots, and documentation generated from this mode carry an explicit
synthetic-data label. Statistics quoted in Section 6 come from this mode
and describe the demonstration data set only.

## 3. Methodology

### 3.1 Point-in-time universe

On the last trading day of each month, the formation date, stocks are
ranked by market capitalization as of that day's close and the top 500 are
admitted. Membership decided at a formation date governs the entire
subsequent holding period and is frozen between formation dates. Because
membership uses only information available at the formation close, and
because membership rotates as capitalizations evolve, the universe is free
of both look-ahead and survivorship flavor.

### 3.2 Factor definitions

Momentum is the total return from 252 trading days ago to 21 trading days
ago, the standard 12-1 construction that omits the most recent month to
avoid contamination by short-term reversal. The value factor is earnings
yield, lagged six months (126 trading days) to model reporting delay;
book-to-market lagged six months is used instead when it is the available
fundamental. Each factor is winsorized cross-sectionally at the 2nd and
98th percentiles of each formation date and then z-scored
cross-sectionally. The composite signal is the equal-weighted average of
available factor z-scores.

### 3.3 Portfolio construction

At each formation date the universe is sorted on the composite signal and
split into ten equal-count deciles. Every decile portfolio is
equal-weighted, so each decile weight vector sums to exactly one. The
long-short portfolio holds decile 10 long and decile 1 short with equal
notional, making it dollar-neutral with weights summing to zero. The
long-only variant holds decile 10 alone and is compared with a naive
equal-weighted portfolio of the full universe.

### 3.4 Performance metric suite

For every tracked book (each decile, the long-short composite, the
long-only top decile, and the naive equal-weighted benchmark) the engine
computes the same metric suite on both the gross and the net return
series: annualized return (CAGR), annualized volatility, the Sharpe ratio,
the Sortino ratio (which penalizes only downside variance, since upside
variance is not a risk an investor wants penalized), maximum drawdown, the
Calmar ratio (annualized return divided by the magnitude of the worst
drawdown), and the hit rate. Turnover, both one-way and two-way, is
reported per rebalance and averaged per book. Finally, a rough
order-of-magnitude capacity estimate is computed from the median daily
dollar volume of universe members, the number of names held per side, and
the average turnover per rebalance, under an assumed maximum participation
rate of 5% of a name's daily dollar volume; this is a liquidity sanity
check, not a market-impact model, and is documented as such.

### 3.5 Trade simulation and timing audit

The timing convention is the core of the design. Signals are computed from
information available at the close of the formation date T. Trades execute
at the open of the next trading day E. On day E, the pre-existing holdings
earn the overnight return from the close of T to the open of E, the new
weights are established at the open, a one-way cost of ten basis points is
charged on the absolute weight change summed across names, and the new
holdings earn the open-to-close return of day E. On all subsequent days
the portfolio earns close-to-close returns and weights drift with realized
returns. It follows mechanically that no position can earn a return from a
period preceding its signal, and that execution prices are always next-day
opens rather than formation-day closes. Both gross and net curves are
maintained; net differs from gross only by the subtraction of realized
costs.

### 3.6 Diagnostics

The rank information coefficient is the Spearman correlation between the
signal and the forward 21-trading-day return at each formation date;
mean IC, IC standard deviation, IC information ratio, and the fraction of
positive periods are reported. Decile monotonicity is assessed from the
mean forward return of each decile and a Spearman correlation between
decile number and mean return. Factor decay is measured by the rank IC at
forward horizons of 21, 63, 126, and 252 trading days, corresponding to
roughly 1, 3, 6, and 12 months. Signal persistence is measured by the
rank autocorrelation of scores between adjacent formation dates, and
trading intensity by one-way and two-way turnover per rebalance.

### 3.7 Robustness protocol

Six complementary checks probe overfitting, fragility, and the
possibility of a hidden data leak. First, formation-window perturbations
rerun the long-short backtest under 9-1, 12-1, and 12-3 momentum
specifications; a genuine effect should keep its sign and rough magnitude
under these neighbors. Second, cost sensitivity reruns the backtest at 0,
5, 10, and 20 basis points one-way; net performance must decline
monotonically, and the cost level at which the edge disappears is itself
informative. Third, validation and test sample separation evaluates the
fixed, ex ante configuration on the first and second halves of the sample;
the second half serves as a pseudo out-of-sample test. Fourth, a coarse
calendar regime breakdown reports performance within up to four
contiguous multi-year sub-periods so that concentration of returns in a
single regime is visible, and a companion true calendar-year table reports
net performance for every individual year in the sample, which is the
finer-grained sub-period view the specification calls for and the one
that would expose a single anomalous year (a momentum-crash year or a
value-drawdown year) that a coarser regime table could mask. Fifth, a
negative control repeats the composite signal five times with its
cross-sectional values randomly reassigned across tickers at each
formation date, destroying any true stock-to-signal correspondence while
leaving universe membership, timing, and cost mechanics unchanged; a
leak-free engine must show the placebo signal's pooled rank IC
statistically indistinguishable from zero. Finally, a written trial-count
note discloses how many variants were examined, in recognition that
silent selection among many trials inflates false-discovery risk.

### 3.8 Runtime correctness audit

Every execution of the engine ends with hard assertions: each decile
weight vector sums to one and the long-short book to zero; the first
nonzero return of every book occurs no earlier than the first execution
day; all daily returns are finite; cumulative net performance weakly
underperforms cumulative gross performance for every book; an
independently reconstructed, monthly rebalanced equal-weight universe
series matches the engine's benchmark book to high precision; and one
execution day's profit and loss is exported with weights and raw prices so
the identity weighted stock returns minus cost equals portfolio return can
be verified by hand.

## 4. Step-by-step application

1. Install dependencies with pip install -r requirements.txt.
2. Run the offline demonstration: `python
   Point_in_Time_Factor_Backtesting_Engine.py --mode synthetic`. For a
   quick smoke test, add `--fast`.
3. To run on real data, prepare a long-format adjusted OHLCV CSV with the
   schema in Section 2.1, optionally including mktcap and earnings_yield
   or book_to_market columns, then run with `--mode csv --data prices.csv`.
4. Inspect the console: it prints data dimensions, the metric suite,
   turnover, diagnostics, robustness tables, and the audit results.
5. Open the output directory: performance_summary.csv and the plot files
   give the headline view; robustness_*.csv and trial_count_note.txt give
   the honesty layer; hand_check_snapshot.csv supports manual P&L
   verification.

## 5. Implementation notes

The engine is a single Python file organized into clearly delimited
sections: configuration, synthetic data generation, real data loading,
panel construction, universe construction, factor computation, signal
processing, portfolio construction, the explicit backtest loop, metrics,
diagnostics, robustness, correctness checks, plotting, and orchestration.
The backtest loop walks every trading day in order and holds portfolio
state in a small PortfolioBook class whose drift, execution, and cost
arithmetic is written out explicitly. All cross-sectional operations are
vectorized with pandas; all accounting identities are asserted at runtime.

## 6. Output interpretation (synthetic demonstration data)

The figures below describe the artificial demonstration data set generated
by synthetic mode. They show that the machinery behaves as designed; they
are not evidence about real markets.

### 6.1 Headline performance

Over the full synthetic sample, the dollar-neutral long-short composite
earned an annualized net return of approximately 12.5% with an
annualized volatility of approximately 7.7%, for a net Sharpe
ratio of approximately 1.57, a net Sortino ratio of approximately
2.77, a net Calmar ratio of approximately 1.55, and a maximum
drawdown of approximately -8.1%. The long-only top decile earned
approximately 21.5% annualized net, versus approximately 11.9%
for the naive equal-weighted benchmark. Gross and net curves differ by
realized costs, and the audit confirms net is always the weaker of the
two. A rough, explicitly non-rigorous capacity estimate (5% of median
daily dollar volume, spread across the roughly 100 names held by the
long-short book) puts the order of magnitude at approximately 2.4
billion dollars of NAV before a single name's rebalance trade would
exceed that participation threshold; this is a liquidity sanity check,
not a market-impact model, and should not be read as a precise capacity
figure.

### 6.2 Signal quality

The composite signal achieved a mean monthly rank IC of approximately
0.032 with an IC information ratio of 0.67 and 70.5% of
months positive, values squarely inside the range the empirical
literature considers plausible for monthly equity factors. Mean forward
returns rise broadly across deciles, and the decile monotonicity
statistic (Spearman correlation between decile number and mean forward
return) is 0.88. The signal's month-to-month rank autocorrelation
averages 0.95, indicating a persistent, low-churn ranking, and the
long-short book turns over roughly 48.7% of notional per
rebalance on a one-way basis.

### 6.3 Decay and robustness

Rank IC in the demonstration data is 0.032 at one month, rising to
0.115 at twelve months, reflecting the persistent value component
embedded in the generator; in real data, practitioners should expect
momentum-dominated signals to decay rather than rise with horizon.
Formation-window perturbations leave the long-short result with the same
sign and similar magnitude (net Sharpe 1.74 for 9-1, 1.57
for 12-1, 1.63 for 12-3), indicating the finding is not an
artifact of one parameter choice. Cost sensitivity shows the expected
monotone decline in net performance as one-way costs rise from 0 to 20
basis points (net Sharpe 1.72 at 0 bps down to 1.42 at 20 bps).
Bucket count and weighting also degrade gracefully rather than inverting:
quintiles (net Sharpe 2.10) and vol-inverse weighting (net Sharpe
1.77) both keep the same sign and order of magnitude as the decile,
equal-weighted baseline (net Sharpe 1.57), which is the qualitative
behavior the specification calls for from a genuine, non-overfit signal.
Rebalance frequency shows the expected turnover reduction as rebalancing
slows (two-way turnover per rebalance rises from 0.97 monthly to 1.62
quarterly to 2.16 semiannually, while the number of rebalances per year
falls correspondingly, so annualized turnover still declines); in this
demonstration data net Sharpe does not degrade at lower frequencies
(1.57 monthly, 1.59 quarterly, 1.84 semiannual) because the
synthetic signal is unusually persistent, which is itself a reminder that
this behavior is a property of the synthetic generator and should not be
extrapolated to real markets, where staler signals typically do lose
edge. Split-sample results are reported in robustness_split_sample.csv,
regime results in robustness_regimes.csv, and the full calendar-year
breakdown in robustness_calendar_year.csv; all are descriptive and
accompany the written trial-count disclosure in trial_count_note.txt.

### 6.4 Correctness audit

All runtime checks passed on the demonstration run: weights feasible, no
position before its signal, returns finite, costs reducing performance,
the independent equal-weight benchmark matching the engine's accounting
(daily return correlation 1.0000, cumulative growth gap
0.0011), and the hand-checkable P&L identity for a sampled execution
day reproduced to machine precision. A negative control adds an
independent leak check: shuffling the composite signal cross-sectionally
across five independent draws and rerunning the full backtest yields a
pooled mean rank IC of -0.0020 (t-statistic -1.24, not
statistically distinguishable from zero), in clear contrast to the real
signal's mean IC of 0.032. Because the placebo signal carries no
information yet the engine still executes it under the identical timing
and cost mechanics as the real signal, this is direct evidence that the
real signal's performance comes from the deliberately embedded factor
structure rather than from a timing or accounting leak in the simulation.

### 6.5 Comparison to published factor evidence and an honest caveat about magnitude

The published empirical literature on 12-1 price momentum long-short
portfolios in liquid, large-capitalization equity universes generally
reports a gross Sharpe ratio in the rough range of 0.4 to 0.8, with
net Sharpe lower once realistic trading costs and, on the short leg,
borrow costs are applied. This project's demonstration run reports a
materially higher figure: a gross long-short Sharpe of 1.72 and a
net Sharpe of 1.57. That gap is real and is stated plainly rather
than smoothed over: these synthetic-data results do not corroborate the
published evidence, and they are not intended to.

The gap is attributable to how the synthetic generator is built, not to a
bug in the engine. Every runtime correctness check passes, the
independent equal-weight benchmark reconciles to within 0.11% of
cumulative growth, the hand-checked P&L identity matches to machine
precision, and the negative control shows a shuffled signal's predictive
power collapses to statistical noise. What produces the unusually high
Sharpe instead is that the synthetic momentum-drift trait driving expected
returns is a smoothly, slowly mean-reverting process (daily autoregressive
coefficient 0.98) that is shared consistently between the ranking date
and the realized holding period, diversified across roughly fifty names
on each leg of the long-short book. Real markets do not offer a signal
this clean: factor regimes shift, crowding erodes edges over time, factor
loadings decay unevenly across names, and idiosyncratic shocks are
fat-tailed in ways this generator does not reproduce. The reader's
takeaway should therefore be narrow and specific: this run demonstrates
that the engine computes exactly what it claims to compute, under a
synthetic data-generating process with a strong embedded signal. It does
not demonstrate that a 12-1 momentum plus lagged earnings-yield composite
would earn a Sharpe near 1.6 on real markets, and treating these
headline numbers as market evidence would be a misreading of what the
demonstration mode is for.

## 7. Validation against underfitting and overfitting

Underfitting is guarded against by using economically motivated, standard
factor definitions rather than a specification tuned to the data, and by
confirming that the signal carries genuine cross-sectional information
(positive mean IC, broad decile ordering, low churn). Overfitting is
addressed on five fronts: formation-window perturbations show the result
is not knife-edged in parameter space; the validation and test split keeps
the second half of the sample untouched by any design choice; cost
sensitivity shows whether the edge survives realistic frictions; the
regime breakdown exposes any concentration of returns in one calendar
period; and the trial-count note keeps the multiple-testing burden
explicit. IC magnitudes are deliberately kept in the realistic 0.02 to
0.08 band in the demonstration generator, because synthetic demonstrations
that produce near-perfect ICs teach the wrong intuition about what real
factors look like.

## 8. Limitations

1. The headline statistics quoted here come from synthetic data with an
   embedded factor structure chosen by the author; they demonstrate
   mechanics only.
2. The risk-free rate is taken as zero in Sharpe calculations.
3. The short side of the long-short book ignores borrow costs, short
   availability, and margin requirements, all of which degrade real
   net performance.
4. Equal-weighted deciles ignore capacity; a real implementation should
   study weighting by liquidity and market impact beyond the flat 10 bps
   assumption.
5. Stocks with missing daily returns are treated as flat for the day;
   corporate actions and delistings in real data require dedicated
   handling and a point-in-time vendor.
6. Sector exposures are not neutralized; sector-tilted factor loads can
   masquerade as stock-selection skill.
7. The trial-count disclosure mitigates but does not formally correct for
   multiple testing; a deflated Sharpe ratio or similar adjustment is a
   natural extension.

## 9. Conclusion

The project delivers a transparent, fully auditable cross-sectional factor
backtesting engine in which timing discipline, cost realism, and
correctness verification are first-class design features rather than
post-hoc checks. The demonstration run confirms that the machinery
produces internally consistent, hand-verifiable results: feasible weights,
no pre-signal positions, monotone cost impact, plausible information
coefficients, and stable performance under perturbation. The same code
accepts real adjusted OHLCV input with optional point-in-time
fundamentals, providing a direct path from offline demonstration to
genuine research. The natural next steps are sector-neutral scoring,
borrow and capacity modeling for the short book, and a formal
multiple-testing adjustment.
