# Plain English Project Notes

## What we made

A computer program that answers a simple but important question: does a
stock-picking rule actually work? The rule we test combines two classic
ideas. First, momentum: stocks that went up over the past year (skipping
the last month) tend to keep doing well for a while. Second, value:
stocks that are cheap relative to their earnings tend to do better than
expensive ones. The program takes a large list of stocks, scores each one
on both ideas every month, sorts them into ten buckets from worst to best
score, and then simulates what would have happened if you had bought the
top bucket, shorted the bottom bucket, and repeated this every month for
years, including realistic trading costs.

## The core concept

The hard part of backtesting is not the math; it is honesty. It is very
easy to accidentally cheat by using information you could not have known
at the time. Three classic forms of cheating, and how we prevent each:

1. Look-ahead bias: using today's closing price to trade at today's close.
   We freeze the signal at the prior close and only trade at the next
   day's opening price.
2. Survivorship bias: testing only on companies that exist today. We
   rebuild the list of the 500 largest companies at the start of every
   month, using only what was knowable then.
3. Ignoring costs: every trade costs money. We charge 10 basis points
   (0.10 percent) of the amount traded, every time we trade, and we show
   results both before and after costs.

## How we made it, step by step

1. Data: the program reads daily prices (open, high, low, close, volume)
   for each stock. If you do not have real data handy, it can generate a
   clearly labeled fake market so you can still see the whole machine
   working. Fake results are always marked as fake.
2. Universe: on the last trading day of each month, we rank companies by
   market value and keep the top 500. That list is frozen for the month.
3. Signals: for each stock in the list we compute the momentum score
   (return from 12 months ago to 1 month ago) and the value score
   (earnings yield from 6 months ago, because company reports arrive
   late). We trim extreme outliers, rescale both scores so they are
   comparable, and average them into one number per stock.
4. Portfolios: we sort the stocks into ten buckets by score. Bucket 1 is
   the worst, bucket 10 is the best. The main strategy buys bucket 10 and
   shorts bucket 1 with equal dollars, so it is market neutral. We also
   track a simple buy-and-hold version of bucket 10 and a plain benchmark
   that owns every stock in the list equally.
5. Trading simulation: the next morning, at the opening price, we move
   from the old buckets to the new ones, pay the trading cost, and then
   track the value of each portfolio day by day until the next rebalance.
6. Diagnostics: we ask whether the scores actually predicted returns. The
   main measure is rank IC, which is the correlation between our ranking
   and what stocks actually did over the next month. We also check that
   bucket 1 did worse than bucket 2, bucket 2 worse than bucket 3, and so
   on (monotonicity), how quickly the signal fades over 1, 3, 6, and 12
   months (decay), how stable the ranking is month to month
   (autocorrelation), and how much trading the strategy requires
   (turnover).
7. Robustness: we rerun everything with slightly different momentum
   windows, with higher and lower costs, on the first and second halves of
   the sample separately, within different multi-year calendar periods, and
   again broken out year by year. We also try five buckets instead of ten,
   weight stocks by inverse volatility instead of equally, and rebalance
   quarterly or twice a year instead of monthly, to see how much less
   trading (and therefore less cost) trades off against a staler signal. A
   real effect should survive these nudges; a fluke usually does not.
8. Self-checks: every run ends with an audit. The program proves to
   itself that the portfolio weights add up correctly, that no trade
   happened before its signal existed, that costs always made results
   worse rather than better, and that one full day of profit and loss can
   be reproduced by hand from raw prices. If any check fails, the program
   stops with an error instead of showing pretty but wrong charts.
9. Negative control: as a final honesty test, the program randomly
   reshuffles which score belongs to which stock, breaking the connection
   between signal and outcome on purpose, and reruns the whole backtest.
   A trustworthy engine should show this shuffled version earning
   essentially nothing, because there is no longer any real information in
   it. If the shuffled version still made money, that would be a sign of a
   hidden bug letting the program see the future. In this project, the
   shuffled version's predictive power is statistically indistinguishable
   from zero, which is the expected and reassuring result.

## Interview-ready explanation

If an interviewer asks you to explain this project in two minutes, say
something like this:

"I built a cross-sectional factor backtesting engine from scratch, with
no backtesting library, because I wanted every assumption to be visible
in the code. It tests a composite of 12-1 momentum and lagged earnings
yield on a monthly rebalanced top-500 universe. The two things I focused
on were timing discipline and honest accounting. Signals are fixed at the
prior close and executed at the next open, so there is no look-ahead.
Universe membership is rebuilt point-in-time every month, so there is no
survivorship bias. Costs are 10 basis points one way, and I report gross
and net side by side. For validation I used rank IC, decile
monotonicity, factor decay, rank autocorrelation, and turnover, plus
robustness checks: formation-window perturbations, cost sensitivity,
validation and test sample separation, regime and calendar-year
breakdowns, and a negative control that shuffles the signal and confirms
the shuffled version earns nothing. The part I am most proud of is the
runtime audit: the engine asserts that weights are feasible, that no
position earns a return before its signal exists, that costs always
reduce performance, and that a single execution day of P&L can be
reconstructed by hand from raw prices. The engine also runs on a clearly
labeled synthetic data mode so anyone can reproduce the full pipeline
offline. I am upfront that the synthetic long-short Sharpe comes out
higher than published momentum evidence, because the synthetic signal is
cleaner than a real market; the honest reading is that synthetic results
prove the machinery works, not that the factor would perform this well in
real markets."

If they ask what you would do next: add point-in-time fundamentals from a
real vendor, sector-neutral scoring, capacity and borrow-cost modeling
for the short side, and a proper multiple-testing adjustment such as a
deflated Sharpe ratio.
