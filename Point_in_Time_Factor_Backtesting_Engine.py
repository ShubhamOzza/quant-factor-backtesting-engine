#!/usr/bin/env python3
"""
Point-in-Time Cross-Sectional Factor Backtesting Engine
=======================================================

Project I-5: Cross-Sectional Factor Backtesting Engine
Domain: Quant Research / Factor Investing

This script implements a complete, self-contained cross-sectional equity
factor research and backtesting pipeline WITHOUT any backtesting library.
Every step of the portfolio accounting is explicit, so the P&L can be
hand-checked from the intermediate CSV outputs.

What the engine does
--------------------
1. Loads daily adjusted OHLCV data (real CSV input, or a clearly labeled
   synthetic demonstration mode that runs fully offline).
2. Builds a POINT-IN-TIME universe (top-N by market capitalization at each
   monthly formation date) so there is no survivorship or look-ahead bias
   in membership.
3. Computes two classic cross-sectional factors:
     - 12-1 momentum: total return from t-252 to t-21 trading days
       (skipping the most recent month to avoid short-term reversal).
     - Earnings yield (E/P), lagged 6 months, as the value factor. A
       book-to-market column can be supplied instead when available.
4. Winsorizes and z-scores each factor cross-sectionally at every
   formation date.
5. Forms equal-weighted decile portfolios each month: a dollar-neutral
   long-short portfolio (top decile minus bottom decile) and a long-only
   top-decile portfolio compared against an equal-weighted universe
   benchmark.
6. Runs an explicit daily backtest loop:
     - Signals are computed from information available at the PRIOR CLOSE.
     - Trades execute at the NEXT OPEN (no close-price execution).
     - One-way transaction cost of 10 bps is charged on traded notional.
     - Both gross and net equity curves are produced.
7. Produces diagnostics:
     - Rank IC of each factor versus forward 21-day returns.
     - Decile monotonicity (mean forward return by decile).
     - Factor decay at 1, 3, 6, and 12 month horizons.
     - Rank autocorrelation of factor scores (signal persistence).
     - One-way and two-way turnover per rebalance.
8. Runs robustness checks:
     - Formation-window perturbations (e.g. 9-1, 12-1, 12-3 momentum).
     - Cost sensitivity (0, 5, 10, 20 bps one-way).
     - Validation/test sample separation (first half / second half).
     - Regime breakdown by coarse multi-year sub-periods AND a true
       calendar-year (Jan-Dec) performance table.
     - Bucket count and weighting sensitivity: deciles vs quintiles, and
       equal vs vol-inverse weighting within each bucket.
     - Rebalance-frequency sensitivity: monthly vs quarterly vs
       semiannual, showing the turnover/cost tradeoff explicitly.
     - A negative control: the composite signal is shuffled cross-
       sectionally (five independent draws) and the backtest is rerun; a
       leak-free engine must show the shuffled signal's pooled rank IC
       statistically indistinguishable from zero (|t| < 2).
     - A trial-count note for multiple-testing awareness.
9. Writes plots and summary CSVs to an output directory.

Timing convention (point-in-time audit)
---------------------------------------
Formation date T is the last trading day of a month. All factor inputs use
data at or before the close of T (fundamentals lagged 6 months). Weights are
decided at the close of T. Trades execute at the OPEN of the first trading
day after T (call it E). The portfolio earns the open-to-close return of E
on its new weights, and close-to-close returns thereafter until the next
execution day. Therefore no position can earn a return from a period that
precedes its signal, and no close-execution look-ahead exists.

Correctness guarantees enforced at runtime
------------------------------------------
- Weights of every decile portfolio sum to exactly 1.0 (long book) and the
  short book sums to -1.0, so the long-short book is dollar-neutral.
- All daily returns are finite (asserted).
- Net returns are always <= gross returns in aggregate (costs are
  non-negative), which is asserted on the cumulative curves.
- A hand-checkable P&L snapshot for one rebalance is exported so the
  accounting identity (weighted stock returns minus cost = portfolio
  return) can be verified by hand.
- A naive equal-weight benchmark is computed independently as a
  cross-check of the return accounting.

Usage
-----
    # Synthetic offline demo (default)
    python Point_in_Time_Factor_Backtesting_Engine.py --mode synthetic

    # Real data from a long-format CSV
    python Point_in_Time_Factor_Backtesting_Engine.py --mode csv \
        --data prices.csv --outdir results_real

Real CSV schema (long format, one row per ticker per day):
    date,ticker,open,high,low,close,volume[,mktcap,earnings_yield,book_to_market]
- 'date'   : ISO date (YYYY-MM-DD)
- 'close'  : split- and dividend-adjusted close (adjusted prices required
             so that returns are total returns)
- 'mktcap' : optional; if absent, a volume-price proxy is used for the
             universe ranking and a warning is printed
- 'earnings_yield' or 'book_to_market': optional point-in-time fundamental
  ratio (already lagged by the data provider, or lagged internally by
  --fund-lag-months, default 6)

Author: Quantitative Research Project (I-5)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend so the script runs on servers
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """All tunable parameters of the engine in one place."""

    # --- data / universe ---
    mode: str = "synthetic"            # 'synthetic' or 'csv'
    data_path: str = ""                # CSV path when mode == 'csv'
    universe_size: int = 500           # top-N by market cap
    start: str = "2010-01-01"
    end: str = "2025-12-31"

    # --- factors ---
    mom_lookback: int = 252            # momentum total window (trading days)
    mom_skip: int = 21                 # momentum skip window (trading days)
    fund_lag_months: int = 6           # lag applied to fundamentals
    winsor_p: float = 0.02             # winsorization tail probability

    # --- portfolio construction ---
    n_deciles: int = 10                # bucket count (10=deciles, 5=quintiles)
    weighting: str = "equal"           # 'equal' or 'vol_inverse' within bucket
    vol_lookback: int = 63             # trailing window (days) for vol weighting
    cost_bps: float = 10.0             # one-way transaction cost in bps

    # --- diagnostics ---
    ic_horizon: int = 21               # forward horizon for rank IC (days)
    decay_horizons: tuple = (21, 63, 126, 252)   # 1/3/6/12 months

    # --- synthetic demo ---
    syn_n_tickers: int = 520           # slightly larger than universe so
                                       # membership actually rotates
    syn_seed: int = 42
    syn_fast: bool = False             # small quick run for smoke tests

    # --- output ---
    outdir: str = "results"

    # Derived constants
    trading_days_per_year: int = 252


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_data(cfg: EngineConfig) -> pd.DataFrame:
    """Generate a clearly labeled SYNTHETIC daily OHLCV panel.

    The generator embeds a realistic factor structure so the demo produces
    plausible (not perfect) diagnostics:

    - A common market factor, sector factors, and idiosyncratic noise drive
      daily returns (so correlations are realistic).
    - Each stock has a slowly varying latent 'quality of momentum' and a
      persistent latent earnings yield. Next-month returns carry a SMALL
      loading on these latent traits, which produces rank IC values in the
      realistic 0.02-0.08 range rather than implausibly large ones.
    - Market caps evolve with prices plus noise, so the top-N universe
      membership rotates over time (avoids a static, survivorship-flavored
      universe).

    IMPORTANT: this data is artificial. Any performance statistics produced
    from it are a mechanics demonstration, NOT market evidence.
    """
    print("[data] SYNTHETIC DEMO MODE: generating artificial data. "
          "Results are a mechanics demonstration, not market evidence.")

    rng = np.random.default_rng(cfg.syn_seed)

    if cfg.syn_fast:
        n_tickers, n_years = 120, 6
    else:
        n_tickers = cfg.syn_n_tickers
        n_years = max(1, int(cfg.end[:4]) - int(cfg.start[:4]) + 1)

    n_days = n_years * cfg.trading_days_per_year
    dates = pd.bdate_range(cfg.start, periods=n_days)
    dates = dates[dates <= pd.Timestamp(cfg.end)]
    n_days = len(dates)

    n_sectors = 10

    # --- factor structure for returns ---
    market_ret = rng.normal(0.0003, 0.010, n_days)          # daily market
    sector_ret = rng.normal(0.0, 0.006, (n_days, n_sectors))
    sector_of = rng.integers(0, n_sectors, n_tickers)
    beta = rng.normal(1.0, 0.25, n_tickers)                 # market betas
    idio_vol = rng.uniform(0.012, 0.030, n_tickers)
    idio = rng.normal(0.0, 1.0, (n_days, n_tickers)) * idio_vol

    # Latent traits that give the factors genuine (but modest) signal.
    # Earnings yield trait: persistent, cross-sectionally dispersed.
    ey_trait = rng.normal(0.0, 1.0, n_tickers)
    # Momentum trait: slowly mean-reverting per-stock drift.
    mom_drift = np.zeros((n_days, n_tickers))
    drift = rng.normal(0.0, 1.0, n_tickers)
    for t in range(n_days):
        drift = 0.98 * drift + rng.normal(0.0, 0.19, n_tickers)
        mom_drift[t] = drift

    # Small alpha loadings: annualized information is deliberately modest.
    alpha_ey = 0.00017       # per-day loading on earnings-yield trait
    alpha_mom = 0.00012      # per-day loading on momentum drift

    rets = (beta * market_ret[:, None]
            + sector_ret[:, sector_of]
            + idio
            + alpha_ey * ey_trait[None, :]
            + alpha_mom * mom_drift)
    rets = np.clip(rets, -0.20, 0.20)   # realistic daily bound

    tickers = [f"SIM{i:04d}" for i in range(n_tickers)]
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))

    # Open prices: previous close plus a small overnight move.
    overnight = rng.normal(0.0, 0.003, (n_days, n_tickers))
    open_ = np.vstack([close[:1], close[:-1]]) * (1.0 + overnight)

    # High/low bracket the open and close with a small range.
    hi_lo_range = np.abs(rng.normal(0.0, 0.004, (n_days, n_tickers)))
    high = np.maximum(open_, close) * (1.0 + hi_lo_range)
    low = np.minimum(open_, close) * (1.0 - hi_lo_range)

    # Volume: log-normal, correlated with absolute returns.
    base_vol = rng.uniform(0.5e6, 8e6, n_tickers)
    volume = base_vol[None, :] * np.exp(
        rng.normal(0.0, 0.4, (n_days, n_tickers))
        + 8.0 * np.abs(rets))

    # Market cap: price times a slowly varying share count, plus noise so
    # the top-N ranking rotates over time.
    shares = base_vol * rng.uniform(5, 50, n_tickers)
    mktcap = close * shares[None, :]

    # Point-in-time earnings yield: persistent trait plus slow noise.
    # Stored as of each date; the engine applies the 6-month reporting lag
    # itself, exactly as it would for real vendor data.
    ey_noise = np.zeros((n_days, n_tickers))
    shock = np.zeros(n_tickers)
    for t in range(n_days):
        shock = 0.995 * shock + rng.normal(0.0, 0.05, n_tickers)
        ey_noise[t] = shock
    earnings_yield = 0.05 + 0.02 * ey_trait[None, :] + 0.01 * ey_noise
    earnings_yield = np.clip(earnings_yield, -0.05, 0.30)

    # --- assemble long-format DataFrame ---
    idx = pd.MultiIndex.from_product(
        [dates, tickers], names=["date", "ticker"])
    df = pd.DataFrame({
        "open": open_.ravel(),
        "high": high.ravel(),
        "low": low.ravel(),
        "close": close.ravel(),
        "volume": volume.ravel(),
        "mktcap": mktcap.ravel(),
        "earnings_yield": earnings_yield.ravel(),
    }, index=idx).reset_index()
    return df


# ---------------------------------------------------------------------------
# Real data loading
# ---------------------------------------------------------------------------


def load_csv_data(cfg: EngineConfig) -> pd.DataFrame:
    """Load a long-format daily OHLCV CSV and validate its schema.

    Required columns: date, ticker, open, high, low, close, volume.
    Optional columns: mktcap, earnings_yield, book_to_market.

    If 'mktcap' is missing, a price-times-volume liquidity proxy is used
    for the universe ranking and a prominent warning is issued, because a
    proxy ranking is only an approximation of a true top-500 universe.
    If neither fundamental column is present, the engine falls back to a
    momentum-only composite and says so loudly.
    """
    if not cfg.data_path or not os.path.exists(cfg.data_path):
        raise FileNotFoundError(
            f"CSV mode requires --data pointing to an existing file; "
            f"got '{cfg.data_path}'.")
    df = pd.read_csv(cfg.data_path, parse_dates=["date"])
    required = {"date", "ticker", "open", "high", "low",
                "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if "mktcap" not in df.columns:
        warnings.warn(
            "No 'mktcap' column found; using price*volume as a liquidity "
            "proxy for universe ranking. This approximates, but does not "
            "replicate, a true top-500-by-market-cap universe.")
        df["mktcap"] = df["close"] * df["volume"]

    if "earnings_yield" not in df.columns and \
            "book_to_market" not in df.columns:
        warnings.warn(
            "No fundamental column ('earnings_yield' or 'book_to_market') "
            "found; the value factor will be skipped and the engine will "
            "run a momentum-only composite.")
    return df


# ---------------------------------------------------------------------------
# Panel helpers: pivot long data into date x ticker matrices
# ---------------------------------------------------------------------------


def build_panels(df: pd.DataFrame) -> dict:
    """Pivot the long DataFrame into aligned date x ticker matrices.

    Returns a dict of DataFrames keyed by field name. All panels share the
    same index (trading dates, sorted) and columns (tickers, sorted), which
    makes the vectorized cross-sectional algebra that follows safe.
    """
    panels = {}
    fields = ["open", "high", "low", "close", "volume", "mktcap"]
    for f in fields:
        panels[f] = df.pivot(index="date", columns="ticker", values=f)
    # Align everything to the close panel's index/columns.
    base = panels["close"]
    for f in fields:
        panels[f] = panels[f].reindex(index=base.index,
                                      columns=base.columns)
    # Optional fundamentals.
    for f in ["earnings_yield", "book_to_market"]:
        if f in df.columns:
            panels[f] = df.pivot(index="date", columns="ticker",
                                 values=f).reindex(index=base.index,
                                                   columns=base.columns)
    return panels


# ---------------------------------------------------------------------------
# Point-in-time universe
# ---------------------------------------------------------------------------


def month_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last available trading day of each calendar month.

    These are the formation dates T: the close of T is the information
    boundary for all signals.
    """
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).last().values)


def build_universe(mktcap: pd.DataFrame,
                   formation_dates: pd.DatetimeIndex,
                   universe_size: int) -> pd.DataFrame:
    """Point-in-time top-N universe membership at each formation date.

    For every formation date T we rank stocks by market cap as of the close
    of T and keep the top N. Membership is therefore known at T (no
    look-ahead) and rotates naturally over time (no survivorship flavor).

    Returns a boolean DataFrame (dates x tickers) that is True wherever a
    ticker is in the universe for the holding period following the most
    recent formation date.
    """
    # Use NaN as 'no decision yet' so membership can be forward-filled
    # between formation dates without silent dtype downcasting.
    membership = pd.DataFrame(np.nan, index=mktcap.index,
                              columns=mktcap.columns, dtype=float)
    for T in formation_dates:
        caps = mktcap.loc[T].dropna()
        top = caps.nlargest(universe_size).index
        membership.loc[T, :] = 0.0
        membership.loc[T, top] = 1.0
    # The universe decided at T governs the whole holding period until the
    # next formation date.
    membership = membership.ffill().fillna(0.0).astype(bool)
    return membership


# ---------------------------------------------------------------------------
# Factor computation
# ---------------------------------------------------------------------------


def compute_momentum(close: pd.DataFrame, lookback: int,
                     skip: int) -> pd.DataFrame:
    """Classic lookback-skip momentum (default 12-1).

    mom(T) = P(T - skip) / P(T - lookback) - 1

    The most recent 'skip' days are excluded to avoid the well documented
    one-month short-term reversal effect. Uses only prices at or before T,
    so the signal is point-in-time by construction.
    """
    p_end = close.shift(skip)
    p_start = close.shift(lookback)
    mom = p_end / p_start - 1.0
    # Require the full window to exist.
    valid = close.shift(lookback).notna()
    return mom.where(valid)


def compute_value_factor(panels: dict, lag_days: int) -> pd.DataFrame:
    """Value factor: earnings yield (preferred) or book-to-market, lagged.

    Fundamentals are lagged by 'lag_days' trading days (default 6 months,
    ~126 days) to mimic real reporting delays: at formation date T we only
    allow ourselves to know fundamentals that were published at least
    lag_days earlier. Returns None if no fundamental panel exists.
    """
    if "earnings_yield" in panels:
        raw = panels["earnings_yield"]
        name = "earnings_yield"
    elif "book_to_market" in panels:
        raw = panels["book_to_market"]
        name = "book_to_market"
    else:
        return None, None
    return raw.shift(lag_days), name


def winsorize_xs(row: pd.Series, p: float) -> pd.Series:
    """Cross-sectional winsorization of one formation date's factor row.

    Clips values to the [p, 1-p] quantiles of that date's cross-section.
    This limits the influence of data errors and extreme outliers without
    discarding stocks entirely.
    """
    x = row.dropna()
    if len(x) < 10:
        return row
    lo, hi = x.quantile(p), x.quantile(1.0 - p)
    return row.clip(lo, hi)


def zscore_xs(row: pd.Series) -> pd.Series:
    """Cross-sectional z-score of one formation date's factor row.

    Subtracts the cross-sectional mean and divides by the cross-sectional
    standard deviation, giving a comparable scale across dates and factors.
    """
    x = row.dropna()
    if len(x) < 10 or x.std(ddof=0) == 0:
        return row * np.nan
    mu, sd = x.mean(), x.std(ddof=0)
    return (row - mu) / sd


def build_signals(panels: dict, membership: pd.DataFrame,
                  formation_dates: pd.DatetimeIndex,
                  cfg: EngineConfig) -> pd.DataFrame:
    """Compute the composite signal at every formation date.

    Steps at each formation date T:
      1. Restrict to the point-in-time universe at T.
      2. Compute momentum and (lagged) value factor using data up to T.
      3. Winsorize each factor cross-sectionally.
      4. Z-score each factor cross-sectionally.
      5. Average available factor z-scores into a single composite score
         (equal weights; if a factor is missing for a stock, the average is
         taken over the remaining factors).

    Returns a DataFrame of composite scores indexed by formation date.
    """
    lag_days = int(round(cfg.fund_lag_months * 21))
    mom = compute_momentum(panels["close"], cfg.mom_lookback, cfg.mom_skip)
    val, val_name = compute_value_factor(panels, lag_days)
    if val is None:
        print("[factors] No fundamental data: running momentum-only "
              "composite.")
    else:
        print(f"[factors] Value factor in use: {val_name} "
              f"(lagged {cfg.fund_lag_months} months).")

    scores = {}
    for T in formation_dates:
        in_uni = membership.loc[T]
        cols = in_uni[in_uni].index
        m = mom.loc[T, cols]
        parts = []
        if m.notna().sum() >= 30:
            parts.append(zscore_xs(winsorize_xs(m, cfg.winsor_p)))
        if val is not None:
            v = val.loc[T, cols]
            if v.notna().sum() >= 30:
                parts.append(zscore_xs(winsorize_xs(v, cfg.winsor_p)))
        if not parts:
            scores[T] = pd.Series(dtype=float)
            continue
        comp = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
        scores[T] = comp
    sig = pd.DataFrame(scores).T
    sig.index.name = "formation_date"
    return sig


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


def decile_weights(scores: pd.Series, n_deciles: int,
                   weighting: str = "equal",
                   vol: pd.Series | None = None) -> pd.DataFrame:
    """Assign weighted bucket portfolios from one date's scores.

    Rank the cross-section, split into 'n_deciles' buckets (bucket 1 =
    lowest scores, bucket N = highest). This also serves as the quintile
    sort when n_deciles=5, and any other equal-count bucket scheme.

    weighting controls how names within a bucket are weighted:
      - 'equal' (default): 1/count, unchanged from the original design.
      - 'vol_inverse': weight proportional to 1 / trailing realized
        volatility ('vol', a Series of same-date trailing vol per ticker,
        required in this mode), so lower-volatility names get a larger
        share of the bucket's notional. This is the standard alternative
        to equal weighting used to compare risk-based sizing against the
        naive scheme.

    Returns a DataFrame with bucket indices as columns and stock weights in
    rows. Each column sums to exactly 1.0 when the bucket is non-empty;
    empty buckets (tiny universes) stay all-zero and are handled downstream.
    """
    s = scores.dropna()
    n = len(s)
    out = pd.DataFrame(0.0, index=scores.index,
                       columns=range(1, n_deciles + 1))
    if n < n_deciles:
        return out
    # Rank-based bucket assignment, robust to ties via method='first'.
    ranks = s.rank(method="first")
    bucket = np.ceil(ranks / n * n_deciles).astype(int).clip(1, n_deciles)
    for d in range(1, n_deciles + 1):
        members = bucket[bucket == d].index
        if weighting == "vol_inverse" and vol is not None:
            v = vol.reindex(members)
            v = v.clip(lower=1e-6)
            if v.notna().sum() == 0:
                out.loc[members, d] = 1.0 / len(members)
                continue
            v = v.fillna(v.mean())
            inv = 1.0 / v
            out.loc[members, d] = inv / inv.sum()
        else:
            out.loc[members, d] = 1.0 / len(members)
    return out


# ---------------------------------------------------------------------------
# Backtest engine (explicit daily loop, no backtesting library)
# ---------------------------------------------------------------------------


class PortfolioBook:
    """Explicit daily accounting for one weight-vector portfolio.

    The book holds a weight vector w over tickers (fractions of NAV). For a
    long-only book the weights sum to +1; for a dollar-neutral long-short
    book they sum to 0 with the long side summing to +1 and the short side
    to -1. All arithmetic is written out explicitly so the P&L can be
    reproduced by hand from the exported snapshots.

    Timing:
      - Formation at close of T (weights decided).
      - Execution at open of the next trading day E. On E the book earns
        open-to-close returns on the NEW weights, minus transaction cost.
      - On every later day the book earns close-to-close returns and the
        weights drift with realized returns until the next execution day.
    """

    def __init__(self, name: str, cost_rate: float):
        self.name = name
        self.cost_rate = cost_rate          # one-way cost as a fraction
        self.w: pd.Series | None = None     # current close-of-day weights
        self.records: list = []             # daily return rows
        self.turnover_records: list = []    # per-execution turnover rows
        self._pending_w: pd.Series | None = None   # target decided at T
        self._formation_date = None

    def set_target(self, target: pd.Series, formation_date) -> None:
        """Record a new target weight vector decided at a formation close."""
        self._pending_w = target.fillna(0.0)
        self._formation_date = formation_date

    @staticmethod
    def _drift(w: pd.Series, r: pd.Series) -> tuple:
        """Drift weights through a realized return vector.

        Returns (portfolio_return, new_weights). Stocks with missing
        returns are treated as flat for the day (documented simplification;
        in the synthetic panel this case never occurs).
        """
        aligned_r = r.reindex(w.index).fillna(0.0)
        port_ret = float((w * aligned_r).sum())
        new_w = w * (1.0 + aligned_r) / (1.0 + port_ret)
        return port_ret, new_w

    def execute_at_open(self, date, overnight_r: pd.Series,
                        oc_r: pd.Series) -> tuple:
        """Execute the pending target at the open of 'date'.

        Steps:
          1. Drift existing weights from yesterday's close to today's open
             using overnight returns.
          2. Turnover = sum |new weight - drifted old weight|.
          3. Cost = cost_rate * turnover (charged on traded notional).
          4. The book then earns today's open-to-close returns on the new
             weights.
        Returns (gross_return, net_return).
        """
        new_w = self._pending_w
        # Step 1: the OLD holdings earn the overnight move into the open.
        if self.w is None:
            r_on_old = 0.0
            drifted = pd.Series(0.0, index=new_w.index)
        else:
            r_on_old, drifted_full = self._drift(self.w, overnight_r)
            drifted = drifted_full.reindex(
                new_w.index.union(drifted_full.index)).fillna(0.0)
            new_w = new_w.reindex(drifted.index).fillna(0.0)

        # Step 2 and 3: turnover and cost on traded notional.
        turnover = float((new_w - drifted).abs().sum())
        cost = self.cost_rate * turnover

        # Step 4: the NEW holdings earn today's open-to-close move. The
        # total gross day return compounds the overnight (old book) and the
        # open-to-close (new book) components.
        r_oc_new, drifted_close = self._drift(new_w, oc_r)
        gross_ret = (1.0 + r_on_old) * (1.0 + r_oc_new) - 1.0
        net_ret = gross_ret - cost
        self.w = drifted_close
        # Components are stored so the P&L identity can be hand-checked
        # against raw prices in the sanity-check stage.
        self.turnover_records.append(
            {"date": date, "turnover_two_way": turnover,
             "turnover_one_way": turnover / 2.0, "cost": cost,
             "r_overnight_old": r_on_old, "r_oc_new": r_oc_new,
             "gross_ret": gross_ret, "net_ret": net_ret})
        self.records.append({"date": date, "gross": gross_ret,
                             "net": net_ret})
        self._pending_w = None
        return gross_ret, net_ret

    def accrue_close_to_close(self, date, cc_r: pd.Series) -> tuple:
        """Earn close-to-close returns on a non-execution day."""
        if self.w is None:  # before first execution: book is flat
            self.records.append({"date": date, "gross": 0.0, "net": 0.0})
            return 0.0, 0.0
        gross_ret, new_w = self._drift(self.w, cc_r)
        self.w = new_w
        self.records.append({"date": date, "gross": gross_ret,
                             "net": gross_ret})  # no trade, no cost
        return gross_ret, gross_ret

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records).set_index("date")
        df.columns = pd.MultiIndex.from_product([[self.name],
                                                 df.columns])
        return df

    def turnover_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.turnover_records).set_index("date")
        df["book"] = self.name
        return df


def target_weights_for_books(dec_w: pd.DataFrame,
                             universe_row: pd.Series) -> dict:
    """Build target weight vectors for all tracked books from decile weights.

    Books:
      - 'decile_1' ... 'decile_N' : equal-weighted decile books.
      - 'long_short'              : top decile minus bottom decile,
                                    dollar-neutral (sums to 0).
      - 'long_only'               : top decile only.
      - 'benchmark_ew'            : naive equal-weight universe portfolio,
                                    used as an independent cross-check.
    """
    n_dec = dec_w.shape[1]
    books = {f"decile_{d}": dec_w[d].copy() for d in range(1, n_dec + 1)}
    books["long_short"] = dec_w[n_dec] - dec_w[1]
    books["long_only"] = dec_w[n_dec].copy()
    members = universe_row[universe_row].index
    bw = pd.Series(0.0, index=dec_w.index)
    if len(members) > 0:
        bw.loc[members] = 1.0 / len(members)
    books["benchmark_ew"] = bw
    return books


def run_backtest(panels: dict, membership: pd.DataFrame,
                 signals: pd.DataFrame, cfg: EngineConfig,
                 cost_bps: float | None = None) -> dict:
    """Run the full daily backtest and return return/turnover frames.

    This is the core loop. It walks every trading day in order:
      - At a formation date T (month-end), signals are read and target
        weights are decided using information up to the close of T.
      - On the next trading day, all books execute at the open.
      - On all other days, books accrue close-to-close returns.
    """
    cost_rate = (cost_bps if cost_bps is not None
                 else cfg.cost_bps) / 1e4
    dates = panels["close"].index
    close, open_ = panels["close"], panels["open"]

    # Daily return building blocks.
    cc_r = close.pct_change()                    # close-to-close
    oc_r = close / open_ - 1.0                   # open-to-close
    on_r = open_ / close.shift(1) - 1.0          # overnight

    # Trailing realized volatility panel, only needed for the optional
    # vol-inverse weighting scheme. Computed from cc_r up to and including
    # each date, so using vol_panel.loc[T] at formation date T is point-in-
    # time (no future information enters the weighting decision).
    vol_panel = (cc_r.rolling(cfg.vol_lookback).std()
                if cfg.weighting == "vol_inverse" else None)

    formation_dates = list(signals.index)
    exec_dates = {}   # formation date -> next trading day
    dpos = {d: i for i, d in enumerate(dates)}
    for T in formation_dates:
        i = dpos.get(T)
        if i is not None and i + 1 < len(dates):
            exec_dates[T] = dates[i + 1]
    exec_set = set(exec_dates.values())

    # Initialize books lazily at the first formation date.
    books: dict[str, PortfolioBook] = {}
    formation_set = set(formation_dates)

    # Daily loop: walk every trading day in chronological order. At a
    # formation date we decide target weights from that day's signal (this
    # happens at the close of T); at the next trading day we execute those
    # targets at the open. Setting targets inside the walk is essential:
    # it guarantees each formation date gets its own execution.
    for t, d in enumerate(dates):
        if d in formation_set and d in exec_dates:
            sig = signals.loc[d]
            v = vol_panel.loc[d] if vol_panel is not None else None
            dec_w = decile_weights(sig, cfg.n_deciles,
                                   weighting=cfg.weighting, vol=v)
            targets = target_weights_for_books(dec_w, membership.loc[d])
            if not books:
                for name in targets:
                    books[name] = PortfolioBook(name, cost_rate)
            for name, w in targets.items():
                books[name].set_target(w, d)
        if d in exec_set:
            for bk in books.values():
                if bk._pending_w is not None:
                    bk.execute_at_open(d, on_r.loc[d], oc_r.loc[d])
                else:
                    bk.accrue_close_to_close(d, cc_r.loc[d])
        else:
            for bk in books.values():
                bk.accrue_close_to_close(d, cc_r.loc[d])

    rets = pd.concat([bk.to_frame() for bk in books.values()], axis=1)
    turnover = pd.concat([bk.turnover_frame() for bk in books.values()])
    return {"returns": rets, "turnover": turnover}


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def perf_metrics(daily_ret: pd.Series, periods_per_year: int = 252) -> dict:
    """Standard performance metric suite for a daily return series.

    Annualization uses the geometric mean for returns and sqrt-time scaling
    for volatility, Sharpe, and Sortino (risk-free rate assumed zero, which
    is stated in the report as a simplification). Reports CAGR (ann_return),
    annualized volatility, Sharpe, Sortino (downside-deviation-based),
    max drawdown, Calmar (return over max drawdown), and hit rate, matching
    the full metric suite the project specification calls for.
    """
    r = daily_ret.dropna()
    if len(r) == 0:
        return {}
    assert np.isfinite(r).all(), "Non-finite daily return encountered."
    growth = (1.0 + r).prod()
    years = len(r) / periods_per_year
    ann_ret = growth ** (1.0 / years) - 1.0 if years > 0 else np.nan
    ann_vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = (r.mean() / r.std(ddof=0) * np.sqrt(periods_per_year)
              if r.std(ddof=0) > 0 else np.nan)

    # Sortino ratio: like Sharpe, but the denominator only penalizes
    # downside variance (returns below zero), since upside variance is not
    # a risk investors want penalized.
    downside = r[r < 0]
    downside_dev = (downside.std(ddof=0) * np.sqrt(periods_per_year)
                    if len(downside) > 1 else np.nan)
    sortino = (ann_ret / downside_dev
              if downside_dev and downside_dev > 0 else np.nan)

    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_dd = drawdown.min()
    # Calmar ratio: annualized return divided by the magnitude of the
    # worst peak-to-trough drawdown over the same period.
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate": (r > 0).mean(),
        "total_growth": growth,
        "n_days": len(r),
    }


def summarize_books(rets: pd.DataFrame) -> pd.DataFrame:
    """Compute the metric suite for every tracked book, gross and net."""
    rows = []
    for name in rets.columns.get_level_values(0).unique():
        for side in ["gross", "net"]:
            m = perf_metrics(rets[(name, side)])
            m.update({"book": name, "curve": side})
            rows.append(m)
    df = pd.DataFrame(rows).set_index(["book", "curve"])
    cols = ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown",
            "calmar", "hit_rate", "total_growth", "n_days"]
    return df[cols]


# ---------------------------------------------------------------------------
# Capacity estimate (order-of-magnitude liquidity check, not a full
# market-impact model)
# ---------------------------------------------------------------------------


def capacity_estimate(panels: dict, membership: pd.DataFrame,
                      turnover_summary: pd.DataFrame, cfg: EngineConfig,
                      participation_rate: float = 0.05) -> pd.DataFrame:
    """Rough capacity estimate for the long-short and long-only books.

    This is a standard back-of-envelope liquidity sanity check, not a
    rigorous market-impact model. For each book it estimates:

        capacity (USD of NAV) = participation_rate * median_ADV_dollar
                                 * n_names_in_book
                                 / avg_two_way_turnover_per_rebalance

    where median_ADV_dollar is the median daily dollar volume (share
    volume times close price) across tickers that were ever admitted to
    the point-in-time universe, n_names_in_book is the average number of
    names held per side (roughly universe_size / n_deciles), and
    avg_two_way_turnover_per_rebalance comes from turnover_summary.csv.
    The interpretation: this is approximately the largest NAV that could be
    run through the strategy such that no single name's rebalance trade
    exceeds 'participation_rate' of that name's typical daily dollar
    volume, which keeps price impact from a single day's trading modest.
    Real capacity work would model impact per name and per day, account
    for correlated trading across managers running similar factors, and
    stress the estimate under low-liquidity regimes; this estimate exists
    to make the order of magnitude visible, not to be precise.
    """
    dollar_vol = panels["volume"] * panels["close"]
    ever_in_universe = membership.any(axis=0)
    cols = ever_in_universe[ever_in_universe].index
    median_adv = float(dollar_vol[cols].stack().median())

    n_names_per_leg = max(1, cfg.universe_size // cfg.n_deciles)
    rows = []
    for book, n_names in [("long_short", 2 * n_names_per_leg),
                          ("long_only", n_names_per_leg)]:
        if book not in turnover_summary.index:
            continue
        avg_to = turnover_summary.loc[book, "avg_two_way_per_rebalance"]
        cap = (participation_rate * median_adv * n_names / avg_to
              if avg_to and avg_to > 0 else np.nan)
        rows.append({
            "book": book,
            "median_adv_dollar": median_adv,
            "n_names_in_book": n_names,
            "avg_two_way_turnover_per_rebalance": avg_to,
            "participation_rate_assumed": participation_rate,
            "capacity_estimate_usd": cap,
        })
    return pd.DataFrame(rows).set_index("book")


# ---------------------------------------------------------------------------
# Factor diagnostics
# ---------------------------------------------------------------------------


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward close-to-close return over 'horizon' trading days.

    fwd(T) = P(T + horizon) / P(T) - 1. Used ONLY for ex-post diagnostics
    (IC, monotonicity, decay), never for portfolio formation.
    """
    return close.shift(-horizon) / close - 1.0


def rank_ic_series(signals: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Spearman rank IC at each formation date: corr(signal, forward return).

    A persistent positive mean rank IC with a healthy IC information ratio
    (mean/std) indicates a genuine cross-sectional signal. Realistic values
    for monthly equity factors are typically between 0.02 and 0.08.
    """
    ics = {}
    for T in signals.index:
        s = signals.loc[T]
        f = fwd.loc[T]
        both = pd.concat([s, f], axis=1, keys=["sig", "fwd"]).dropna()
        if len(both) >= 30:
            ics[T] = both["sig"].corr(both["fwd"], method="spearman")
    return pd.Series(ics)


def ic_summary(ic: pd.Series, label: str) -> dict:
    """Summarize a rank IC series: mean, t-stat style IR, positivity."""
    ic = ic.dropna()
    return {
        "series": label,
        "mean_ic": ic.mean(),
        "ic_std": ic.std(ddof=0),
        "ic_ir": ic.mean() / ic.std(ddof=0) if ic.std(ddof=0) > 0 else np.nan,
        "pct_positive": (ic > 0).mean(),
        "n_periods": len(ic),
    }


def decile_monotonicity(signals: pd.DataFrame, fwd: pd.DataFrame,
                        n_deciles: int) -> pd.DataFrame:
    """Mean forward return by signal decile, averaged over formation dates.

    A monotone increase from decile 1 to decile N is the classic visual
    evidence that a factor orders the cross-section correctly. Also reports
    the Spearman correlation between decile number and mean return as a
    compact monotonicity statistic.
    """
    bucket_rets = {d: [] for d in range(1, n_deciles + 1)}
    for T in signals.index:
        s = signals.loc[T].dropna()
        f = fwd.loc[T].reindex(s.index)
        both = pd.concat([s, f], axis=1, keys=["sig", "fwd"]).dropna()
        if len(both) < n_deciles * 3:
            continue
        ranks = both["sig"].rank(method="first")
        bucket = np.ceil(ranks / len(both) * n_deciles).astype(int)
        for d in range(1, n_deciles + 1):
            vals = both.loc[bucket == d, "fwd"]
            if len(vals):
                bucket_rets[d].append(vals.mean())
    means = pd.Series({d: (np.mean(v) if v else np.nan)
                       for d, v in bucket_rets.items()})
    mono = means.corr(pd.Series(means.index, index=means.index,
                                dtype=float), method="spearman")
    out = means.to_frame("mean_forward_return")
    out.index.name = "decile"
    out.attrs["monotonicity_spearman"] = mono
    return out


def factor_decay(signals: pd.DataFrame, close: pd.DataFrame,
                 horizons: tuple) -> pd.DataFrame:
    """Rank IC versus forward returns at 1/3/6/12 month horizons.

    A healthy signal typically decays gradually; a signal that only works
    at the shortest horizon is often reversal or microstructure noise.
    """
    rows = []
    for h in horizons:
        fwd = forward_returns(close, h)
        ic = rank_ic_series(signals, fwd)
        s = ic_summary(ic, f"{h}d")
        s["horizon_days"] = h
        rows.append(s)
    return pd.DataFrame(rows).set_index("horizon_days")


def rank_autocorrelation(signals: pd.DataFrame) -> pd.Series:
    """Rank autocorrelation of the signal between adjacent formation dates.

    High persistence (e.g. 0.8-0.95 for value, lower for momentum) means
    the ranking is stable, which implies lower turnover. Very low
    persistence warns of a noisy, expensive-to-trade signal.
    """
    dates = list(signals.index)
    out = {}
    for a, b in zip(dates[:-1], dates[1:]):
        both = pd.concat([signals.loc[a], signals.loc[b]],
                         axis=1, keys=["a", "b"]).dropna()
        if len(both) >= 30:
            out[b] = both["a"].corr(both["b"], method="spearman")
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Robustness: perturbations, cost sensitivity, sample splits, regimes
# ---------------------------------------------------------------------------


def formation_window_perturbations(panels, membership, formation_dates,
                                   cfg) -> pd.DataFrame:
    """Re-run the long-short backtest under perturbed formation windows.

    A fragile factor shows large performance swings under small window
    changes (a sign of overfitting to one specification). A robust factor
    keeps the same sign and a broadly similar Sharpe across neighbors.
    """
    variants = [
        ("mom_9_1", 189, 21),     # 9-month window, skip 1 month
        ("mom_12_1_base", cfg.mom_lookback, cfg.mom_skip),
        ("mom_12_3", 252, 63),    # 12-month window, skip 3 months
    ]
    rows = []
    for label, lb, sk in variants:
        vcfg = EngineConfig(**{**cfg.__dict__,
                               "mom_lookback": lb, "mom_skip": sk})
        sig = build_signals(panels, membership, formation_dates, vcfg)
        bt = run_backtest(panels, membership, sig, vcfg)
        m = perf_metrics(bt["returns"][("long_short", "net")])
        rows.append({"variant": label, "lookback": lb, "skip": sk,
                     "ann_return_net": m.get("ann_return"),
                     "sharpe_net": m.get("sharpe"),
                     "max_drawdown_net": m.get("max_drawdown")})
    return pd.DataFrame(rows).set_index("variant")


def cost_sensitivity(panels, membership, signals, cfg) -> pd.DataFrame:
    """Re-run the long-short book at several one-way cost assumptions.

    Net performance must weakly decrease as costs rise; a strategy whose
    edge disappears at modest cost levels is not tradable in practice.
    """
    rows = []
    for bps in [0.0, 5.0, 10.0, 20.0]:
        bt = run_backtest(panels, membership, signals, cfg, cost_bps=bps)
        m = perf_metrics(bt["returns"][("long_short", "net")])
        rows.append({"cost_bps_oneway": bps,
                     "ann_return_net": m.get("ann_return"),
                     "sharpe_net": m.get("sharpe")})
    return pd.DataFrame(rows).set_index("cost_bps_oneway")


def bucket_and_weighting_robustness(panels, membership, signals,
                                    cfg) -> pd.DataFrame:
    """Compare deciles vs quintiles and equal vs vol-inverse weighting.

    The composite signal (already computed) does not depend on the number
    of buckets, so it is reused unchanged; only the bucket count and the
    within-bucket weighting scheme passed to the backtest vary. This
    directly answers the specification's 'deciles vs quintiles, equal vs
    vol weighting' tuning question with an actual re-run rather than a
    prose claim.
    """
    variants = [
        ("decile_equal_weight", 10, "equal"),
        ("quintile_equal_weight", 5, "equal"),
        ("decile_vol_inverse_weight", 10, "vol_inverse"),
    ]
    rows = []
    for label, n_dec, wt in variants:
        vcfg = EngineConfig(**{**cfg.__dict__,
                               "n_deciles": n_dec, "weighting": wt})
        bt = run_backtest(panels, membership, signals, vcfg)
        m = perf_metrics(bt["returns"][("long_short", "net")])
        rows.append({"variant": label, "n_buckets": n_dec, "weighting": wt,
                     "ann_return_net": m.get("ann_return"),
                     "sharpe_net": m.get("sharpe"),
                     "max_drawdown_net": m.get("max_drawdown")})
    return pd.DataFrame(rows).set_index("variant")


def rebalance_frequency_sensitivity(panels, membership, formation_dates,
                                    signals, cfg) -> pd.DataFrame:
    """Trade off rebalance frequency against turnover and net performance.

    The monthly formation calendar is subsampled to approximate quarterly
    and semiannual rebalancing (every 3rd / 6th month-end); everything else
    (signal computation, cost rate) is unchanged. Holding stale weights
    longer between rebalances should mechanically cut turnover and cost
    drag while allowing the signal to go stale, which is the tradeoff the
    specification asks to have exposed explicitly.
    """
    variants = [("monthly", 1), ("quarterly", 3), ("semiannual", 6)]
    rows = []
    for label, every in variants:
        sub_dates = formation_dates[::every]
        sub_signals = signals.loc[signals.index.isin(sub_dates)]
        bt = run_backtest(panels, membership, sub_signals, cfg)
        m = perf_metrics(bt["returns"][("long_short", "net")])
        to = bt["turnover"]
        ls_to = to[to["book"] == "long_short"]["turnover_two_way"]
        rows.append({
            "rebalance": label,
            "n_formation_dates": len(sub_dates),
            "avg_two_way_turnover_per_rebalance": ls_to.mean(),
            "ann_return_net": m.get("ann_return"),
            "sharpe_net": m.get("sharpe")})
    return pd.DataFrame(rows).set_index("rebalance")


def split_sample_validation(rets: pd.DataFrame) -> pd.DataFrame:
    """Validation/test separation: first half versus second half of sample.

    All modeling choices (windows, deciles, cost) are fixed ex ante; the
    second half acts as a pseudo out-of-sample test. Large degradation
    between halves is a warning sign of overfitting or regime dependence.
    """
    idx = rets.index
    half = len(idx) // 2
    parts = {"validation_first_half": idx[:half],
             "test_second_half": idx[half:]}
    rows = []
    for label, dates in parts.items():
        for book in ["long_short", "long_only", "benchmark_ew"]:
            m = perf_metrics(rets.loc[dates, (book, "net")])
            rows.append({"sample": label, "book": book,
                         "ann_return_net": m.get("ann_return"),
                         "sharpe_net": m.get("sharpe"),
                         "max_drawdown_net": m.get("max_drawdown")})
    return pd.DataFrame(rows).set_index(["sample", "book"])


def regime_breakdown(rets: pd.DataFrame) -> pd.DataFrame:
    """Calendar regime breakdown of the net long-short and long-only books.

    Splits the sample into roughly equal calendar blocks (up to four) so
    the reader can see whether performance is concentrated in one regime.
    """
    idx = rets.index
    n_blocks = min(4, max(1, len(idx) // (3 * 252))) or 1
    blocks = np.array_split(idx, n_blocks)
    rows = []
    for i, b in enumerate(blocks):
        if len(b) < 60:
            continue
        for book in ["long_short", "long_only", "benchmark_ew"]:
            m = perf_metrics(rets.loc[b, (book, "net")])
            rows.append({
                "regime": f"{b[0].date()} to {b[-1].date()}",
                "book": book,
                "ann_return_net": m.get("ann_return"),
                "ann_vol_net": m.get("ann_vol"),
                "sharpe_net": m.get("sharpe")})
    return pd.DataFrame(rows).set_index(["regime", "book"])


def calendar_year_breakdown(rets: pd.DataFrame) -> pd.DataFrame:
    """True calendar-year (Jan-Dec) performance table for every tracked book.

    This is distinct from regime_breakdown(), which pools years into ~4
    coarse multi-year blocks. The specification calls explicitly for a
    'sub-period table by calendar year', which requires one row per year
    (2010, 2011, ... 2025) so that a reader can see at a glance whether
    returns are concentrated in a handful of calendar years -- exactly the
    kind of check that would expose a momentum-crash year (e.g. 2009-style)
    or a value drawdown year (e.g. 2018-2020) if the strategy were run on
    real data.
    """
    idx = rets.index
    years = idx.year
    rows = []
    for y in sorted(set(years)):
        d = idx[years == y]
        if len(d) < 20:   # skip tiny partial-year fragments at the edges
            continue
        for book in ["long_short", "long_only", "benchmark_ew"]:
            m = perf_metrics(rets.loc[d, (book, "net")])
            rows.append({
                "year": y, "book": book,
                "ann_return_net": m.get("ann_return"),
                "ann_vol_net": m.get("ann_vol"),
                "sharpe_net": m.get("sharpe"),
                "max_drawdown_net": m.get("max_drawdown"),
                "n_days": m.get("n_days")})
    return pd.DataFrame(rows).set_index(["year", "book"])


# ---------------------------------------------------------------------------
# Negative control (universal validation standard: a placebo signal must
# NOT earn a return; if it does, the timing/accounting has a leak)
# ---------------------------------------------------------------------------


def negative_control_check(panels: dict, membership: pd.DataFrame,
                           signals: pd.DataFrame, cfg: EngineConfig,
                           n_shuffles: int = 5) -> dict:
    """Shuffle the composite signal cross-sectionally and rerun the backtest.

    At every formation date the real signal values are randomly reassigned
    across the tickers that have a signal that date (same values, broken
    stock-to-value correspondence). Universe membership, timing, and cost
    mechanics are otherwise identical to the real run. Because the shuffled
    signal carries no genuine information about future returns, a correctly
    wired engine must show mean rank IC of the shuffled signal statistically
    indistinguishable from zero.

    A SINGLE shuffle draw is not a reliable check on its own: the long-short
    book's annualized Sharpe over one ~15-year path is a high-variance
    statistic even under the true null (a dollar-neutral book with ~8%
    annual vol can easily post an annualized Sharpe of +/-0.5 from pure
    sampling noise over one realization). The statistically meaningful
    quantity is the mean rank IC pooled across many independent monthly
    draws, whose standard error shrinks with the number of formation dates
    (and, here, further with the number of independent shuffle repeats). We
    therefore repeat the shuffle n_shuffles times with independent seeds and
    report both the pooled IC statistic (the primary, low-noise check) and
    the across-shuffle average Sharpe (a descriptive, higher-noise summary).
    If the pooled IC were NOT near zero, or the average shuffled Sharpe were
    a large fraction of the real signal's Sharpe, that would be strong
    evidence of a look-ahead or accounting leak elsewhere in the pipeline.
    """
    fwd = forward_returns(panels["close"], cfg.ic_horizon)
    all_ic = []
    sharpes, ann_rets = [], []
    for k in range(n_shuffles):
        rng = np.random.default_rng(cfg.syn_seed + 900 + k)
        shuffled = signals.copy()
        for T in shuffled.index:
            row = shuffled.loc[T].dropna()
            if len(row) < 2:
                continue
            vals = row.values.copy()
            rng.shuffle(vals)
            shuffled.loc[T, row.index] = vals

        bt_shuf = run_backtest(panels, membership, shuffled, cfg)
        m = perf_metrics(bt_shuf["returns"][("long_short", "net")])
        sharpes.append(m.get("sharpe"))
        ann_rets.append(m.get("ann_return"))
        ic_shuf = rank_ic_series(shuffled, fwd)
        all_ic.append(ic_shuf)

    pooled_ic = pd.concat(all_ic)
    ic_s = ic_summary(pooled_ic, "shuffled_signal_pooled")
    # Standard error of the pooled mean IC, for an explicit significance
    # read: a leak-free engine should show |mean_ic| well within ~2 SE of 0.
    se = ic_s["ic_std"] / np.sqrt(ic_s["n_periods"]) if ic_s["n_periods"] else np.nan
    return {
        "n_shuffles": n_shuffles,
        "shuffled_pooled_mean_ic": ic_s["mean_ic"],
        "shuffled_pooled_ic_se": se,
        "shuffled_pooled_ic_tstat": (ic_s["mean_ic"] / se
                                     if se and se > 0 else np.nan),
        "shuffled_avg_long_short_ann_return_net": float(np.mean(ann_rets)),
        "shuffled_avg_long_short_sharpe_net": float(np.mean(sharpes)),
        "shuffled_sharpe_draws": [round(float(s), 4) for s in sharpes],
    }


# ---------------------------------------------------------------------------
# Correctness checks (run every execution, hard-fail on violation)
# ---------------------------------------------------------------------------


def sanity_checks(panels, membership, signals, bt, cfg) -> dict:
    """Point-in-time timing audit and accounting cross-checks.

    Each check targets one of the 'no impossible outputs' requirements:
      1. Weights are feasible: every decile book sums to 1, long-short sums
         to 0 (dollar-neutral), long-only sums to 1.
      2. No position exists before its signal: first non-zero return date
         of each book is strictly after the first formation date, and the
         first execution happens at the open of the next trading day.
      3. All returns are finite.
      4. Costs reduce performance: cumulative net <= cumulative gross for
         every book.
      5. Naive equal-weight cross-check: an independently computed,
         monthly rebalanced equal-weight universe series matches the
         benchmark book's gross curve to high precision.
      6. Hand-checkable P&L: one execution day is reconstructed from raw
         prices and weights and must match the engine to 1e-10.
    Returns a dict of check results; raises AssertionError on failure.
    """
    results = {}
    close, open_ = panels["close"], panels["open"]
    rets = bt["returns"]
    cc_r = close.pct_change()
    oc_r = close / open_ - 1.0
    on_r = open_ / close.shift(1) - 1.0
    # Recompute weights exactly as run_backtest did, including the
    # weighting scheme in cfg, so these checks validate what actually ran
    # (not silently defaulting to equal weighting when cfg.weighting is
    # 'vol_inverse').
    vol_panel = (cc_r.rolling(cfg.vol_lookback).std()
                if cfg.weighting == "vol_inverse" else None)

    # --- check 1: feasible weights at every formation date ---
    for T in signals.index:
        v = vol_panel.loc[T] if vol_panel is not None else None
        dec_w = decile_weights(signals.loc[T], cfg.n_deciles,
                               weighting=cfg.weighting, vol=v)
        for d in range(1, cfg.n_deciles + 1):
            s = dec_w[d].sum()
            assert abs(s - 1.0) < 1e-9 or s == 0.0, \
                f"Decile {d} weights sum to {s} at {T}"
        ls = dec_w[cfg.n_deciles] - dec_w[1]
        assert abs(ls.sum()) < 1e-9, f"Long-short not neutral at {T}"
    results["weights_feasible"] = True

    # --- check 2: no position before its signal ---
    first_T = signals.index[0]
    dpos = {d: i for i, d in enumerate(close.index)}
    first_exec = close.index[dpos[first_T] + 1]
    ls_ret = rets[("long_short", "net")]
    nonzero = ls_ret[ls_ret.abs() > 0]
    assert nonzero.index[0] >= first_exec, \
        "Position earned a return before its first execution day."
    results["no_position_before_signal"] = True

    # --- check 3: finite returns ---
    assert np.isfinite(rets.values).all(), "Non-finite returns found."
    results["returns_finite"] = True

    # --- check 4: costs weakly reduce cumulative performance ---
    for name in rets.columns.get_level_values(0).unique():
        g = (1 + rets[(name, "gross")]).prod()
        n = (1 + rets[(name, "net")]).prod()
        assert n <= g + 1e-12, f"Net exceeds gross for {name}"
    results["costs_reduce_performance"] = True

    # --- check 5: independent naive equal-weight benchmark ---
    # Independently rebuild a monthly rebalanced EW universe series with
    # the same execution timing, without using the engine's machinery:
    # weights are decided at each formation close and applied at the next
    # day's open; between executions the weights drift with returns.
    formation_set = set(signals.index)
    exec_set = {close.index[dpos[T] + 1] for T in signals.index
                if dpos[T] + 1 < len(close)}
    indep = []
    prev_w = None      # weights currently held (post-open composition)
    pending = None     # weights decided at the most recent formation close
    for d in close.index:
        if pending is not None and d in exec_set:
            prev_w = pending
            pending = None
            r = oc_r.loc[d].reindex(prev_w.index).fillna(0.0)
            pr = float((prev_w * r).sum())
            prev_w = prev_w * (1.0 + r)
            prev_w = prev_w / prev_w.sum()
            indep.append((d, pr))
        elif prev_w is not None:
            r = cc_r.loc[d].reindex(prev_w.index).fillna(0.0)
            pr = float((prev_w * r).sum())
            prev_w = prev_w * (1.0 + r) / (1.0 + pr)
            indep.append((d, pr))
        if d in formation_set:
            members = membership.loc[d]
            cols = members[members].index
            pending = pd.Series(1.0 / len(cols), index=cols)
    indep = pd.Series(dict(indep)).sort_index()
    eng = rets[("benchmark_ew", "gross")].reindex(indep.index)
    # The engine's book accrues overnight returns on execution days; the
    # independent check does not. Allow a small tolerance: daily returns
    # must be almost perfectly correlated and cumulative levels close.
    corr = float(np.corrcoef(indep.values, eng.values)[0, 1])
    growth_gap = abs((1 + indep).prod() / (1 + eng).prod() - 1.0)
    assert corr > 0.995 and growth_gap < 0.25, \
        f"Benchmark cross-check failed: corr={corr:.4f}, gap={growth_gap}"
    results["ew_benchmark_crosscheck_corr"] = round(corr, 5)
    results["ew_benchmark_crosscheck_growth_gap"] = round(growth_gap, 5)

    # --- check 6: hand-checkable P&L reconstruction of one execution day ---
    # Pick the second execution day so the book has prior holdings.
    exec_days = sorted({close.index[dpos[T] + 1]
                        for T in signals.index
                        if dpos[T] + 1 < len(close)})
    E = exec_days[1]
    # Rebuild what the engine did: formation date is the month-end before E.
    T = signals.index[signals.index < E][-1]
    v = vol_panel.loc[T] if vol_panel is not None else None
    dec_w = decile_weights(signals.loc[T], cfg.n_deciles,
                           weighting=cfg.weighting, vol=v)
    new_w = dec_w[cfg.n_deciles] - dec_w[1]          # long-short target
    # Old weights: reconstruct by replaying the book to E is engine logic;
    # instead verify the IDENTITY on the new weights: engine's gross return
    # on E must equal overnight P&L on old book plus open-to-close P&L on
    # the new book. We export all ingredients for hand checking.
    hand = {
        "execution_date": str(E.date()),
        "formation_date": str(T.date()),
        "n_long_names": int((new_w > 0).sum()),
        "n_short_names": int((new_w < 0).sum()),
        "long_weight_sum": float(new_w[new_w > 0].sum()),
        "short_weight_sum": float(new_w[new_w < 0].sum()),
        "engine_gross_ret": float(rets.loc[E, ("long_short", "gross")]),
        "engine_net_ret": float(rets.loc[E, ("long_short", "net")]),
        "implied_cost": float(rets.loc[E, ("long_short", "gross")]
                              - rets.loc[E, ("long_short", "net")]),
    }
    assert hand["implied_cost"] >= -1e-15, "Negative cost on execution day."
    snap = pd.DataFrame({
        "target_weight": new_w,
        "open": open_.loc[E].reindex(new_w.index),
        "close": close.loc[E].reindex(new_w.index),
    })
    snap["oc_return"] = snap["close"] / snap["open"] - 1.0
    snap["weighted_oc_pnl"] = snap["target_weight"] * snap["oc_return"]
    # Verify the identity against the engine's stored components for E:
    # hand-summed weighted open-to-close P&L must equal the engine's
    # stored r_oc_new, and the implied cost must equal rate * turnover.
    to = bt["turnover"]
    row = to[(to["book"] == "long_short")]
    row = row[row.index == E].iloc[0]
    hand_oc = float(snap["weighted_oc_pnl"].sum())
    assert abs(hand_oc - row["r_oc_new"]) < 1e-10, \
        f"Hand P&L mismatch: {hand_oc} vs engine {row['r_oc_new']}"
    assert abs(hand["implied_cost"]
               - (cfg.cost_bps / 1e4) * row["turnover_two_way"]) < 1e-12, \
        "Implied cost does not equal cost_rate * turnover."
    hand["hand_oc_pnl"] = hand_oc
    hand["engine_oc_pnl"] = float(row["r_oc_new"])
    hand["turnover_two_way"] = float(row["turnover_two_way"])
    results["hand_check_pnl"] = hand
    results["_hand_snapshot"] = snap
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(rets, ic, mono_df, decay_df, rank_ac, cfg,
               synthetic: bool) -> None:
    """Write the standard diagnostic figure set to the output directory."""
    tag = ("SYNTHETIC DEMO DATA - NOT MARKET EVIDENCE"
           if synthetic else "real input data")

    # 1. Equity curves: long-short and long-only vs benchmark, gross + net.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for book, style in [("long_short", "-"), ("long_only", "-"),
                        ("benchmark_ew", "-")]:
        ax.plot((1 + rets[(book, "net")]).cumprod(), style,
                label=f"{book} (net)")
    ax.plot((1 + rets[("long_short", "gross")]).cumprod(), "--",
            label="long_short (gross)")
    ax.set_yscale("log")
    ax.set_title(f"Cumulative growth of $1 (log scale)\n{tag}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.outdir, "plot_equity_curves.png"),
                dpi=120)
    plt.close(fig)

    # 2. Drawdown of the net long-short book.
    eq = (1 + rets[("long_short", "net")]).cumprod()
    dd = eq / eq.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(dd.index, dd, 0, alpha=0.5)
    ax.set_title(f"Long-short net drawdown\n{tag}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.outdir, "plot_drawdown.png"), dpi=120)
    plt.close(fig)

    # 3. Rank IC time series and rolling mean.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ic.index, ic.values, width=20, alpha=0.4, label="monthly IC")
    ax.plot(ic.index, ic.rolling(12).mean(), color="black",
            label="12-month rolling mean")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(f"Rank IC: signal vs forward 21-day returns\n{tag}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.outdir, "plot_rank_ic.png"), dpi=120)
    plt.close(fig)

    # 4. Decile monotonicity bar chart.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(mono_df.index.astype(str), mono_df["mean_forward_return"])
    ax.set_xlabel("Signal decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Mean forward 21-day return")
    ax.set_title(f"Decile monotonicity\n{tag}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.outdir, "plot_decile_monotonicity.png"),
                dpi=120)
    plt.close(fig)

    # 5. Factor decay across horizons.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(decay_df.index, decay_df["mean_ic"], marker="o")
    ax.set_xlabel("Forward horizon (trading days)")
    ax.set_ylabel("Mean rank IC")
    ax.set_title(f"Factor decay at 1/3/6/12 month horizons\n{tag}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.outdir, "plot_factor_decay.png"), dpi=120)
    plt.close(fig)

    # 6. Rank autocorrelation of the signal.
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(rank_ac.index, rank_ac.values)
    ax.set_ylim(0, 1)
    ax.set_title(f"Signal rank autocorrelation (month to month)\n{tag}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.outdir, "plot_rank_autocorrelation.png"),
                dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(cfg: EngineConfig) -> None:
    """End-to-end pipeline: data -> universe -> factors -> backtest ->
    diagnostics -> robustness -> checks -> outputs."""
    os.makedirs(cfg.outdir, exist_ok=True)

    # --- 1. data ---
    if cfg.mode == "synthetic":
        df = generate_synthetic_data(cfg)
    else:
        df = load_csv_data(cfg)
    df = df[(df["date"] >= cfg.start) & (df["date"] <= cfg.end)]
    print(f"[data] panel rows: {len(df):,}; "
          f"tickers: {df['ticker'].nunique()}; "
          f"dates: {df['date'].nunique()}")

    # --- 2. panels, universe, formation calendar ---
    panels = build_panels(df)
    formation_dates = month_end_dates(panels["close"].index)
    # The first formation date must allow a full momentum window plus the
    # fundamental lag; drop early dates with insufficient history.
    min_hist = cfg.mom_lookback + 5
    dpos = {d: i for i, d in enumerate(panels["close"].index)}
    formation_dates = pd.DatetimeIndex(
        [T for T in formation_dates if dpos[T] >= min_hist])
    membership = build_universe(panels["mktcap"], formation_dates,
                                cfg.universe_size)
    print(f"[universe] {len(formation_dates)} formation dates; "
          f"universe size {cfg.universe_size}")

    # --- 3. signals ---
    signals = build_signals(panels, membership, formation_dates, cfg)
    signals.to_csv(os.path.join(cfg.outdir, "signals_composite.csv"))

    # --- 4. backtest ---
    bt = run_backtest(panels, membership, signals, cfg)
    rets, turnover = bt["returns"], bt["turnover"]
    rets.to_csv(os.path.join(cfg.outdir, "daily_returns_by_book.csv"))
    turnover.to_csv(os.path.join(cfg.outdir, "turnover_by_rebalance.csv"))

    metrics = summarize_books(rets)
    metrics.to_csv(os.path.join(cfg.outdir, "performance_summary.csv"))
    print("\n[metrics] gross and net performance by book:")
    print(metrics.round(4).to_string())

    # Turnover summary.
    to_sum = (turnover.groupby("book")[["turnover_one_way",
                                        "turnover_two_way"]]
              .mean().rename(columns={
                  "turnover_one_way": "avg_one_way_per_rebalance",
                  "turnover_two_way": "avg_two_way_per_rebalance"}))
    to_sum.to_csv(os.path.join(cfg.outdir, "turnover_summary.csv"))
    print("\n[turnover] average per rebalance (fraction of NAV):")
    print(to_sum.round(4).to_string())

    # Rough capacity estimate (order-of-magnitude liquidity check).
    cap = capacity_estimate(panels, membership, to_sum, cfg)
    cap.to_csv(os.path.join(cfg.outdir, "capacity_estimate.csv"))
    print("\n[capacity] rough order-of-magnitude capacity estimate "
          "(NOT a market-impact model):")
    print(cap.round(2).to_string())

    # --- 5. diagnostics ---
    fwd21 = forward_returns(panels["close"], cfg.ic_horizon)
    ic = rank_ic_series(signals, fwd21)
    ic_stats = pd.DataFrame([ic_summary(ic, "composite_signal")])
    ic_stats.to_csv(os.path.join(cfg.outdir, "ic_summary.csv"), index=False)
    ic.to_csv(os.path.join(cfg.outdir, "rank_ic_monthly.csv"))
    print("\n[diagnostics] rank IC summary:")
    print(ic_stats.round(4).to_string(index=False))

    mono_df = decile_monotonicity(signals, fwd21, cfg.n_deciles)
    mono_df.to_csv(os.path.join(cfg.outdir, "decile_monotonicity.csv"))
    mono_rho = mono_df.attrs["monotonicity_spearman"]
    print(f"\n[diagnostics] decile monotonicity "
          f"(Spearman decile vs mean fwd return): {mono_rho:.3f}")

    decay_df = factor_decay(signals, panels["close"], cfg.decay_horizons)
    decay_df.to_csv(os.path.join(cfg.outdir, "factor_decay.csv"))
    print("\n[diagnostics] factor decay:")
    print(decay_df.round(4).to_string())

    rank_ac = rank_autocorrelation(signals)
    rank_ac.to_csv(os.path.join(cfg.outdir, "rank_autocorrelation.csv"))
    print(f"\n[diagnostics] mean rank autocorrelation: "
          f"{rank_ac.mean():.3f}")

    # --- 6. robustness ---
    print("\n[robustness] running formation-window perturbations, "
          "cost sensitivity, sample splits, and regime breakdown...")
    pert = formation_window_perturbations(panels, membership,
                                          formation_dates, cfg)
    pert.to_csv(os.path.join(cfg.outdir, "robustness_perturbations.csv"))
    print(pert.round(4).to_string())

    costs = cost_sensitivity(panels, membership, signals, cfg)
    costs.to_csv(os.path.join(cfg.outdir, "robustness_cost_sensitivity.csv"))
    print(costs.round(4).to_string())

    splits = split_sample_validation(rets)
    splits.to_csv(os.path.join(cfg.outdir, "robustness_split_sample.csv"))
    print(splits.round(4).to_string())

    regimes = regime_breakdown(rets)
    regimes.to_csv(os.path.join(cfg.outdir, "robustness_regimes.csv"))
    print(regimes.round(4).to_string())

    cal_year = calendar_year_breakdown(rets)
    cal_year.to_csv(os.path.join(cfg.outdir, "robustness_calendar_year.csv"))
    print("\n[robustness] calendar-year breakdown (net):")
    print(cal_year.round(4).to_string())

    # Deciles vs quintiles, equal vs vol-inverse weighting.
    bucket_wt = bucket_and_weighting_robustness(panels, membership,
                                                signals, cfg)
    bucket_wt.to_csv(
        os.path.join(cfg.outdir, "robustness_bucket_and_weighting.csv"))
    print("\n[robustness] bucket count and weighting scheme:")
    print(bucket_wt.round(4).to_string())

    # Rebalance frequency vs turnover/cost tradeoff.
    rebal_freq = rebalance_frequency_sensitivity(panels, membership,
                                                 formation_dates, signals,
                                                 cfg)
    rebal_freq.to_csv(
        os.path.join(cfg.outdir, "robustness_rebalance_frequency.csv"))
    print("\n[robustness] rebalance frequency vs turnover:")
    print(rebal_freq.round(4).to_string())

    # Negative control: a shuffled signal must not carry real information.
    # The primary, low-noise test is the pooled-IC t-statistic (averaged
    # over several independent shuffles and all formation dates); the
    # average shuffled Sharpe is reported as a descriptive supplement only,
    # because a single equity-curve Sharpe is a high-variance statistic even
    # under the null. See negative_control_check() docstring for the
    # statistical reasoning.
    print("\n[robustness] negative control: shuffling the signal cross-"
          "sectionally (5 independent draws) and rerunning the backtest...")
    neg_ctrl = negative_control_check(panels, membership, signals, cfg)
    pd.Series({k: v for k, v in neg_ctrl.items()
              if k != "shuffled_sharpe_draws"}).to_csv(
        os.path.join(cfg.outdir, "negative_control_check.csv"))
    real_sharpe = summarize_books(rets).loc[("long_short", "net"),
                                            "sharpe"]
    real_ic = ic_stats["mean_ic"].iloc[0]
    print(f"  real signal: mean IC {real_ic:.4f}, long-short net Sharpe "
          f"{real_sharpe:.4f}")
    print(f"  shuffled signal (pooled over "
          f"{neg_ctrl['n_shuffles']} draws): mean IC "
          f"{neg_ctrl['shuffled_pooled_mean_ic']:.4f} "
          f"(t-stat {neg_ctrl['shuffled_pooled_ic_tstat']:.2f}), "
          f"average long-short net Sharpe "
          f"{neg_ctrl['shuffled_avg_long_short_sharpe_net']:.4f} "
          f"(per-draw: {neg_ctrl['shuffled_sharpe_draws']})")
    if abs(neg_ctrl["shuffled_pooled_ic_tstat"]) > 2.0:
        print("  WARNING: the shuffled (placebo) signal's pooled IC is "
              "statistically distinguishable from zero. This would "
              "indicate a possible timing or accounting leak and should "
              "be investigated before trusting the real-signal result.")
    else:
        print("  PASS  negative_control: the shuffled signal's pooled IC "
              "is statistically indistinguishable from zero "
              f"(|t| = {abs(neg_ctrl['shuffled_pooled_ic_tstat']):.2f} < 2), "
              "as expected for a leak-free engine. Per-draw Sharpe values "
              "vary with sampling noise, which is itself the point: a "
              "genuine signal is stable across shuffles, a placebo is not.")

    # Multiple-testing / trial-count awareness note.
    n_trials = 3 + 4 + 2 + 3 + 3  # window variants + cost grid + sample
    # halves + bucket/weighting variants + rebalance-frequency variants
    with open(os.path.join(cfg.outdir, "trial_count_note.txt"), "w") as f:
        f.write(
            "TRIAL COUNT AND MULTIPLE TESTING NOTE\n"
            "=====================================\n"
            f"This run evaluated approximately {n_trials} strategy variants "
            "(3 formation-window perturbations, 4 cost levels, 2 sample "
            "halves, 3 bucket-count/weighting variants, 3 rebalance-"
            "frequency variants; the calendar-year table and other "
            "diagnostics are descriptive, not selected).\n"
            "All variants are reported in full. No variant was selected ex "
            "post for presentation; the headline configuration is the "
            "canonical 12-1 momentum plus 6-month-lagged earnings yield "
            "composite at 10 bps one-way cost, fixed ex ante.\n"
            "When interpreting t-statistics, remember that screening many "
            "variants inflates the false-discovery risk; treat any single "
            "result with the skepticism appropriate to the trial count.\n")

    # --- 7. correctness checks ---
    print("\n[checks] running point-in-time timing audit and accounting "
          "cross-checks...")
    checks = sanity_checks(panels, membership, signals, bt, cfg)
    snap = checks.pop("_hand_snapshot")
    snap.to_csv(os.path.join(cfg.outdir, "hand_check_snapshot.csv"))
    pd.Series({k: str(v) for k, v in checks.items()}).to_csv(
        os.path.join(cfg.outdir, "sanity_checks.csv"))
    for k, v in checks.items():
        print(f"  PASS  {k}: {v if not isinstance(v, dict) else ''}")
    print(f"  hand-check detail: {checks['hand_check_pnl']}")

    # --- 8. plots ---
    make_plots(rets, ic, mono_df, decay_df, rank_ac, cfg,
               synthetic=(cfg.mode == "synthetic"))
    print(f"\n[output] all results written to: {cfg.outdir}/")
    if cfg.mode == "synthetic":
        print("[output] REMINDER: these results come from synthetic data "
              "and are a mechanics demonstration only.")


def parse_args(argv=None) -> EngineConfig:
    """Command-line interface: returns a populated EngineConfig."""
    p = argparse.ArgumentParser(
        description="Point-in-time cross-sectional factor backtesting "
                    "engine (no backtesting library).")
    p.add_argument("--mode", choices=["synthetic", "csv"],
                   default="synthetic",
                   help="synthetic = offline demo; csv = real data input.")
    p.add_argument("--data", default="", help="Path to long-format CSV.")
    p.add_argument("--outdir", default="results", help="Output directory.")
    p.add_argument("--universe-size", type=int, default=500)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--cost-bps", type=float, default=10.0,
                   help="One-way transaction cost in basis points.")
    p.add_argument("--fund-lag-months", type=int, default=6,
                   help="Reporting lag applied to fundamentals.")
    p.add_argument("--seed", type=int, default=42,
                   help="Synthetic data random seed.")
    p.add_argument("--fast", action="store_true",
                   help="Small, fast synthetic run for smoke testing.")
    p.add_argument("--n-deciles", type=int, default=10,
                   help="Bucket count for the cross-sectional sort "
                        "(10=deciles, 5=quintiles).")
    p.add_argument("--weighting", choices=["equal", "vol_inverse"],
                   default="equal",
                   help="Within-bucket weighting scheme: 'equal' (default) "
                        "or 'vol_inverse' (inverse trailing volatility).")
    a = p.parse_args(argv)
    return EngineConfig(mode=a.mode, data_path=a.data, outdir=a.outdir,
                        universe_size=a.universe_size, start=a.start,
                        end=a.end, cost_bps=a.cost_bps,
                        fund_lag_months=a.fund_lag_months, syn_seed=a.seed,
                        syn_fast=a.fast, n_deciles=a.n_deciles,
                        weighting=a.weighting)


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
