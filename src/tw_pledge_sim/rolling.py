"""Rolling-start scans over historical paths and the break-even rate solver."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .broker import GENERIC, BrokerProfile
from .data import StockInputs
from .engine import PledgeSimulator

HOLDING_PERIODS_YEARS = [5, 10, 15]
PLEDGE_RATES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]


def nearest_trading_day_idx(dates: list[str], target_iso: str) -> Optional[int]:
    """Index of the first trading day on or after ``target_iso``."""
    for i, d in enumerate(dates):
        if d >= target_iso:
            return i
    return None


def _add_years(iso: str, years: int) -> str:
    return f"{int(iso[:4]) + years}{iso[4:]}"


def rolling_triples(
    dates: list[str], holding_years: list[int]
) -> list[tuple[int, int, int]]:
    """All (onset_idx, years, end_idx): one onset per calendar month.

    Windows whose end falls past the sample are skipped, never fabricated.
    """
    triples = []
    for ym in sorted({d[:7] for d in dates}):
        onset_idx = nearest_trading_day_idx(dates, ym + "-01")
        if onset_idx is None:
            continue
        for years in holding_years:
            end_idx = nearest_trading_day_idx(
                dates, _add_years(dates[onset_idx], years)
            )
            if end_idx is not None:
                triples.append((onset_idx, years, end_idx))
    return triples


def run_rolling(
    stock_no: str,
    inputs: StockInputs,
    broker: BrokerProfile = GENERIC,
    dividend_income_tax_rate: float = 0.0,
    pledge_rates: Optional[list[float]] = None,
    holding_periods_years: Optional[list[int]] = None,
    reinvest_dividends: bool = False,
) -> list[dict]:
    """Full scan: rolling start x holding period x pledge rate."""
    pledge_rates = PLEDGE_RATES if pledge_rates is None else pledge_rates
    years_list = (
        HOLDING_PERIODS_YEARS
        if holding_periods_years is None
        else holding_periods_years
    )
    dates = inputs.dates
    sim = PledgeSimulator(
        dates=dates,
        prices=inputs.prices,
        ex_dividend_dates=inputs.ex_dates,
        ex_dividend_value=inputs.ex_values,
        dividend_income_tax_rate=dividend_income_tax_rate,
        split_events=inputs.split_events,
        reinvest_dividends=reinvest_dividends,
        **broker.simulator_kwargs(),
    )
    rows = []
    for onset_idx, years, end_idx in rolling_triples(dates, years_list):
        onset_date = dates[onset_idx]
        for pr in pledge_rates:
            r = sim.run(onset_idx, end_idx, pr)
            rows.append(
                {
                    "stock_no": stock_no,
                    "onset_date": onset_date,
                    "holding_years": years,
                    "pledge_rate": pr,
                    "onset_year": int(onset_date[:4]),
                    "n_margin_calls": r.n_margin_calls,
                    "liquidated": r.liquidated,
                    "term_expired_forced_sale": r.term_expired_forced_sale,
                    "final_equity": r.final_equity,
                    "final_equity_unlevered": r.final_equity_unlevered,
                    "max_drawdown_equity": r.max_drawdown_equity,
                    "total_interest_paid": r.total_interest_paid,
                }
            )
    return rows


def relative_multiples(
    inputs: StockInputs,
    pledge_rate: float,
    holding_years: int,
    broker: BrokerProfile = GENERIC,
    dividend_income_tax_rate: float = 0.0,
    reinvest_dividends: bool = False,
) -> list[float]:
    """``final_equity / final_equity_unlevered`` for every rolling start."""
    rows = run_rolling(
        "x",
        inputs,
        broker=broker,
        dividend_income_tax_rate=dividend_income_tax_rate,
        pledge_rates=[pledge_rate],
        holding_periods_years=[holding_years],
        reinvest_dividends=reinvest_dividends,
    )
    return [
        r["final_equity"] / r["final_equity_unlevered"]
        for r in rows
        if r["final_equity_unlevered"] and r["final_equity_unlevered"] > 0
    ]


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def win_rate(xs: list[float], threshold: float = 1.0) -> float:
    return sum(1 for x in xs if x > threshold) / len(xs) if xs else float("nan")


def solve_breakeven_rate(
    inputs: StockInputs,
    pledge_rate: float,
    holding_years: int = 10,
    metric: str = "median",
    broker: BrokerProfile = GENERIC,
    lo: float = 0.005,
    hi: float = 0.25,
    tol: float = 1e-5,
    max_iter: int = 60,
    reinvest_dividends: bool = False,
) -> float:
    """Bisect the constant loan rate at which leverage breaks even.

    metric 'median': median(relative multiple) == 1.0.
    metric 'winrate': share of starts with multiple > 1.0 equals 0.5.
    The multiple falls monotonically in the rate, so bisection applies.
    """

    def value_at(rate: float) -> float:
        ms = relative_multiples(
            inputs,
            pledge_rate,
            holding_years,
            broker=replace(broker, annual_rate=rate),
            reinvest_dividends=reinvest_dividends,
        )
        return median(ms) if metric == "median" else win_rate(ms)

    target = 1.0 if metric == "median" else 0.5
    f_lo = value_at(lo) - target
    f_hi = value_at(hi) - target
    if f_lo * f_hi > 0:
        # No sign change: return the nearer endpoint; the caller checks the bracket.
        return lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = value_at(mid) - target
        if abs(hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def breakeven_rate_scan(
    inputs: StockInputs,
    pledge_rate: float = 0.30,
    holding_years: int = 10,
    rates: Optional[list[float]] = None,
) -> list[tuple[float, float, float]]:
    """Loan rate grid (default 2%-8% in 0.5% steps) -> (rate, median multiple, win rate)."""
    if rates is None:
        rates = [0.02 + 0.005 * k for k in range(13)]
    out = []
    for rate in rates:
        ms = relative_multiples(
            inputs,
            pledge_rate,
            holding_years,
            broker=replace(GENERIC, annual_rate=rate),
        )
        out.append((rate, median(ms), win_rate(ms)))
    return out
