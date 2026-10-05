"""Household overlay tests on hand-built price paths (fictional numbers)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tw_pledge_sim import (
    BrokerProfile,
    HouseholdProfile,
    StockInputs,
    run_household_overlay,
    solve_min_buffer,
)
from tw_pledge_sim.household import mortgage_monthly_payment

RATE = BrokerProfile(annual_rate=0.0385)


def daily_dates(n_months: int, start: str = "2010-01-04") -> list[str]:
    d0 = date.fromisoformat(start)
    end_y = d0.year + (d0.month - 1 + n_months) // 12
    end_m = (d0.month - 1 + n_months) % 12 + 1
    stop = date(end_y, end_m, 1) + timedelta(days=5)
    out, cur = [], d0
    while cur < stop:
        if cur.weekday() < 5:
            out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def flat_inputs(dates: list[str], prices: dict[str, float]) -> StockInputs:
    return StockInputs(dates, prices, set(), {}, [])


def crash_path(dates: list[str], floor: float = 20.0) -> dict[str, float]:
    months = sorted({d[:7] for d in dates})
    out = {}
    for d in dates:
        mi = months.index(d[:7])
        if mi <= 5:
            out[d] = 100.0
        elif mi <= 10:
            out[d] = max(floor, 100.0 - (100.0 - floor) * (mi - 5) / 5.0)
        else:
            out[d] = floor
    return out


def overlay(collat, loan, buffer=0.0, pledge_rate=0.25, profile=HouseholdProfile()):
    return run_household_overlay(
        collat,
        loan,
        0,
        len(collat.dates) - 1,
        pledge_rate,
        "X",
        broker=RATE,
        initial_buffer=buffer,
        profile=profile,
    )


def test_mortgage_payment_annuity_and_zero_rate():
    assert mortgage_monthly_payment(3_000_000.0, 0.025, 30) == pytest.approx(
        11_853.0, abs=1.0
    )
    assert mortgage_monthly_payment(4_800_000.0, 0.0, 40) == pytest.approx(10_000.0)


def test_monthly_cash_flow_matches_hand_calculation():
    """Flat price, no dividends: surplus minus DCA pays interest, the rest builds the buffer."""
    dates = daily_dates(12)
    inputs = flat_inputs(dates, {d: 100.0 for d in dates})
    profile = HouseholdProfile()
    r = overlay(inputs, inputs, profile=profile)
    loan = 0.25 * profile.portfolio_principal
    avail = (
        profile.salary_monthly
        - profile.mortgage_payment()
        - profile.living_monthly
        - profile.dca_monthly
    )
    assert r.salary_covers_interest_all_months is True
    assert r.dividends_diverted_to_interest_total == 0.0
    assert r.collateral_units_sold_value_total == 0.0
    assert r.n_margin_calls == 0 and r.forced_disposal is False
    assert r.monthly_buffer_delta_min >= avail - loan * 0.0385 * 31 / 365.0
    assert r.monthly_buffer_delta_max <= avail - loan * 0.0385 * 28 / 365.0 + 1e-6
    assert r.monthly_buffer_delta_min > 0.0


def test_margin_call_forces_disposal_unless_buffer_cures_it():
    dates = daily_dates(24)
    collat = flat_inputs(dates, crash_path(dates))
    loan = flat_inputs(dates, {d: 100.0 for d in dates})
    bare = overlay(collat, loan, buffer=0.0)
    assert bare.n_margin_calls >= 1
    assert bare.forced_disposal is True and bare.disposal_month is not None
    cured = overlay(collat, loan, buffer=6_000_000.0)
    assert cured.n_margin_calls >= 1
    assert cured.forced_disposal is False


def test_forced_disposal_is_monotone_in_the_buffer():
    dates = daily_dates(24)
    collat = flat_inputs(dates, crash_path(dates, floor=18.0))
    loan = flat_inputs(dates, {d: 100.0 for d in dates})
    flags = [
        overlay(collat, loan, buffer=b).forced_disposal
        for b in (0.0, 5e5, 1e6, 1.5e6, 2e6, 3e6, 6e6)
    ]
    assert flags[0] is True and flags[-1] is False
    assert flags == sorted(flags, reverse=True)  # True...True, False...False


def test_loan_target_price_flows_into_final_wealth():
    dates = daily_dates(18)
    collat = flat_inputs(dates, {d: 100.0 for d in dates})
    flat = flat_inputs(dates, {d: 100.0 for d in dates})
    n = len(dates) - 1
    doubling = flat_inputs(
        dates, {d: 100.0 + 100.0 * i / n for i, d in enumerate(dates)}
    )
    profile = HouseholdProfile()
    gain = (
        overlay(collat, doubling).final_net_wealth
        - overlay(collat, flat).final_net_wealth
    )
    loan_shares = 0.25 * profile.portfolio_principal / 100.0
    assert gain == pytest.approx(loan_shares * 100.0, rel=0.02)


def test_min_buffer_is_zero_when_nothing_is_ever_disposed():
    dates = daily_dates(30, start="2008-01-02")
    flat = flat_inputs(dates, {d: 100.0 for d in dates})
    out = solve_min_buffer(flat, flat, 0.25, "X", 1, broker=RATE)
    assert out["finite"] is True and out["min_buffer"] == 0.0
