"""Rolling-start bookkeeping, the solver on a closed-form path, and the cache layer."""

from __future__ import annotations

import json

import pytest

from tw_pledge_sim import StockInputs, run_rolling, solve_breakeven_rate
from tw_pledge_sim.data import (
    extract_dividend_events,
    load_adjusted_price_series,
    load_stock_inputs,
)
from tw_pledge_sim.rolling import nearest_trading_day_idx, rolling_triples
from tw_pledge_sim.synthetic import weekdays


def test_nearest_trading_day_idx():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    assert nearest_trading_day_idx(dates, "2020-01-04") == 2
    assert nearest_trading_day_idx(dates, "2020-01-07") is None


def test_rolling_triples_skip_windows_past_the_sample():
    dates = weekdays("2010-01-04", "2012-06-29")
    triples = rolling_triples(dates, [1, 2])
    assert triples, "expected at least one window"
    assert all(
        dates[e] >= f"{int(dates[o][:4]) + y}{dates[o][4:]}" for o, y, e in triples
    )
    assert max(y for _, y, _ in triples) == 2
    last_two_year_onset = max(dates[o] for o, y, _ in triples if y == 2)
    assert last_two_year_onset < "2010-07-01"


def test_zero_pledge_rate_matches_unlevered_equity():
    dates = weekdays("2010-01-04", "2013-12-31")
    prices = {d: 100.0 + 0.05 * i for i, d in enumerate(dates)}
    inputs = StockInputs(dates, prices, set(), {}, [])
    rows = run_rolling("S", inputs, pledge_rates=[0.0], holding_periods_years=[1])
    assert rows
    for r in rows:
        assert r["final_equity"] == pytest.approx(r["final_equity_unlevered"])
        assert r["liquidated"] is False


def test_breakeven_rate_matches_closed_form_on_a_steady_uptrend():
    """Constant daily growth g: leverage breaks even when the loan rate equals the
    asset's annual growth (no dividends), up to the simple-interest approximation."""
    dates = weekdays("2010-01-04", "2014-12-31")
    g = 1.0003
    prices = {d: 100.0 * g**i for i, d in enumerate(dates)}
    inputs = StockInputs(dates, prices, set(), {}, [])
    rate = solve_breakeven_rate(inputs, 0.2, 2, "median", lo=0.005, hi=0.4)
    annual_growth = g**252 - 1.0
    assert rate == pytest.approx(annual_growth, abs=0.02)


def write_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TW_PLEDGE_SIM_DATA", str(tmp_path))
    series = {"2024-01-02": 100.0, "2024-01-03": 101.0, "2024-07-15": 99.0}
    (tmp_path / "prices_raw_AAAA.json").write_text(
        json.dumps({"stock_no": "AAAA", "series": series, "no_data_months": []})
    )
    table = {
        "stat": "OK",
        "data": [
            ["113年07月15日", "AAAA", "x", "x", "x", "1.50"],
            ["113年07月15日", "BBBB", "x", "x", "x", "9.00"],
        ],
    }
    (tmp_path / "dividends_2024.json").write_text(json.dumps(table))


def test_cache_roundtrip_builds_engine_inputs(tmp_path, monkeypatch):
    write_cache(tmp_path, monkeypatch)
    assert extract_dividend_events("AAAA", 2024, 2024) == [
        {"date": "2024-07-15", "ex_rights_value": 1.5}
    ]
    assert extract_dividend_events("CCCC", 2024, 2024) == []
    adjusted, events = load_adjusted_price_series("AAAA")
    assert adjusted["2024-01-02"] == 100.0 and events == []
    load_stock_inputs.cache_clear()
    inputs = load_stock_inputs("AAAA", 2024)
    assert inputs.dates == sorted(adjusted)
    assert inputs.ex_dates == {"2024-07-15"}


def test_missing_cache_is_reported_not_invented(tmp_path, monkeypatch):
    monkeypatch.setenv("TW_PLEDGE_SIM_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_adjusted_price_series("NOPE")
