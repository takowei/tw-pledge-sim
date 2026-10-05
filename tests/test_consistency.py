"""Numerical consistency with the reference engine this package was extracted from.

``fixtures/legacy_reference.json`` holds values computed once by the reference
implementation on the deterministic data of ``synthetic.py`` (a rolling-start scan
with 1:4 split, ex-dividend dates and a crash).  The refactor must reproduce them.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from tw_pledge_sim.synthetic import make_inputs

from tw_pledge_sim import (
    BrokerProfile,
    HouseholdProfile,
    breakeven_rate_scan,
    forced_disposal_stats,
    relative_multiples,
    run_overlay_matrix_cell,
    run_rolling,
    solve_breakeven_rate,
    solve_min_buffer,
)
from tw_pledge_sim.rolling import median, win_rate

REF = json.loads((Path(__file__).parent / "fixtures/legacy_reference.json").read_text())
COLLAT = make_inputs(1)
LOAN = make_inputs(2)
PROFILE = HouseholdProfile(
    portfolio_principal=REF["meta"]["profile"]["principal"],
    salary_monthly=REF["meta"]["profile"]["salary"],
    living_monthly=REF["meta"]["profile"]["living"],
    dca_monthly=REF["meta"]["profile"]["dca"],
    mortgage_principal=REF["meta"]["profile"]["m_principal"],
    mortgage_years=REF["meta"]["profile"]["m_years"],
    mortgage_annual_rate=REF["meta"]["profile"]["m_rate"],
)
APPROX = {"rel": 1e-9, "abs": 1e-9}


def assert_row_equal(new: dict, ref: dict) -> None:
    assert new.keys() == ref.keys()
    for key, value in ref.items():
        if isinstance(value, float):
            assert new[key] == pytest.approx(value, **APPROX), key
        else:
            assert new[key] == value, key


def test_split_adjusted_series_matches_reference():
    chk = REF["adjusted_checksum"]
    assert len(COLLAT.prices) == chk["n"]
    assert sum(COLLAT.prices.values()) == pytest.approx(chk["sum"], **APPROX)
    assert COLLAT.prices[COLLAT.dates[0]] == pytest.approx(chk["first"], **APPROX)
    assert COLLAT.prices[COLLAT.dates[-1]] == pytest.approx(chk["last"], **APPROX)
    detected = [e for e in COLLAT.split_events if e["is_split"]]
    assert len(detected) == len(REF["split_events"])
    for new, ref in zip(detected, REF["split_events"]):
        assert_row_equal(new, ref)


@pytest.mark.parametrize(
    "case", REF["rolling"], ids=lambda c: f"{c['rate']}-{c['mode']}"
)
def test_rolling_scan_matches_reference(case):
    broker = BrokerProfile(annual_rate=case["rate"], liquidation_mode=case["mode"])
    rows = run_rolling(
        "S",
        COLLAT,
        broker=broker,
        pledge_rates=[0.0, 0.3, 0.6, 0.7],
        holding_periods_years=[1, 2],
    )
    assert len(rows) == case["n"]
    assert sum(r["liquidated"] for r in rows) == case["n_liquidated"]
    assert sum(r["term_expired_forced_sale"] for r in rows) == case["n_term_sale"]
    assert sum(r["n_margin_calls"] for r in rows) == case["n_calls"]
    assert sum(r["final_equity"] for r in rows) == pytest.approx(
        case["sum_final_equity"], **APPROX
    )
    assert sum(r["final_equity_unlevered"] for r in rows) == pytest.approx(
        case["sum_unlevered"], **APPROX
    )
    assert sum(r["total_interest_paid"] for r in rows) == pytest.approx(
        case["sum_interest"], **APPROX
    )
    assert max(r["max_drawdown_equity"] for r in rows) == pytest.approx(
        case["max_dd"], **APPROX
    )
    for pos, key in ((0, "first"), (len(rows) // 2, "mid"), (-1, "last")):
        assert_row_equal(rows[pos], case[key])


def test_relative_multiples_and_summary_stats_match_reference():
    ms = relative_multiples(COLLAT, 0.4, 2, broker=BrokerProfile(annual_rate=0.05))
    assert ms == pytest.approx(REF["relative_multiples_0.05_0.4_2"], **APPROX)
    assert median(ms) == pytest.approx(REF["median_winrate"][0], **APPROX)
    assert win_rate(ms) == pytest.approx(REF["median_winrate"][1], **APPROX)


def test_breakeven_solver_matches_reference():
    assert solve_breakeven_rate(COLLAT, 0.4, 2, "median") == pytest.approx(
        REF["breakeven_median"], abs=1e-12
    )
    assert solve_breakeven_rate(COLLAT, 0.4, 2, "winrate") == pytest.approx(
        REF["breakeven_winrate"], abs=1e-12
    )
    scan = breakeven_rate_scan(COLLAT, 0.4, 2, rates=[0.02, 0.05, 0.08])
    assert scan == pytest.approx([tuple(r) for r in REF["scan"]], **APPROX)


@pytest.mark.parametrize(
    "key,pledge_rate,buffer,mode",
    [
        ("calls_nobuf", 0.7, 0.0, "all"),
        ("bigbuf", 0.7, 600_000.0, "all"),
        ("mild", 0.3, 0.0, "partial"),
    ],
)
def test_household_overlay_rolling_matrix_matches_reference(
    key, pledge_rate, buffer, mode
):
    ref = REF["overlay"][key]
    results = run_overlay_matrix_cell(
        COLLAT,
        LOAN,
        pledge_rate,
        "Y",
        2,
        broker=BrokerProfile(annual_rate=0.0385, liquidation_mode=mode),
        initial_buffer=buffer,
        profile=PROFILE,
    )
    assert len(results) == ref["n"]
    assert sum(r.final_net_wealth for r in results) == pytest.approx(
        ref["sum_final_wealth"], **APPROX
    )
    for new, expected in zip(
        (results[0], results[len(results) // 2], results[-1]), ref["results"]
    ):
        assert_row_equal(dataclasses.asdict(new), expected)
    stats = forced_disposal_stats(results)
    assert stats["n_forced_disposal"] == ref["stats"]["n_forced_disposal"]
    assert stats["forced_disposal_prob"] == pytest.approx(
        ref["stats"]["forced_disposal_prob"], **APPROX
    )
    assert {str(k): v for k, v in stats["by_onset_year"].items()} == ref["stats"][
        "by_onset_year"
    ]
    assert stats["worst_year"] == ref["stats"]["worst_year"]


@pytest.mark.parametrize("key,pledge_rate", [("infinite_0.7", 0.7), ("zero_0.4", 0.4)])
def test_min_buffer_solver_matches_reference(key, pledge_rate):
    got = solve_min_buffer(
        COLLAT,
        LOAN,
        pledge_rate,
        "Y",
        2,
        broker=BrokerProfile(annual_rate=0.0385),
        profile=PROFILE,
        buf_hi=6_000_000.0,
        tol=20_000.0,
    )
    ref = REF["min_buffer"][key]
    assert got["finite"] == ref["finite"]
    assert got["min_buffer"] == ref["min_buffer"]
    if ref["finite"]:
        assert got["stats_at_zero"]["n_forced_disposal"] == 0
    else:
        assert got["residual_prob_at_hi"] == pytest.approx(ref["residual_prob_at_hi"])
        assert got["worst_year"] == ref["worst_year"]
