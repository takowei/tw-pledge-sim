"""Demo: rolling-start pledge statistics and the break-even loan rate.

    python examples/demo.py            # offline, synthetic prices (no network)
    python examples/demo.py --live     # real TWSE prices; downloads and caches data

All household numbers below are made up: 1,000,000 of capital, 50,000 monthly income.
The synthetic path is a seeded random walk with one crash, so the printed numbers
illustrate the method only and say nothing about the market.
"""

from __future__ import annotations

import argparse
from datetime import date

from tw_pledge_sim import (
    GENERIC,
    BrokerProfile,
    HouseholdProfile,
    StockInputs,
    breakeven_rate_scan,
    forced_disposal_stats,
    run_overlay_matrix_cell,
    run_rolling,
    solve_breakeven_rate,
)
from tw_pledge_sim.data import fetch_all, load_stock_inputs
from tw_pledge_sim.rolling import median, win_rate
from tw_pledge_sim.synthetic import make_inputs

TICKER = "0050"
HOLDING_YEARS = 2
PLEDGE_RATES = [0.0, 0.2, 0.4, 0.6, 0.7]


def live_inputs() -> StockInputs:
    today = date.today()
    fetch_all([TICKER], today.year, today.month, first_div_year=2008)
    load_stock_inputs.cache_clear()
    return load_stock_inputs(TICKER, today.year)


def rolling_table(inputs: StockInputs, broker: BrokerProfile) -> list[str]:
    rows = run_rolling(
        "demo",
        inputs,
        broker=broker,
        pledge_rates=PLEDGE_RATES,
        holding_periods_years=[HOLDING_YEARS],
    )
    lines = [
        f"{'pledge':>7} {'starts':>6} {'forced sale':>12} {'median x':>9} {'win rate':>9}"
    ]
    for pr in PLEDGE_RATES:
        sel = [r for r in rows if r["pledge_rate"] == pr]
        forced = sum(r["liquidated"] or r["term_expired_forced_sale"] for r in sel)
        mult = [
            r["final_equity"] / r["final_equity_unlevered"]
            for r in sel
            if r["final_equity_unlevered"] > 0
        ]
        lines.append(
            f"{pr:>7.0%} {len(sel):>6} {forced / len(sel):>12.1%}"
            f" {median(mult):>9.3f} {win_rate(mult):>9.1%}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="use real TWSE data")
    args = parser.parse_args()
    inputs = live_inputs() if args.live else make_inputs(1)
    loan_inputs = inputs if args.live else make_inputs(2)
    broker = BrokerProfile(name="generic", annual_rate=0.0385)

    source = "TWSE " + TICKER if args.live else "synthetic"
    print(
        f"Source: {source}, holding {HOLDING_YEARS} years, loan rate {broker.annual_rate:.2%}"
    )
    print("\nRolling-start results (one start per month):")
    print("\n".join(rolling_table(inputs, broker)))

    rate = solve_breakeven_rate(inputs, 0.4, HOLDING_YEARS, "median", broker=GENERIC)
    print(f"\nBreak-even loan rate at 40% pledge (median multiple = 1): {rate:.2%}")
    scan = breakeven_rate_scan(inputs, 0.4, HOLDING_YEARS, rates=[0.02, 0.05, 0.08])
    for r, med, wr in scan:
        print(f"  rate {r:.0%}: median multiple {med:.3f}, win rate {wr:.1%}")

    profile = HouseholdProfile()
    results = run_overlay_matrix_cell(
        inputs, loan_inputs, 0.7, "Y", HOLDING_YEARS, broker=broker, profile=profile
    )
    stats = forced_disposal_stats(results)
    print(
        f"\nHousehold overlay at 70% pledge, no cash buffer: forced disposal in "
        f"{stats['n_forced_disposal']} of {stats['n_starts']} starts "
        f"({stats['forced_disposal_prob']:.1%}); by onset year {stats['by_onset_year']}"
    )


if __name__ == "__main__":
    main()
