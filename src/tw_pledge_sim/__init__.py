"""tw-pledge-sim: real-history simulator for Taiwan securities-backed loans."""

from .broker import GENERIC, BrokerProfile
from .data import StockInputs
from .engine import PledgeSimulator, SimResult
from .household import (
    HouseholdProfile,
    forced_disposal_stats,
    run_household_overlay,
    run_overlay_matrix_cell,
    solve_min_buffer,
)
from .rolling import (
    breakeven_rate_scan,
    relative_multiples,
    run_rolling,
    solve_breakeven_rate,
)

__all__ = [
    "GENERIC",
    "BrokerProfile",
    "HouseholdProfile",
    "PledgeSimulator",
    "SimResult",
    "StockInputs",
    "breakeven_rate_scan",
    "forced_disposal_stats",
    "relative_multiples",
    "run_household_overlay",
    "run_overlay_matrix_cell",
    "run_rolling",
    "solve_breakeven_rate",
    "solve_min_buffer",
]
