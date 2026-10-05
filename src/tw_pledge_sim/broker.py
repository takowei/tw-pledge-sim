"""Broker-specific (medium-confidence) behaviour, kept apart from statute.

The regulation fixes thresholds and timetables; how a broker prices the loan and
how it disposes of collateral is practice and varies.  ``GENERIC`` uses neutral
defaults and names no institution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrokerProfile:
    name: str = "generic"
    annual_rate: float = 0.035  # loan rate, simple daily interest
    # 'all': disposal sells all collateral (literal reading of Art. 25).
    # 'partial': sell only until the loan is repaid (common practice).
    liquidation_mode: str = "all"
    auto_cure: bool = True  # post cash/collateral to reach the cure target
    allow_rollover: bool = True  # the broker approves the 6-month extensions

    def simulator_kwargs(self) -> dict:
        return {
            "annual_rate": self.annual_rate,
            "liquidation_mode": self.liquidation_mode,
            "auto_cure": self.auto_cure,
            "allow_rollover": self.allow_rollover,
        }


GENERIC = BrokerProfile()
