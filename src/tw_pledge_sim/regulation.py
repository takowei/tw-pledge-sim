"""Statutory parameters of Taiwan's unrestricted-purpose securities-backed loans.

Source: the Securities Finance Enterprises' "unrestricted-purpose loan" operating
rules (article numbers below refer to that regulation, version of 2026-08).
Everything here is high-confidence regulation: brokers may be stricter but not
more lenient.  Broker-specific behaviour lives in :mod:`tw_pledge_sim.broker`.
"""

from __future__ import annotations

HAIRCUT = (
    0.60  # Art. 16: loan value of an eligible listed security is 60% of market value
)
MARGIN_CALL_RATIO = 1.30  # Art. 20(3): maintenance ratio below this triggers a call
# Art. 20: cure target is 1 / 60% = 166.67%.  Keep the exact reciprocal; the rounded
# literal 1.6667 once misclassified a flat-price day-0 ratio through float error.
CURE_TARGET_RATIO = 1.0 / HAIRCUT
CURE_WINDOW_DAYS = 2  # Art. 20(3): cure within 2 business days of the notice
LIQUIDATION_START_BUSINESS_DAY = 3  # Art. 25(1): disposal from the 3rd business day
EX_DIVIDEND_LOOKBACK_DAYS = 6  # Art. 21: ex-date haircut applies 6 business days before
DEFAULT_PENALTY_RATE_MULT = 0.10  # Art. 26(2): penalty = 10% of the loan rate
TERM_MONTHS = 6  # Art. 4: a loan term is at most 6 months
MAX_ROLLOVERS = 2  # Art. 4: two extensions, 6 + 6 + 6 = 18 months hard cap

# Taxes on dividend cash (not part of the loan regulation, but part of the cash flow).
NHI_SUPP_RATE = 0.0211  # supplementary national-health-insurance premium rate
NHI_SUPP_SINGLE_THRESHOLD = 20_000.0  # charged per payment of at least this amount
NHI_SUPP_SINGLE_CAP = 10_000_000.0  # per-payment cap on the premium base
