"""Deterministic synthetic market data for demos and offline tests (no exchange data).

A seeded random walk over weekdays with a built-in crash, one annual ex-dividend
date, and one 1:4 split, so that every code path (calls, disposal, split and
dividend adjustment) is exercised offline.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

from .data import StockInputs
from .adjust import build_split_adjusted_series
from .data import stock_inputs_from_series

SPLIT_DATE = "2010-06-03"
SPLIT_RATIO = 4.0


def weekdays(start: str, end: str) -> list[str]:
    cur, stop = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while cur <= stop:
        if cur.weekday() < 5:
            out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def raw_series(seed: int, with_split: bool = True) -> tuple[dict, list[dict]]:
    """Raw (split-unadjusted) closes and raw per-share ex-dividend events."""
    rng = random.Random(seed)
    days = weekdays("2008-01-02", "2012-12-31")
    if with_split:  # a trading gap around the split date, as in a real halt
        days = [d for d in days if not ("2010-05-31" <= d <= "2010-06-02")]
    prices, level = {}, 100.0
    for d in days:
        drift = -0.003 if "2008-09-01" <= d <= "2008-12-31" else 0.0008
        level *= math.exp(drift + 0.009 * rng.gauss(0.0, 1.0))
        prices[d] = level
    events = []
    for year in range(2008, 2013):
        ex = next(d for d in days if d >= f"{year}-07-20")
        events.append({"date": ex, "ex_rights_value": 3.0})
    if with_split:
        for d in prices:
            if d < SPLIT_DATE:
                prices[d] *= SPLIT_RATIO
        for e in events:
            if e["date"] < SPLIT_DATE:
                e["ex_rights_value"] *= SPLIT_RATIO
    return prices, events


def make_inputs(seed: int) -> StockInputs:
    raw, events = raw_series(seed)
    adjusted, split_events = build_split_adjusted_series(raw)
    return stock_inputs_from_series(adjusted, split_events, events)
