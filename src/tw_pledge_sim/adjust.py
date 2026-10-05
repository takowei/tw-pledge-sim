"""Split detection and split/dividend adjustment of raw close-price series."""

from __future__ import annotations

from datetime import date
from typing import Optional

SPLIT_JUMP_THRESHOLD = 0.40  # 單日跳空超過 ±40% 視為分割候選
SPLIT_RATIO_CANDIDATES = [2, 3, 4, 5, 6, 8, 10]
SPLIT_RATIO_TOLERANCE = 0.05  # 與整數倍的容忍誤差


def detect_splits(series: dict[str, float]) -> list[dict]:
    """在收盤價序列中偵測疑似股票分割（單日跳空 >40% 且與交易中斷相鄰）。

    回傳每筆偵測到的事件：{date, prev_date, prev_close, close, raw_ratio,
    matched_ratio, trading_gap_days, is_split}
    `is_split` 只在漲跌比值夠接近某個整數倍且中間有交易中斷時才判 True；
    其餘跳空只標記不調整（可能是除息或真實暴漲暴跌，交給人工覆核）。
    """
    dates = sorted(series.keys())
    events = []
    for i in range(1, len(dates)):
        prev_d, cur_d = dates[i - 1], dates[i]
        prev_close, cur_close = series[prev_d], series[cur_d]
        if prev_close <= 0:
            continue
        pct_change = cur_close / prev_close - 1.0
        if abs(pct_change) < SPLIT_JUMP_THRESHOLD:
            continue
        gap_days = (date.fromisoformat(cur_d) - date.fromisoformat(prev_d)).days
        raw_ratio = prev_close / cur_close if cur_close > 0 else float("inf")
        matched_ratio = None
        for cand in SPLIT_RATIO_CANDIDATES:
            if abs(raw_ratio - cand) / cand <= SPLIT_RATIO_TOLERANCE:
                matched_ratio = cand
                break
            if abs(raw_ratio - (1.0 / cand)) / (1.0 / cand) <= SPLIT_RATIO_TOLERANCE:
                matched_ratio = 1.0 / cand
                break
        is_split = matched_ratio is not None and gap_days >= 2
        events.append(
            {
                "date": cur_d,
                "prev_date": prev_d,
                "prev_close": prev_close,
                "close": cur_close,
                "raw_ratio": raw_ratio,
                "matched_ratio": matched_ratio,
                "trading_gap_days": gap_days,
                "is_split": is_split,
            }
        )
    return events


def build_split_adjusted_series(
    series: dict[str, float],
) -> tuple[dict[str, float], list[dict]]:
    """回傳（分割回溯調整後序列, 偵測到的事件列表）。

    只對 `is_split=True` 的事件做回溯調整：分割日之前所有價格 ÷ matched_ratio，
    讓序列連續（不出現假崩盤）。多個分割事件依時間序疊加調整。
    """
    events = detect_splits(series)
    adjusted = dict(series)
    split_events = [e for e in events if e["is_split"]]
    # 由新到舊依序調整，確保多次分割時因子正確疊乘
    for ev in sorted(split_events, key=lambda e: e["date"], reverse=True):
        ratio = ev["matched_ratio"]
        cutoff = ev["date"]
        for d in list(adjusted.keys()):
            if d < cutoff:
                adjusted[d] = adjusted[d] / ratio
    return adjusted, events


def adjust_dividend_events_for_splits(
    events: list[dict], split_events: list[dict]
) -> list[dict]:
    """把 TWT49U 原始「權值+息值」換算成與分割回溯調整後價格序列同單位。

    2026-09-08 抽查發現：`extract_dividend_events` 回傳的 `ex_rights_value` 是
    TWSE 原始股數基礎（分割前），但 `load_adjusted_price_series` 產生的價格序列
    對分割日之前全部 ÷matched_ratio（見 `build_split_adjusted_series`）。
    若直接把未調整的息值拿去乘「調整後股數」（分割前的股數會被放大 matched_ratio 倍），
    會系統性高估分割前那段期間的配息現金 matched_ratio 倍（ETF 1:4 split 案例約 4 倍）。
    修正：對分割日之前的除息事件，息值同樣 ÷matched_ratio，與價格序列同步調整。
    """
    adjusted = [dict(e) for e in events]
    real_splits = [e for e in split_events if e["is_split"]]
    for ev in sorted(real_splits, key=lambda e: e["date"], reverse=True):
        ratio = ev["matched_ratio"]
        cutoff = ev["date"]
        for e in adjusted:
            if e["date"] < cutoff:
                e["ex_rights_value"] = e["ex_rights_value"] / ratio
    return adjusted


def split_adjust_ex_values(
    ex_value: dict[str, float], split_events: Optional[list[dict]]
) -> dict[str, float]:
    """把分割日之前的每股息值 ÷matched_ratio，與價格序列的分割回溯調整同步（見 §8.2）。"""
    out = dict(ex_value)
    if not split_events:
        return out
    real = [e for e in split_events if e.get("is_split")]
    for ev in sorted(real, key=lambda e: e["date"], reverse=True):
        ratio = ev["matched_ratio"]
        cutoff = ev["date"]
        for dt in list(out.keys()):
            if dt < cutoff:
                out[dt] = out[dt] / ratio
    return out
