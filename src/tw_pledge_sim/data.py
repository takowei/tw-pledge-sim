"""TWSE public-data access (prices, ex-dividend tables) with an on-disk cache.

Raw responses are cached under ``data_dir()`` (default ``./data``, override with the
``TW_PLEDGE_SIM_DATA`` environment variable).  The cache is git-ignored: re-fetch
instead of committing exchange data.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

from .adjust import build_split_adjusted_series

# First month the rolling-start scan can use (the earliest ETF listing month).
START_YEAR, START_MONTH = 2003, 6

USER_AGENT = "Mozilla/5.0 (research; tw-pledge-sim)"


def data_dir() -> Path:
    path = Path(os.environ.get("TW_PLEDGE_SIM_DATA", "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


class StockInputs(NamedTuple):
    """Everything the engine needs for one ticker (unpackable like a tuple)."""

    dates: list[str]
    prices: dict[str, float]  # split-adjusted closes
    ex_dates: set[str]
    ex_values: dict[str, float]  # per-share ex-rights value, NOT split-adjusted
    split_events: list[dict]


def roc_to_iso(roc_label: str) -> str:
    """民國年日期字串轉西元 ISO。

    支援 TWSE 兩種常見格式：STOCK_DAY_AVG/MFI94U 用 ' 92/06/30'（斜線分隔），
    TWT49U 用 '113年01月17日'（中文年月日）——2026-09-08 抽查發現後者原本沒被處理，
    導致 `roc_to_iso` 對每一筆除權息資料都拋 ValueError，被呼叫端的 except 靜默吞掉，
    結果是除息事件抽取永遠是空集合，第 21 條的凹陷邏輯形同沒被餵過任何真實資料。
    """
    label = roc_label.strip()
    if "年" in label:
        parts = re.split(r"[年月日]", label)
        parts = [p for p in parts if p != ""]
    else:
        parts = label.split("/")
    if len(parts) != 3:
        raise ValueError(f"unrecognized ROC date label: {roc_label!r}")
    y = int(parts[0]) + 1911
    m = int(parts[1])
    d = int(parts[2])
    return f"{y:04d}-{m:02d}-{d:02d}"


def http_get_json(url: str, retries: int = 8, timeout: int = 20) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            last_err = e
            time.sleep(min(6.0 * (attempt + 1), 60.0))
    raise RuntimeError(
        f"http_get_json failed after {retries} attempts: {url}"
    ) from last_err


def _month_range(start_y: int, start_m: int, end_y: int, end_m: int):
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def _price_cache_path(stock_no: str) -> Path:
    return data_dir() / f"prices_raw_{stock_no}.json"


def fetch_prices_raw(
    stock_no: str, end_y: int, end_m: int, sleep_s: float = 1.2
) -> dict:
    """逐月抓 STOCK_DAY_AVG，落地存檔，可斷點續抓（跳過已快取月份）。"""
    cache_path = _price_cache_path(stock_no)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    else:
        cache = {"stock_no": stock_no, "series": {}, "no_data_months": []}

    cached_months = {diso[:7] for diso in cache["series"]}
    for y, m in _month_range(START_YEAR, START_MONTH, end_y, end_m):
        ym = f"{y:04d}-{m:02d}"
        if ym in cached_months or ym in cache["no_data_months"]:
            continue
        url = (
            "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_AVG"
            f"?date={y:04d}{m:02d}01&stockNo={stock_no}&response=json"
        )
        d = http_get_json(url)
        if d.get("stat") != "OK":
            cache["no_data_months"].append(ym)
        else:
            got_any = False
            for row in d.get("data", []):
                label = row[0].strip()
                if label.startswith("月平均"):
                    continue
                close_str = row[1].replace(",", "").strip()
                try:
                    close = float(close_str)
                except ValueError:
                    continue
                diso = roc_to_iso(label)
                cache["series"][diso] = close
                got_any = True
            if not got_any:
                cache["no_data_months"].append(ym)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        time.sleep(sleep_s)
    return cache


def _dividend_cache_path(year: int) -> Path:
    return data_dir() / f"dividends_{year}.json"


def fetch_dividends_year(year: int, sleep_s: float = 0.3) -> dict:
    """抓某一年 TWT49U（除權息計算結果表），落地存檔（含全市場當年資料，之後再過濾標的）。"""
    cache_path = _dividend_cache_path(year)
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    url = (
        "https://www.twse.com.tw/rwd/zh/exRight/TWT49U"
        f"?startDate={year}0101&endDate={year}1231&response=json"
    )
    d = http_get_json(url)
    cache_path.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    time.sleep(sleep_s)
    return d


def extract_dividend_events(
    stock_no: str, start_year: int, end_year: int
) -> list[dict]:
    """從落地的 TWT49U 年度快取抽出某標的的除權息事件：[{date_iso, ex_rights_value}]。"""
    events = []
    for year in range(start_year, end_year + 1):
        cache_path = _dividend_cache_path(year)
        if not cache_path.exists():
            continue
        d = json.loads(cache_path.read_text())
        if d.get("stat") != "OK":
            continue
        for row in d.get("data", []):
            date_label, code = row[0], row[1]
            if code != stock_no:
                continue
            try:
                diso = roc_to_iso(date_label)
                value = float(str(row[5]).replace(",", "").strip())
            except (ValueError, IndexError):
                continue
            events.append({"date": diso, "ex_rights_value": value})
    return sorted(events, key=lambda e: e["date"])


def load_adjusted_price_series(stock_no: str) -> tuple[dict[str, float], list[dict]]:
    cache_path = _price_cache_path(stock_no)
    if not cache_path.exists():
        raise FileNotFoundError(f"no cached price data for {stock_no}: {cache_path}")
    cache = json.loads(cache_path.read_text())
    return build_split_adjusted_series(cache["series"])


def stock_inputs_from_series(
    adjusted: dict[str, float], split_events: list[dict], ex_events: list[dict]
) -> StockInputs:
    """Assemble engine inputs from an adjusted series plus extracted ex-dividend events."""
    dates = sorted(adjusted.keys())
    ex_dates = {e["date"] for e in ex_events if e["date"] in adjusted}
    ex_values = {
        e["date"]: e["ex_rights_value"] for e in ex_events if e["date"] in adjusted
    }
    return StockInputs(dates, adjusted, ex_dates, ex_values, split_events)


@lru_cache(maxsize=None)
def load_stock_inputs(stock_no: str, last_year: int) -> StockInputs:
    """Load cached TWSE data for ``stock_no`` through ``last_year``."""
    adjusted, split_events = load_adjusted_price_series(stock_no)
    if not adjusted:
        return StockInputs([], adjusted, set(), {}, split_events)
    first_year = int(min(adjusted)[:4])
    ex_events = extract_dividend_events(stock_no, first_year, last_year)
    return stock_inputs_from_series(adjusted, split_events, ex_events)


def fetch_all(stocks: list[str], end_y: int, end_m: int, first_div_year: int) -> None:
    """Download prices for each ticker and ex-dividend tables for each year (network)."""
    for stock_no in stocks:
        fetch_prices_raw(stock_no, end_y, end_m)
    for year in range(first_div_year, end_y + 1):
        fetch_dividends_year(year)
