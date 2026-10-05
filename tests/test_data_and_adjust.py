"""Data-layer tests: ROC dates, split detection, split/dividend adjustment."""

from __future__ import annotations


import pytest

from tw_pledge_sim import adjust, data


def _make_flat_dates(n: int, start="2010-01-04"):
    """產生 n 個「假裝是營業日」的連續日期字串（用於單元測試，非真實交易日曆）。"""
    from datetime import date, timedelta

    d0 = date.fromisoformat(start)
    dates = []
    cur = d0
    while len(dates) < n:
        if cur.weekday() < 5:  # 跳過六日，近似營業日
            dates.append(cur.isoformat())
        cur += timedelta(days=1)
    return dates


# ---------------------------------------------------------------------------
# Phase A：資料層單元測試
# ---------------------------------------------------------------------------


def test_roc_to_iso():
    assert data.roc_to_iso(" 92/06/30") == "2003-06-30"
    assert data.roc_to_iso("114/06/10") == "2025-06-10"
    assert data.roc_to_iso("115/09/08") == "2026-09-08"


def test_roc_to_iso_handles_twt49u_chinese_date_format():
    """迴歸測試：TWT49U 用「113年01月17日」格式，不是斜線格式。

    2026-09-08 發現：`roc_to_iso` 原本只認斜線格式，遇到這種格式會拋 ValueError，
    被 `extract_dividend_events` 的 except 靜默吞掉——導致除息事件永遠抽取為空集合，
    第 21 條的除息前 6 個營業日凹陷邏輯形同沒被餵過任何真實資料（不是單位換算錯，
    是資料根本沒進來）。
    """
    assert data.roc_to_iso("113年01月17日") == "2024-01-17"
    assert data.roc_to_iso("113年07月16日") == "2024-07-16"


def test_split_detection_0050_2025_case():
    """規格書已驗證的真實案例：114/06/10 188.65 -> 114/06/18 47.57（1:4 分割）。"""
    series = {
        "2025-06-09": 187.00,
        "2025-06-10": 188.65,
        "2025-06-18": 47.57,
        "2025-06-19": 47.80,
    }
    events = adjust.detect_splits(series)
    split_ev = [e for e in events if e["date"] == "2025-06-18"]
    assert len(split_ev) == 1
    ev = split_ev[0]
    assert ev["is_split"] is True
    assert ev["matched_ratio"] == 4
    assert ev["trading_gap_days"] == 8  # 06/10 -> 06/18 跨越停止交易區間


def test_split_adjustment_makes_series_continuous():
    series = {
        "2025-06-09": 187.00,
        "2025-06-10": 188.65,
        "2025-06-18": 47.57,
        "2025-06-19": 47.80,
    }
    adjusted, events = adjust.build_split_adjusted_series(series)
    # 分割前的價格應回溯 ÷4，使序列不再出現假崩盤
    assert adjusted["2025-06-10"] == pytest.approx(188.65 / 4, rel=1e-6)
    assert adjusted["2025-06-18"] == pytest.approx(47.57)
    pct_change = adjusted["2025-06-18"] / adjusted["2025-06-10"] - 1
    assert abs(pct_change) < 0.10  # 調整後不再是 -75% 假崩盤
    assert any(e["is_split"] for e in events)


def test_split_detection_does_not_flag_ordinary_moves():
    """非分割的正常波動（<40%）不應被標記。"""
    series = {"2020-01-02": 100.0, "2020-01-03": 108.0, "2020-01-06": 95.0}
    events = adjust.detect_splits(series)
    assert all(not e["is_split"] for e in events)


def test_dividend_events_are_split_adjusted_to_match_price_series():
    """迴歸測試（2026-09-08 抽查發現）：除息值須與分割回溯調整後的價格序列同單位。

    `extract_dividend_events` 回傳的是分割前原始股數基礎的息值；價格序列對分割日之前
    全部 ÷4（見 0050 2025-06-18 的 1:4 分割）。若息值不同步調整，用調整後股數乘上
    未調整的息值會高估分割前那段配息現金 4 倍。
    """
    split_events = [
        {
            "date": "2025-06-18",
            "matched_ratio": 4,
            "is_split": True,
        }
    ]
    events = [
        {"date": "2020-01-17", "ex_rights_value": 2.2},  # 分割前，應被 ÷4
        {"date": "2026-01-20", "ex_rights_value": 1.5},  # 分割後，不動
    ]
    adjusted = adjust.adjust_dividend_events_for_splits(events, split_events)
    by_date = {e["date"]: e["ex_rights_value"] for e in adjusted}
    assert by_date["2020-01-17"] == pytest.approx(2.2 / 4)
    assert by_date["2026-01-20"] == pytest.approx(1.5)


def test_split_detection_ignores_jump_without_trading_gap():
    """跳空幅度夠大但沒有交易中斷（例如真實暴跌）不應被誤判為分割。"""
    series = {"2020-01-02": 100.0, "2020-01-03": 55.0}  # 相鄰交易日，無停牌缺口
    events = adjust.detect_splits(series)
    assert len(events) == 1
    assert events[0]["is_split"] is False
    assert events[0]["trading_gap_days"] == 1


# ---------------------------------------------------------------------------
# Phase B：十條規則單元測試（用合成價格序列，數字可控）
# ---------------------------------------------------------------------------
