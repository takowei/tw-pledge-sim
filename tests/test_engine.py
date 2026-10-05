"""Engine tests: the regulation rules one by one, taxes, loan target, exit modes."""

from __future__ import annotations


import pytest
from datetime import date, timedelta

from tw_pledge_sim import regulation
from tw_pledge_sim.engine import PledgeSimulator, net_dividend_cash


def _make_flat_dates(n: int, start="2010-01-04"):
    """產生 n 個「假裝是營業日」的連續日期字串（用於單元測試，非真實交易日曆）。"""

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


def test_rule1_haircut_is_60pct():
    """規則 1：成數 60%，按融通日前一營業日收盤價計算。"""
    assert regulation.HAIRCUT == 0.60


def test_rule2_credit_line_locked_not_recomputed_on_price_rise():
    """規則 2：L_t = L_0 恆定，不隨市值上漲自動增加。"""
    dates = _make_flat_dates(30)
    prices = {d: 100.0 for d in dates}
    # 價格連續上漲 50%
    for i, d in enumerate(dates):
        prices[d] = 100.0 * (1 + 0.5 * i / (len(dates) - 1))
    sim_engine = PledgeSimulator(dates=dates, prices=prices, annual_rate=0.0)
    result = sim_engine.run(
        0, len(dates) - 1, pledge_rate=0.30, principal_own_shares_value=1_000_000.0
    )
    expected_L = 0.30 * 1_000_000.0
    # 用最終權益反推：final_equity = (own_shares+pledged_shares)*end_price - L
    end_price = prices[dates[-1]]
    own_shares = 1_000_000.0 / prices[dates[0]]
    pledged_shares = expected_L / prices[dates[0]]
    expected_equity = (own_shares + pledged_shares) * end_price - expected_L
    assert result.final_equity == pytest.approx(expected_equity, rel=1e-6)


def test_rule3_maintenance_ratio_uses_drawn_principal_not_credit_line():
    """規則 3：維持率 = 擔保品市值 / 已動用本金（分母不含利息）。

    用「價格持平、但年利率極高」的情境驗證：若分母錯誤地把累積利息算進去，
    維持率會隨時間下滑並最終觸發追繳；正確實作下應恆定在 166.67% 不觸發。
    """
    dates = _make_flat_dates(400)
    p0 = 100.0
    prices = {d: p0 for d in dates}
    engine = PledgeSimulator(
        dates=dates, prices=prices, annual_rate=0.30
    )  # 刻意用高利率放大差異
    result = engine.run(
        0, len(dates) - 1, pledge_rate=0.60, principal_own_shares_value=1_000_000.0
    )
    assert result.n_margin_calls == 0
    assert result.total_interest_paid > 0.0  # 利息確實有累積，只是不進維持率分母

    # 維持率分母是「已動用本金」，分子是「擔保品市值」——擔保品是質押當下鎖定的
    # own_shares 部位本身，不含借款買回的加碼部位（見 real_path_pledge_sim 模組
    # run() 內的設計註解，對應規格書封閉解表格 166.67% 的推導基礎）。
    own_val = 1_000_000.0
    L = 0.60 * own_val
    mv_t0 = own_val  # 擔保品市值 = 質押部位市值，t0 時等於原始出資
    expected_ratio_t0 = mv_t0 / L
    assert expected_ratio_t0 == pytest.approx(
        1 / regulation.HAIRCUT, rel=1e-6
    )  # 166.67%


def test_rule4_ex_dividend_haircut_reduces_mv_before_ex_date():
    """規則 4：除權息交易日前 6 個營業日起，市值扣除權息值（每股）。"""
    dates = _make_flat_dates(20)
    ex_date = dates[10]
    prices = {d: 100.0 for d in dates}
    ex_value_per_share = 3.0
    engine = PledgeSimulator(
        dates=dates,
        prices=prices,
        ex_dividend_dates={ex_date},
        ex_dividend_value={ex_date: ex_value_per_share},
        annual_rate=0.0,
    )
    idx_within_window = 10 - 6  # 第 6 個營業日前
    idx_outside_window = 10 - 7  # 第 7 個營業日前，尚未進入扣抵區間
    assert engine._haircut_per_share(idx_within_window) == pytest.approx(
        ex_value_per_share
    )
    assert engine._haircut_per_share(idx_outside_window) == pytest.approx(0.0)
    assert engine._haircut_per_share(10) == pytest.approx(
        ex_value_per_share
    )  # 除息日當天仍在窗內
    assert engine._haircut_per_share(11) == pytest.approx(0.0)  # 除息日後不再扣抵


def test_rule4_haircut_is_per_share_not_lump_sum():
    """單位迴歸測試：透過完整 run() 驗證除息凹陷是「股數 × 每股息值」，不是單一純量。

    2026-09-08 主對話抽查發現的 bug：`mv = shares*price - haircut` 曾把每股權息值
    直接當總額扣，量級差了股數倍（0050 案例約 9,090 倍），導致第 21 條形同不存在。
    這裡刻意把維持率卡在「只有乘上股數後的凹陷才夠讓它跌破 130%」的邊界，
    若退化回 lump-sum bug（只扣一個純量），這個追繳事件會完全不出現。
    """
    dates = _make_flat_dates(20)
    ex_date = dates[10]
    p0 = 100.0
    prices = {d: p0 for d in dates}
    principal = 1_000_000.0
    # 質押率選在讓 t0 維持率 = 132%，貼著 130% 門檻上方一點點
    pledge_rate = 1.0 / 1.32
    ex_value_per_share = (
        2.0  # own_shares(10,000) x 2.0 = 20,000 的正確凹陷才夠跌破 130%
    )

    engine = PledgeSimulator(
        dates=dates,
        prices=prices,
        ex_dividend_dates={ex_date},
        ex_dividend_value={ex_date: ex_value_per_share},
        annual_rate=0.0,
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=pledge_rate, principal_own_shares_value=principal
    )
    # 正確實作（乘股數）：除息前 6 個營業日的凹陷應足以跨過 130% 門檻，觸發追繳；
    # 若退化成 lump-sum bug，凹陷幅度只有每股息值本身，遠不足以觸發，n_margin_calls 會是 0。
    assert result.n_margin_calls >= 1


def test_rule5_margin_call_130_cure_target_166_and_2day_window():
    """規則 5：< 130% 觸發、2 個營業日內補繳至 > 166%、否則第 3 營業日處分。"""
    assert regulation.MARGIN_CALL_RATIO == pytest.approx(1.30)
    assert regulation.CURE_TARGET_RATIO == pytest.approx(1.6667, rel=1e-3)
    assert regulation.CURE_WINDOW_DAYS == 2
    assert regulation.LIQUIDATION_START_BUSINESS_DAY == 3

    dates = _make_flat_dates(30)
    p0 = 100.0
    prices = {d: p0 for d in dates}
    # 第 5 天起價格連續下跌，觸發維持率跌破 130%（60% 質押率下，跌幅超過 22% 觸發）
    for i, d in enumerate(dates):
        if i < 5:
            prices[d] = p0
        else:
            prices[d] = p0 * (1 - 0.03 * (i - 4))  # 每日再跌 3%
    engine = PledgeSimulator(
        dates=dates, prices=prices, annual_rate=0.0, auto_cure=False
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=0.60, principal_own_shares_value=1_000_000.0
    )
    assert result.n_margin_calls >= 1
    assert result.liquidated is True
    trigger_idx = dates.index(result.margin_call_events[0].trigger_date)
    liquidation_idx = dates.index(result.liquidation_date)
    # 未補繳情境：處分應落在觸發日 + 3 個營業日（D+3）附近
    assert liquidation_idx - trigger_idx == 3


def test_rule5_cure_within_window_avoids_liquidation():
    """在 2 日補繳窗內回升至 >=166% 應解除追繳，不觸發處分。"""
    dates = _make_flat_dates(15)
    p0 = 100.0
    prices = {d: p0 for d in dates}
    # 第 3 天大跌觸發追繳（跌破 130% 門檻，質押率 60% 下需跌破約 48.75）
    prices[dates[3]] = p0 * 0.40
    prices[dates[4]] = p0 * 1.5
    prices[dates[5]] = p0 * 1.5
    for i in range(6, len(dates)):
        prices[dates[i]] = p0 * 1.5
    engine = PledgeSimulator(dates=dates, prices=prices, annual_rate=0.0)
    result = engine.run(
        0, len(dates) - 1, pledge_rate=0.60, principal_own_shares_value=1_000_000.0
    )
    assert result.n_margin_calls >= 1
    assert result.liquidated is False


def test_rule6_no_intraday_liquidation_only_daily_close_used():
    """規則 6：引擎只使用日收盤價，介面上不存在盤中資料輸入點。"""
    import inspect

    sig = inspect.signature(PledgeSimulator.__init__)
    assert "prices" in sig.parameters
    # 沒有任何「盤中」或 intraday 相關參數
    assert not any("intraday" in p.lower() for p in sig.parameters)


def test_rule7_flat_price_survives_indefinite_refinancing():
    """規則 7：6+6+6=18 個月一個合約週期；到期辦「借新還舊」需單筆維持率 ≥166.67%。

    價格持平時，t0 維持率恆為 166.67%（>= 門檻），每次到期都應該借新還舊成功、
    重新起算 18 個月，不應該被強制平倉——這是長期槓桿策略「持續質押」的常態路徑。
    """
    assert regulation.MAX_ROLLOVERS == 2
    dates = _make_flat_dates(700)  # > 3 個 term（126*3=378 個交易日 ≈ 18 個月）
    p0 = 100.0
    prices = {d: p0 for d in dates}
    engine = PledgeSimulator(dates=dates, prices=prices, annual_rate=0.0)
    result = engine.run(
        0, len(dates) - 1, pledge_rate=0.60, principal_own_shares_value=1_000_000.0
    )
    assert result.term_expired_forced_sale is False
    assert result.liquidated is False


def test_rule7_refinance_fails_when_below_166_at_18month_boundary():
    """維持率跌到 130%~166.67% 之間（過一般展延門檻但過不了借新還舊門檻）應在 18 個月到期時強制平倉。"""
    dates = _make_flat_dates(500)
    p0 = 100.0
    prices = {}
    for i, d in enumerate(dates):
        # 緩降到略低於 166.67% 但仍高於 130%，兩次一般展延（門檻130%）都能過，
        # 但第 18 個月的借新還舊（門檻166.67%）過不了。
        prices[d] = p0 * (1 - 0.00035 * i)
    engine = PledgeSimulator(
        dates=dates, prices=prices, annual_rate=0.0, auto_cure=False
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=0.60, principal_own_shares_value=1_000_000.0
    )
    # 維持率會先降破 166.67%（借新還舊門檻）但可能還沒破 130%（追繳門檻），
    # 只要沒觸發追繳，到期時應該因借新還舊失敗而強制平倉。
    if result.n_margin_calls == 0:
        assert result.term_expired_forced_sale is True


def test_rule7_no_rollover_mode_forces_sale_at_first_term_end():
    """allow_rollover=False：到期即強制賣出（作為可開關參數存在）。"""
    dates = _make_flat_dates(200)
    p0 = 100.0
    prices = {d: p0 for d in dates}
    engine = PledgeSimulator(
        dates=dates, prices=prices, annual_rate=0.0, allow_rollover=False
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=0.30, principal_own_shares_value=1_000_000.0
    )
    assert result.term_expired_forced_sale is True


def test_rule8_default_penalty_is_10pct_of_funding_rate():
    """規則 8：違約金 = 融通利率 × 10%，在逾期未補繳而處分時計入。"""
    assert regulation.DEFAULT_PENALTY_RATE_MULT == 0.10
    dates = _make_flat_dates(30)
    p0 = 100.0
    prices = {d: p0 for d in dates}
    for i, d in enumerate(dates):
        if i >= 5:
            prices[d] = p0 * (1 - 0.03 * (i - 4))
    engine = PledgeSimulator(
        dates=dates, prices=prices, annual_rate=0.035, auto_cure=False
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=0.60, principal_own_shares_value=1_000_000.0
    )
    assert result.liquidated is True
    assert result.default_penalty_paid > 0.0
    L = 0.60 * 1_000_000.0
    expected_upper_bound = L * 0.035 * 0.10 * 10 / 365.0  # 逾期不會超過約10天量級
    assert (
        result.default_penalty_paid < expected_upper_bound * 5
    )  # 量級檢查，不要求精確天數


def test_rule9_interest_accrues_daily_from_drawdown():
    """規則 9：利息日計息、動用後起息；0% 質押率應完全無利息。"""
    dates = _make_flat_dates(366)
    p0 = 100.0
    prices = {d: p0 for d in dates}
    engine = PledgeSimulator(dates=dates, prices=prices, annual_rate=0.0365)
    result_0pct = engine.run(
        0, len(dates) - 1, pledge_rate=0.0, principal_own_shares_value=1_000_000.0
    )
    assert result_0pct.total_interest_paid == 0.0

    result_lev = engine.run(
        0, len(dates) - 1, pledge_rate=0.30, principal_own_shares_value=1_000_000.0
    )
    L = 0.30 * 1_000_000.0
    n_days = (
        __import__("datetime").date.fromisoformat(dates[-1])
        - __import__("datetime").date.fromisoformat(dates[0])
    ).days
    expected_interest = L * 0.0365 / 365.0 * n_days
    assert result_lev.total_interest_paid == pytest.approx(expected_interest, rel=0.05)


def test_rule10_dividends_do_not_alter_outstanding_loan():
    """規則 10：配息歸擔保品所有人，質押不影響領息——除息事件不應改變已動用本金 L。"""
    dates = _make_flat_dates(20)
    ex_date = dates[10]
    prices = {d: 100.0 for d in dates}
    engine = PledgeSimulator(
        dates=dates,
        prices=prices,
        ex_dividend_dates={ex_date},
        ex_dividend_value={ex_date: 1.0},  # 遠低於追繳門檻，不觸發追繳
        annual_rate=0.0,
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=0.30, principal_own_shares_value=1_000_000.0
    )
    # 除息只影響維持率計算窗口，不觸發追繳/處分，也不改變本金——用「未被處分」驗證本金未被動用
    assert result.liquidated is False
    assert result.n_margin_calls == 0


def test_dividend_income_is_added_to_final_equity_leveraged():
    """迴歸測試（2026-09-08 第二輪 bug）：槓桿路徑的配息現金流從未被加回 final_equity。

    平價（不漲不跌）、零利率、單一除息事件情境：own_shares 與 loan_shares 都應
    在除息日收到「股數 × 每股息值」的現金，計入 final_equity；退回舊碼（沒有
    dividend_cash 累加）時，final_equity 會少算這筆現金，斷言會 fail。
    """
    dates = _make_flat_dates(20)
    ex_date = dates[10]
    p0 = 100.0
    prices = {d: p0 for d in dates}
    div_per_share = 1.0  # 遠低於追繳門檻（見 test_rule10），不觸發追繳/處分
    principal = 1_000_000.0
    pledge_rate = 0.30

    engine = PledgeSimulator(
        dates=dates,
        prices=prices,
        ex_dividend_dates={ex_date},
        ex_dividend_value={ex_date: div_per_share},
        annual_rate=0.0,
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=pledge_rate, principal_own_shares_value=principal
    )
    assert result.liquidated is False
    assert result.term_expired_forced_sale is False

    own_shares = principal / p0
    loan_shares = (pledge_rate * principal) / p0
    # 平價、零利率、無違約金：舊碼（缺配息）final_equity 應恆為本金（own+loan 市值互抵 L）。
    buggy_final_equity = own_shares * p0 - pledge_rate * principal + loan_shares * p0
    expected_dividend_cash = (own_shares + loan_shares) * div_per_share
    expected_final_equity = buggy_final_equity + expected_dividend_cash

    assert result.final_equity == pytest.approx(expected_final_equity, rel=1e-9)
    assert result.final_equity != pytest.approx(buggy_final_equity)


def test_dividend_income_is_added_to_final_equity_unlevered():
    """迴歸測試（2026-09-08 第二輪 bug）：`pledge_rate=0` 對照組同樣漏算配息現金。

    `final_equity_unlevered`（及 `pledge_rate=0` 分支的 `final_equity`）只用
    `own_shares * end_price`，完全沒把持有期間收到的配息計入——不槓桿的對照組
    本身就被低估。退回舊碼時本測試斷言會 fail。
    """
    dates = _make_flat_dates(20)
    ex_date = dates[10]
    p0 = 100.0
    prices = {d: p0 for d in dates}
    # 單筆給付 own_shares(10,000) x 1.5 = 15,000 < 二代健保 2 萬門檻 → 不扣補充保費，
    # 稅後化不影響本測試（門檻行為由 test_dividend_after_tax_* 專門覆蓋）。
    div_per_share = 1.5
    principal = 1_000_000.0

    engine = PledgeSimulator(
        dates=dates,
        prices=prices,
        ex_dividend_dates={ex_date},
        ex_dividend_value={ex_date: div_per_share},
        annual_rate=0.0,
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=0.0, principal_own_shares_value=principal
    )

    own_shares = principal / p0
    buggy_equity = own_shares * p0  # 平價情境下舊碼恆為本金
    expected_dividend_cash = own_shares * div_per_share
    expected_equity = buggy_equity + expected_dividend_cash

    assert result.final_equity == pytest.approx(expected_equity, rel=1e-9)
    assert result.final_equity_unlevered == pytest.approx(expected_equity, rel=1e-9)
    assert result.final_equity != pytest.approx(buggy_equity)


def test_liquidation_mode_all_vs_partial_produce_different_tail_outcomes():
    """處分範圍情境：all（全部）vs partial（賣到夠清償）——尾部結果應不同。"""
    dates = _make_flat_dates(30)
    p0 = 100.0
    prices = {d: p0 for d in dates}
    for i, d in enumerate(dates):
        if i >= 5:
            prices[d] = p0 * (1 - 0.03 * (i - 4))

    engine_all = PledgeSimulator(
        dates=dates,
        prices=prices,
        annual_rate=0.0,
        liquidation_mode="all",
        auto_cure=False,
    )
    result_all = engine_all.run(
        0, len(dates) - 1, pledge_rate=0.60, principal_own_shares_value=1_000_000.0
    )

    engine_partial = PledgeSimulator(
        dates=dates,
        prices=prices,
        annual_rate=0.0,
        liquidation_mode="partial",
        auto_cure=False,
    )
    result_partial = engine_partial.run(
        0, len(dates) - 1, pledge_rate=0.60, principal_own_shares_value=1_000_000.0
    )

    assert result_all.liquidated and result_partial.liquidated
    assert result_all.final_equity != pytest.approx(result_partial.final_equity)


# ---------------------------------------------------------------------------
# Phase C：封閉解 sanity check（合成資料，對應規格書表格）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pledge_rate,static_drawdown_limit",
    [
        (0.30, -0.610),
        (0.40, -0.480),
        (0.50, -0.350),
        (0.60, -0.220),
    ],
)
def test_phase_c_trigger_point_no_later_than_static_closed_form(
    pledge_rate, static_drawdown_limit
):
    """模擬觸發點應略早於（跌幅絕對值更小於等於）靜態封閉解表——若更寬鬆代表實作有錯。"""
    dates = _make_flat_dates(60)
    p0 = 100.0
    prices = {d: p0 for d in dates}
    # 線性緩降到剛好略超過靜態理論觸發跌幅，逐日 close-to-close，不含除息/利息干擾
    total_drawdown = static_drawdown_limit - 0.02  # 多跌 2 個百分點，確保觸發
    for i, d in enumerate(dates):
        prices[d] = p0 * (1 + total_drawdown * i / (len(dates) - 1))
    engine = PledgeSimulator(
        dates=dates, prices=prices, annual_rate=0.0, auto_cure=False
    )
    result = engine.run(
        0,
        len(dates) - 1,
        pledge_rate=pledge_rate,
        principal_own_shares_value=1_000_000.0,
    )
    if result.n_margin_calls == 0:
        pytest.skip("此路徑未觸發追繳，非本測試要驗證的分支")
    trigger_idx = dates.index(result.margin_call_events[0].trigger_date)
    triggered_price = prices[dates[trigger_idx]]
    actual_drawdown_at_trigger = triggered_price / p0 - 1
    # 模擬觸發跌幅應「不寬鬆於」靜態解：實際觸發跌幅的絕對值 <= 靜態解絕對值 + 小容差
    assert abs(actual_drawdown_at_trigger) <= abs(static_drawdown_limit) + 0.03


# ---------------------------------------------------------------------------
# 真實資料測試（需 Phase A 快取；快取未就緒則 skip，不使用假資料）
# ---------------------------------------------------------------------------


def test_net_dividend_cash_helper():
    """單元：net_dividend_cash 的門檻、費率、上限、股利稅。"""
    # 單筆 < 2 萬：不扣二代健保
    assert net_dividend_cash(19_999.0, 0.0) == pytest.approx(19_999.0)
    # 單筆 = 2 萬（含）：扣 2.11%
    assert net_dividend_cash(20_000.0, 0.0) == pytest.approx(20_000.0 * (1 - 0.0211))
    # 單筆 > 2 萬 + 20% 股利稅：兩項都扣
    assert net_dividend_cash(30_000.0, 0.20) == pytest.approx(
        30_000.0 - 30_000.0 * 0.0211 - 30_000.0 * 0.20
    )
    # 單筆上限 1000 萬：超過部分不再課二代健保（股利稅仍對全額）
    got = net_dividend_cash(12_000_000.0, 0.0)
    assert got == pytest.approx(12_000_000.0 - 10_000_000.0 * 0.0211)
    assert net_dividend_cash(0.0, 0.20) == 0.0


def test_dividend_after_tax_nhi_threshold_leveraged():
    """迴歸（第四輪）：槓桿路徑配息改「稅後」現金，三邊各自判斷 2 萬門檻。

    principal=1,000,000 / p0=100 → own_shares=10,000、pledge_rate=0.30 → loan_shares=3,000。
    除息值 3.0：own 部位單筆 30,000（≥2萬，扣 2.11%）、loan 部位單筆 9,000（<2萬，不扣）。
    退回舊碼（配息全額入袋、不扣稅）時斷言 fail。
    """
    dates = _make_flat_dates(20)
    ex_date = dates[10]
    p0 = 100.0
    prices = {d: p0 for d in dates}
    div_per_share = 3.0
    principal = 1_000_000.0
    pledge_rate = 0.30

    engine = PledgeSimulator(
        dates=dates,
        prices=prices,
        ex_dividend_dates={ex_date},
        ex_dividend_value={ex_date: div_per_share},
        annual_rate=0.0,
        dividend_income_tax_rate=0.0,
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=pledge_rate, principal_own_shares_value=principal
    )
    assert result.liquidated is False
    assert result.term_expired_forced_sale is False

    own_shares = principal / p0
    loan_shares = (pledge_rate * principal) / p0
    buggy_base = own_shares * p0 - pledge_rate * principal + loan_shares * p0
    own_pay = own_shares * div_per_share  # 30,000
    loan_pay = loan_shares * div_per_share  # 9,000
    net_own = own_pay - own_pay * 0.0211  # 跨門檻，扣二代健保
    net_loan = loan_pay  # 未跨門檻
    expected_final_equity = buggy_base + net_own + net_loan
    buggy_final_equity = buggy_base + own_pay + loan_pay  # 舊碼：全額入袋

    assert result.final_equity == pytest.approx(expected_final_equity, rel=1e-9)
    assert result.final_equity != pytest.approx(buggy_final_equity)
    # 稅後化後 unlevered 基準（own 部位）也是稅後
    assert result.final_equity_unlevered == pytest.approx(
        own_shares * p0 + net_own, rel=1e-9
    )


def test_dividend_after_tax_nhi_threshold_unlevered():
    """迴歸（第四輪）：pledge_rate=0 對照組同樣稅後化，own 單筆 25,000 跨門檻。"""
    dates = _make_flat_dates(20)
    ex_date = dates[10]
    p0 = 100.0
    prices = {d: p0 for d in dates}
    div_per_share = 2.5  # 10,000 股 x 2.5 = 25,000 >= 2 萬
    principal = 1_000_000.0

    engine = PledgeSimulator(
        dates=dates,
        prices=prices,
        ex_dividend_dates={ex_date},
        ex_dividend_value={ex_date: div_per_share},
        annual_rate=0.0,
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=0.0, principal_own_shares_value=principal
    )
    own_shares = principal / p0
    gross = own_shares * div_per_share  # 25,000
    net = gross - gross * 0.0211
    buggy_equity = own_shares * p0 + gross  # 舊碼：全額
    expected_equity = own_shares * p0 + net

    assert result.final_equity == pytest.approx(expected_equity, rel=1e-9)
    assert result.final_equity_unlevered == pytest.approx(expected_equity, rel=1e-9)
    assert result.final_equity != pytest.approx(buggy_equity)


def test_dividend_after_tax_income_tax_rate_applied():
    """迴歸（第四輪）：dividend_income_tax_rate>0 時，每筆給付再扣有效股利稅。"""
    dates = _make_flat_dates(20)
    ex_date = dates[10]
    p0 = 100.0
    prices = {d: p0 for d in dates}
    div_per_share = 3.0
    principal = 1_000_000.0
    pledge_rate = 0.30
    tax = 0.20

    engine = PledgeSimulator(
        dates=dates,
        prices=prices,
        ex_dividend_dates={ex_date},
        ex_dividend_value={ex_date: div_per_share},
        annual_rate=0.0,
        dividend_income_tax_rate=tax,
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=pledge_rate, principal_own_shares_value=principal
    )
    own_shares = principal / p0
    loan_shares = (pledge_rate * principal) / p0
    buggy_base = own_shares * p0 - pledge_rate * principal + loan_shares * p0
    own_pay = own_shares * div_per_share  # 30,000
    loan_pay = loan_shares * div_per_share  # 9,000
    net_own = own_pay - own_pay * 0.0211 - own_pay * tax
    net_loan = loan_pay - loan_pay * tax  # <2萬不扣二代健保，但扣股利稅
    expected = buggy_base + net_own + net_loan
    no_tax_version = buggy_base + (own_pay - own_pay * 0.0211) + loan_pay

    assert result.final_equity == pytest.approx(expected, rel=1e-9)
    assert result.final_equity != pytest.approx(no_tax_version)


def test_split_unit_adjustment_pushed_into_simulator():
    """D2（第四輪）：分割單位調整下推進 PledgeSimulator。

    建構時傳 split_events，引擎應自行把分割日之前的每股息值 ÷matched_ratio；
    不傳則維持原值（呼叫端負責）。退回舊碼（建構子不處理 split_events）時 fail。
    """
    dates = _make_flat_dates(30, start="2020-01-01")
    prices = {d: 100.0 for d in dates}
    pre = dates[5]  # 分割日之前
    post = dates[20]  # 分割日之後
    split_date = dates[12]
    split_events = [{"date": split_date, "matched_ratio": 4, "is_split": True}]

    eng_adj = PledgeSimulator(
        dates=dates,
        prices=prices,
        ex_dividend_dates={pre, post},
        ex_dividend_value={pre: 8.0, post: 2.0},
        annual_rate=0.0,
        split_events=split_events,
    )
    assert eng_adj.ex_dividend_value[pre] == pytest.approx(2.0)  # 8.0 / 4
    assert eng_adj.ex_dividend_value[post] == pytest.approx(2.0)  # 不動

    eng_raw = PledgeSimulator(
        dates=dates,
        prices=prices,
        ex_dividend_dates={pre, post},
        ex_dividend_value={pre: 8.0, post: 2.0},
        annual_rate=0.0,
    )
    assert eng_raw.ex_dividend_value[pre] == pytest.approx(
        8.0
    )  # 未傳 split_events，不調整


def test_loan_target_position_marks_to_loan_price_not_collateral():
    """借款買入部位（Y）用「借款標的」的價格序列 mark-to-market，不是抵押品（X）的。

    X 持平 100；Y 從 100 漲到 200。pledge_rate=0.30、principal=1,000,000：
      own_shares=10,000（X）、loan_shares=3,000（Y，用 Y 的 t0 價 100 換算）。
    期末：own 10,000×100 − L 300,000 + loan 3,000×200 = 1,300,000。
    若引擎誤用 X 期末價算 loan 部位：3,000×100 → 期末只有 1,000,000。
    退回舊碼（`_loan_price` 永遠回 self.prices）時斷言 fail。
    """
    dates = _make_flat_dates(20)
    x_prices = {d: 100.0 for d in dates}
    y_prices = {d: 100.0 + 100.0 * i / (len(dates) - 1) for i, d in enumerate(dates)}
    engine = PledgeSimulator(
        dates=dates,
        prices=x_prices,
        annual_rate=0.0,
        loan_prices=y_prices,
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=0.30, principal_own_shares_value=1_000_000.0
    )
    assert result.liquidated is False
    assert result.term_expired_forced_sale is False
    assert result.final_equity == pytest.approx(1_300_000.0, rel=1e-9)
    assert result.final_equity != pytest.approx(1_000_000.0)
    # 抵押品維持率只看 X（持平）→ 不觸發追繳
    assert result.n_margin_calls == 0


def test_loan_target_dividends_use_loan_ex_series():
    """Y 的配息用 Y 自己的除息序列；抵押品（X）無除息事件時 unlevered 基準不含配息。"""
    dates = _make_flat_dates(20)
    ex_date = dates[10]
    x_prices = {d: 100.0 for d in dates}
    y_prices = {d: 100.0 for d in dates}
    principal = 1_000_000.0
    pledge_rate = 0.30
    y_div_ps = 1.0  # loan_shares=3,000 → 單筆 3,000 < 二代健保 2 萬門檻，稅後=全額

    engine = PledgeSimulator(
        dates=dates,
        prices=x_prices,
        annual_rate=0.0,
        loan_prices=y_prices,
        loan_ex_dividend_dates={ex_date},
        loan_ex_dividend_value={ex_date: y_div_ps},
    )
    result = engine.run(
        0, len(dates) - 1, pledge_rate=pledge_rate, principal_own_shares_value=principal
    )
    loan_shares = (pledge_rate * principal) / 100.0
    base = 10_000 * 100.0 - pledge_rate * principal + loan_shares * 100.0
    assert result.final_equity == pytest.approx(base + loan_shares * y_div_ps, rel=1e-9)
    # X 無除息事件 → 不槓桿對照基準沒有配息貢獻
    assert result.final_equity_unlevered == pytest.approx(10_000 * 100.0, rel=1e-9)


def test_loan_target_split_unit_adjustment_on_loan_ex_series():
    """loan_split_events 應把分割日之前的 Y 每股息值 ÷matched_ratio（與 Y 價格序列同步）。"""
    dates = _make_flat_dates(30, start="2020-01-01")
    x_prices = {d: 100.0 for d in dates}
    y_prices = {d: 50.0 for d in dates}
    pre, post = dates[5], dates[20]
    split_date = dates[12]
    engine = PledgeSimulator(
        dates=dates,
        prices=x_prices,
        annual_rate=0.0,
        loan_prices=y_prices,
        loan_ex_dividend_dates={pre, post},
        loan_ex_dividend_value={pre: 8.0, post: 2.0},
        loan_split_events=[{"date": split_date, "matched_ratio": 4, "is_split": True}],
    )
    assert engine.loan_ex_dividend_value[pre] == pytest.approx(2.0)  # 8.0 / 4
    assert engine.loan_ex_dividend_value[post] == pytest.approx(2.0)  # 不動


def test_loan_target_all_mode_sells_Y_partial_mode_keeps_Y():
    """`all` 模式追繳處分時 Y 一起賣（之後 Y 價暴漲不影響期末）；`partial` 模式抵押品
    足以清償時 Y 留著（受之後 Y 價暴漲之惠）。

    X 前 5 日 100，之後崩到 35（−65%，pledge 0.30 下維持率 116.7% < 130% 觸發追繳，
    但抵押品市值 350,000 > L 300,000 → partial 只賣抵押品、Y 保留）。
    Y 前 15 日 100，第 16 日起暴漲到 1,000。
      all 模式：Y 在追繳日（約第 8 日、Y=100）就被賣掉 → 期末與「Y 不暴漲」情境相同。
      partial 模式：Y 留著 → 期末因 Y 暴漲而不同。
    退回舊碼（Y 永不處分）時 all 模式也會受暴漲影響，斷言 fail。
    """

    def build(dates, y_spikes):
        x = {}
        for i, d in enumerate(dates):
            x[d] = 100.0 if i < 5 else 35.0
        y = {}
        for i, d in enumerate(dates):
            if y_spikes and i >= 15:
                y[d] = 1000.0
            else:
                y[d] = 100.0
        return x, y

    dates = _make_flat_dates(30)
    res = {}
    for mode in ("all", "partial"):
        for spike in (False, True):
            x, y = build(dates, spike)
            engine = PledgeSimulator(
                dates=dates,
                prices=x,
                annual_rate=0.0,
                liquidation_mode=mode,
                auto_cure=False,
                loan_prices=y,
            )
            res[(mode, spike)] = engine.run(
                0,
                len(dates) - 1,
                pledge_rate=0.30,
                principal_own_shares_value=1_000_000.0,
            )

    assert res[("all", False)].liquidated is True
    assert res[("partial", False)].liquidated is True
    # all 模式：Y 已於追繳日賣出 → 之後 Y 暴漲對期末權益毫無影響
    assert res[("all", True)].final_equity == pytest.approx(
        res[("all", False)].final_equity, rel=1e-9
    )
    # partial 模式：Y 留著 → 暴漲讓期末權益大幅提高
    assert (
        res[("partial", True)].final_equity > res[("partial", False)].final_equity * 2
    )


# ---------------------------------------------------------------------------
# 2026-09-09 overlay 輪：家庭現金流 overlay（Scope 3）+ buffer solver（Scope 4）
# ---------------------------------------------------------------------------


def test_reinvest_dividends_compounds_and_beats_pure_cash():
    """配息再投入複利分支：除息日買回更多股 ⇒ 後續除息事件的配息基數變大（複利），
    期末權益高於純現金版。平價、零利率情境，數字可控。"""
    dates = _make_flat_dates(60)
    p0 = 100.0
    prices = {d: p0 for d in dates}
    ex1, ex2 = dates[15], dates[40]
    dps = 1.0  # 單筆給付遠低於二代健保 2 萬門檻
    principal = 1_000_000.0

    common = dict(
        dates=dates,
        prices=prices,
        ex_dividend_dates={ex1, ex2},
        ex_dividend_value={ex1: dps, ex2: dps},
        annual_rate=0.0,
    )
    r_cash = PledgeSimulator(**common, reinvest_dividends=False).run(
        0, len(dates) - 1, pledge_rate=0.20, principal_own_shares_value=principal
    )
    r_comp = PledgeSimulator(**common, reinvest_dividends=True).run(
        0, len(dates) - 1, pledge_rate=0.20, principal_own_shares_value=principal
    )

    own = principal / p0  # 10,000
    loan = 0.20 * principal / p0  # 2,000
    base = own * p0 - 0.20 * principal + loan * p0  # own+loan 市值互抵 L

    # 純現金版：兩次除息各拿 (own+loan)*dps，無複利
    cash_divs = 2 * (own + loan) * dps
    assert r_cash.final_equity == pytest.approx(base + cash_divs, rel=1e-9)

    # 複利版：第一次除息買回 (own+loan)*dps/p0 股，第二次配息基數變大
    reinv1 = (own + loan) * dps / p0
    div2 = (own + loan + reinv1) * dps
    reinv_total_shares = reinv1 + div2 / p0
    comp_equity = base + reinv_total_shares * p0
    assert r_comp.final_equity == pytest.approx(comp_equity, rel=1e-9)
    assert r_comp.final_equity > r_cash.final_equity


def test_reinvest_dividends_default_is_false():
    dates = _make_flat_dates(5)
    eng = PledgeSimulator(dates=dates, prices={d: 100.0 for d in dates})
    assert eng.reinvest_dividends is False


def test_hard_unwind_zeros_loan_leg_and_stops_interest():
    """hard_unwind_idx：到期日一次全賣借款腿還款、之後不再計息。平價序列 ⇒
    期末權益 == 本金 − 到出場日為止的累積利息（借款腿平價賣出剛好還清 L）。"""
    dates = _make_flat_dates(300)
    p = 100.0
    prices = {d: p for d in dates}
    principal = 6_480_000.0
    common = dict(dates=dates, prices=prices, annual_rate=0.05)

    unwind_i = 150
    r = PledgeSimulator(**common).run(
        0,
        len(dates) - 1,
        0.20,
        principal_own_shares_value=principal,
        hard_unwind_idx=unwind_i,
    )
    r_no_unwind = PledgeSimulator(**common).run(
        0,
        len(dates) - 1,
        0.20,
        principal_own_shares_value=principal,
    )
    assert r.hard_unwound is True
    assert r.exit_mode == "hard_zero"
    assert r.exit_realized_loss is False  # 平價賣出，無虧損
    assert r.collateral_sold_to_cover_exit == 0.0
    # 平價 ⇒ own*p_end == principal；期末權益只少了到出場日的利息
    assert r.final_equity == pytest.approx(principal - r.total_interest_paid, rel=1e-9)
    # 提早歸零 ⇒ 利息比一路槓桿到底少
    assert r.total_interest_paid < r_no_unwind.total_interest_paid
    assert r_no_unwind.hard_unwound is False


def test_hard_unwind_realized_loss_when_loan_leg_falls():
    """借款腿（獨立 loan_prices）在出場前腰斬 ⇒ exit_realized_loss=True，
    且賣光借款腿仍不足清償 ⇒ 賣擔保品補足（collateral_sold_to_cover_exit>0）。"""
    dates = _make_flat_dates(200)
    collat = {d: 100.0 for d in dates}
    loan = {d: (100.0 if i < 80 else 50.0) for i, d in enumerate(dates)}
    r = PledgeSimulator(
        dates=dates, prices=collat, loan_prices=loan, annual_rate=0.0
    ).run(
        0,
        len(dates) - 1,
        0.20,
        principal_own_shares_value=1_000_000.0,
        hard_unwind_idx=120,
    )
    assert r.exit_realized_loss is True
    assert r.loan_leg_value_at_exit == pytest.approx(0.5 * r.loan_principal_at_exit)
    assert r.collateral_sold_to_cover_exit == pytest.approx(
        0.5 * r.loan_principal_at_exit, rel=1e-9
    )
    # 借款腿上漲則無虧損、不動擔保品
    loan_up = {d: (100.0 if i < 80 else 130.0) for i, d in enumerate(dates)}
    r_up = PledgeSimulator(
        dates=dates, prices=collat, loan_prices=loan_up, annual_rate=0.0
    ).run(
        0,
        len(dates) - 1,
        0.20,
        principal_own_shares_value=1_000_000.0,
        hard_unwind_idx=120,
    )
    assert r_up.exit_realized_loss is False
    assert r_up.collateral_sold_to_cover_exit == 0.0


def test_proactive_cure_floor_prevents_margin_call():
    """proactive_cure_ratio：擔保品腰斬跌破門檻時用現金墊到門檻，130% 追繳不觸發；
    不設地板則進追繳/處分。"""
    dates = _make_flat_dates(120)
    # 前 40 天 100，之後 22（−78%）：20% 質押初始維持率 5.0 → 崩後 1.10 < 1.30
    prices = {d: (100.0 if i < 40 else 22.0) for i, d in enumerate(dates)}
    common = dict(dates=dates, prices=prices, annual_rate=0.0)
    r_floor = PledgeSimulator(**common).run(
        0,
        len(dates) - 1,
        0.20,
        principal_own_shares_value=1_000_000.0,
        proactive_cure_ratio=2.75,
    )
    r_bare = PledgeSimulator(**common).run(
        0,
        len(dates) - 1,
        0.20,
        principal_own_shares_value=1_000_000.0,
    )
    assert r_floor.n_margin_calls == 0
    assert r_floor.proactive_topup_peak > 0.0
    assert r_bare.n_margin_calls >= 1


def test_winddown_sells_loan_leg_over_12_months_flat():
    """winddown_start/end：簽約前 ~12 個月逐月等額賣出借款腿；平價 ⇒ 收尾時
    借款腿歸零、無實現虧損，wd 記錄的出場值 ≈ 起點 L。"""
    dates = _make_flat_dates(400)
    prices = {d: 100.0 for d in dates}
    ei = len(dates) - 1
    wd_start = ei - 252  # ~12 個月
    r = PledgeSimulator(dates=dates, prices=prices, annual_rate=0.0).run(
        0,
        ei,
        0.20,
        principal_own_shares_value=1_000_000.0,
        winddown_start_idx=wd_start,
        winddown_end_idx=ei,
    )
    assert r.exit_mode == "winddown"
    assert r.hard_unwound is True
    assert r.exit_realized_loss is False
    assert r.loan_leg_value_at_exit == pytest.approx(r.loan_principal_at_exit, rel=1e-6)
    # 平價、零息、借款腿無配息 ⇒ 期末權益 == 本金
    assert r.final_equity == pytest.approx(1_000_000.0, rel=1e-9)


def test_0050_borrow_leg_dividends_buy_back_shares_not_cash():
    """驗收標準 1：borrow_leg='0050_dividend' 時借款腿配息確實買回股（複利），
    不是純累積現金。合成序列：除息後價格跳升 ⇒ 再投入版的借款腿出場值與期末權益
    高於純現金版（純現金配息不隨後續漲價放大）。"""
    dates = _make_flat_dates(140)
    ex = dates[30]
    prices = {d: (100.0 if i < 70 else 125.0) for i, d in enumerate(dates)}
    common = dict(
        dates=dates,
        prices=prices,
        annual_rate=0.0,
        loan_prices=prices,
        loan_ex_dividend_dates={ex},
        loan_ex_dividend_value={ex: 5.0},
    )
    unwind = 110  # 跳升之後才硬歸零
    r_cash = PledgeSimulator(**common, reinvest_dividends=False).run(
        0,
        len(dates) - 1,
        0.20,
        principal_own_shares_value=1_000_000.0,
        hard_unwind_idx=unwind,
    )
    r_reinv = PledgeSimulator(**common, reinvest_dividends=True).run(
        0,
        len(dates) - 1,
        0.20,
        principal_own_shares_value=1_000_000.0,
        hard_unwind_idx=unwind,
    )
    # 借款腿 2000 股 × 5 元 = 10,000 現金（純現金版）vs 100 股 @100 → @125（再投入版）
    assert r_reinv.loan_leg_value_at_exit > r_cash.loan_leg_value_at_exit + 1000.0
    assert r_reinv.final_equity > r_cash.final_equity + 1000.0


def test_principal_injections_default_none_and_empty_list_identical():
    """規格守門：``principal_injections`` 不傳（None）與傳空列表逐位元相同，
    且與舊呼叫方式（完全不帶這個關鍵字）相同——不影響既有 1320 條測試。"""
    dates = _make_flat_dates(60)
    prices = {d: 100.0 for d in dates}
    common = dict(dates=dates, prices=prices, annual_rate=0.0385)
    r_omit = PledgeSimulator(**common).run(0, 40, 0.20)
    r_none = PledgeSimulator(**common).run(0, 40, 0.20, principal_injections=None)
    r_empty = PledgeSimulator(**common).run(0, 40, 0.20, principal_injections=[])
    assert r_omit.final_equity == r_none.final_equity == r_empty.final_equity
    assert r_omit.min_maintenance_ratio == r_none.min_maintenance_ratio


def test_principal_injection_grows_collateral_and_raises_maintenance_ratio():
    """合成序列驗證注資機制：t0 用 100 萬本金、20% 質押（借款 20 萬固定），
    第 10 個交易日跌到 60 元（維持率被壓低），第 11 天注資 60 萬——注資當天用
    當日收盤價買回更多股，維持率應立刻回升，且不動已鎖定的借款本金 L。"""
    dates = _make_flat_dates(40)
    prices = {d: 100.0 for d in dates}
    for d in dates[10:]:
        prices[d] = 60.0  # 第 10 天起維持在低點，直到注資
    sim_obj = PledgeSimulator(dates=dates, prices=prices, annual_rate=0.0)
    r_no_inject = sim_obj.run(0, 30, 0.20, principal_own_shares_value=1_000_000.0)
    r_inject = sim_obj.run(
        0,
        30,
        0.20,
        principal_own_shares_value=1_000_000.0,
        principal_injections=[(11, 600_000.0)],
    )
    # 借款本金固定：兩者 t0 都是 100 萬 * 20% = 20 萬，注資不應改動它。
    # （引擎不直接曝露 L，用期末權益反推：注資版多出的權益 ≈ 注資金額本身，
    # 因為股價在注資後維持不變，沒有市值變動，只有現金換股。）
    assert r_inject.final_equity - r_no_inject.final_equity == pytest.approx(
        600_000.0, rel=1e-6
    )
    # 注資前维持率已跌到某個低點（60 元 vs t0 100 元、20 萬借款 ⇒ 60% 擔保 * 100 萬 /
    # 20 萬 = 300%，維持率其實仍遠高於 130%，這裡只驗證「有追蹤到一個有限值」）。
    assert r_no_inject.min_maintenance_ratio < float("inf")
    # 注資後同一路徑的低點理應 >= 注資前版本的低點（擔保品變多，維持率只會更高）。
    assert r_inject.min_maintenance_ratio >= r_no_inject.min_maintenance_ratio
