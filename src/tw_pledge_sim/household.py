"""Monthly household cash-flow overlay on top of the pledge mechanics.

Layer 2 of the model: salary pays the mortgage and living costs first, then the
pledge interest (salary surplus, then buffer, then dividends, then selling
collateral units).  Margin calls are cured from the buffer; if that fails the
collateral is disposed of.  The monthly step is a one-way approximation (no
re-solving of call timing after a disposal).  All defaults are fictional.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date
from typing import Optional

from .adjust import split_adjust_ex_values
from .broker import GENERIC, BrokerProfile
from .data import StockInputs
from .engine import net_dividend_cash
from .regulation import CURE_TARGET_RATIO, MARGIN_CALL_RATIO
from .rolling import median, nearest_trading_day_idx

# Rolling starts of the overlay matrix begin here so that different loan targets are
# compared on the same set of onset months.
OVERLAY_START_MONTH = "2008-01"


@dataclass(frozen=True)
class HouseholdProfile:
    """Monthly household flows.  Defaults are round, clearly fictional numbers."""

    portfolio_principal: float = 1_000_000.0
    salary_monthly: float = 50_000.0
    living_monthly: float = 25_000.0
    dca_monthly: float = 3_000.0  # monthly buy of the collateral asset
    mortgage_principal: float = 3_000_000.0
    mortgage_years: int = 30
    mortgage_annual_rate: float = 0.025

    def mortgage_payment(self) -> float:
        return mortgage_monthly_payment(
            self.mortgage_principal, self.mortgage_annual_rate, self.mortgage_years
        )


def mortgage_monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    """Level-payment (annuity) monthly mortgage instalment."""
    r = annual_rate / 12.0
    n = years * 12
    if r == 0.0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def _month_end_obs(dates: list[str], onset_idx: int, end_idx: int):
    """[(month_key 'YYYY-MM', 該月最後一個交易日在 dates 的 index)]，涵蓋 onset..end。"""
    out: list[tuple[str, int]] = []
    cur_key: Optional[str] = None
    last_i = onset_idx
    for i in range(onset_idx, end_idx + 1):
        k = dates[i][:7]
        if cur_key is None or k == cur_key:
            cur_key = k
            last_i = i
        else:
            out.append((cur_key, last_i))
            cur_key = k
            last_i = i
    if cur_key is not None:
        out.append((cur_key, last_i))
    return out


def _days_in_month(month_key: str) -> int:
    y, m = int(month_key[:4]), int(month_key[5:7])
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return (date(ny, nm, 1) - date(y, m, 1)).days


def _asof_price(sorted_dates: list[str], series: dict[str, float], d: str) -> float:
    """series 在 d（含）之前最近一個交易日的收盤價。"""
    j = bisect.bisect_right(sorted_dates, d) - 1
    if j < 0:
        j = 0
    return series[sorted_dates[j]]


def _hh_dispose(
    mode: str,
    px: float,
    py: float,
    collateral_shares: float,
    loan_shares: float,
    L: float,
    loan_sale_cash: float,
) -> tuple[float, float, float, float]:
    """overlay 版強制處分：抵押品（all 全賣 / partial 賣到夠清償），Y 依 mode（與
    PledgeSimulator Scope 2 同語意）。回傳 (loan_sale_cash, collateral_shares,
    loan_shares, L)。"""
    if mode == "all":
        proceeds = collateral_shares * px
        collateral_shares = 0.0
    else:  # partial：只賣到足以清償 L
        sell = min(collateral_shares, (L / px) if px > 0 else collateral_shares)
        proceeds = sell * px
        collateral_shares -= sell
    L = max(0.0, L - proceeds)
    if loan_shares > 0.0 and py > 0.0:
        if mode == "all":
            lp = loan_shares * py
            loan_shares = 0.0
            applied = min(lp, L)
            L = max(0.0, L - applied)
            loan_sale_cash += lp - applied
        elif L > 0.0:  # partial 且抵押品清償不足
            sell_val = min(L, loan_shares * py)
            loan_shares -= sell_val / py
            L = max(0.0, L - sell_val)
    return loan_sale_cash, collateral_shares, loan_shares, L


@dataclass
class HouseholdResult:
    onset_date: str
    end_date: str
    onset_year: int
    holding_years: int
    pledge_rate: float
    loan_target: str
    dividend_income_tax_rate: float
    annual_rate: float
    initial_buffer: float
    final_net_wealth: float  # 含 buffer 的家庭流動淨財富
    invest_position_final: float  # 不含 buffer 的投資部位淨值
    max_drawdown_invest_equity: float  # 投資部位（不含 buffer）峰谷回撤
    n_margin_calls: int
    forced_disposal: bool
    disposal_month: Optional[str]
    term_forced_sale: bool
    monthly_buffer_delta_min: float
    monthly_buffer_delta_median: float
    monthly_buffer_delta_max: float
    buffer_path_min: float
    buffer_path_median: float
    buffer_path_max: float
    salary_covers_interest_all_months: bool
    worst_interest_to_salary_surplus: float  # max_m interest_m / (薪資結餘−DCA)
    dividends_diverted_to_interest_total: float
    dividend_tax_drag_total: float
    collateral_units_sold_value_total: float


def run_household_overlay(
    collat: StockInputs,
    loan: StockInputs,
    onset_idx: int,
    end_idx: int,
    pledge_rate: float,
    loan_target: str,
    *,
    broker: BrokerProfile = GENERIC,
    dividend_income_tax_rate: float = 0.0,
    initial_buffer: float = 0.0,
    profile: HouseholdProfile = HouseholdProfile(),
    reinvest_dividends: bool = False,
) -> HouseholdResult:
    """Monthly household overlay.  Collateral is ``collat``; the loan buys ``loan``.

    ``ex_values`` of both inputs are raw (not split-adjusted); the adjustment by
    1/matched_ratio before each split is applied here, in step with the price series.
    With ``reinvest_dividends`` the dividends not diverted to interest buy more
    collateral shares instead of going to the buffer.
    """
    collat_dates, collat_prices = collat.dates, collat.prices
    collat_ex_dates = collat.ex_dates
    loan_dates, loan_prices, loan_ex_dates = loan.dates, loan.prices, loan.ex_dates
    annual_rate = broker.annual_rate
    liquidation_mode = broker.liquidation_mode
    portfolio_principal = profile.portfolio_principal
    collat_ex_value = split_adjust_ex_values(collat.ex_values, collat.split_events)
    loan_ex_value = split_adjust_ex_values(loan.ex_values, loan.split_events)
    onset_date = collat_dates[onset_idx]
    end_date = collat_dates[end_idx]
    holding_years = round(
        (date.fromisoformat(end_date) - date.fromisoformat(onset_date)).days / 365.25
    )
    px0 = collat_prices[onset_date]
    py0 = _asof_price(loan_dates, loan_prices, onset_date)

    collateral_shares = portfolio_principal / px0
    L = pledge_rate * portfolio_principal
    loan_shares = (L / py0) if L > 0 else 0.0
    buffer = float(initial_buffer)
    posted_cash = 0.0  # 現金補繳，計入維持率分子，期末歸還權益
    loan_sale_cash = 0.0

    mortgage_pmt = profile.mortgage_payment()
    salary_surplus = profile.salary_monthly - mortgage_pmt - profile.living_monthly
    salary_avail_for_interest = salary_surplus - profile.dca_monthly

    n_margin_calls = 0
    forced_disposal = False
    disposal_month: Optional[str] = None
    term_forced_sale = False
    div_diverted = 0.0
    div_gross_total = 0.0
    div_net_total = 0.0
    units_sold_value = 0.0
    salary_covers_all = True
    worst_int_ratio = 0.0

    buffer_deltas: list[float] = []
    buffer_path: list[float] = []
    peak_invest = None
    max_dd = 0.0

    months = _month_end_obs(collat_dates, onset_idx, end_idx)
    term_counter = 0  # 每 6 個月一期，18 個月一個合約週期

    for month_key, mi in months:
        md = collat_dates[mi]
        px = collat_prices[md]
        py = _asof_price(loan_dates, loan_prices, md)
        buffer_before = buffer

        # --- 本月配息（用月初持股數，除息事件落在本月即入帳）---
        div_cash_month = 0.0
        for exd in collat_ex_dates:
            if exd[:7] == month_key:
                gross = collateral_shares * collat_ex_value.get(exd, 0.0)
                div_gross_total += gross
                net = net_dividend_cash(gross, dividend_income_tax_rate)
                div_net_total += net
                div_cash_month += net
        for exd in loan_ex_dates:
            if exd[:7] == month_key:
                gross = loan_shares * loan_ex_value.get(exd, 0.0)
                div_gross_total += gross
                net = net_dividend_cash(gross, dividend_income_tax_rate)
                div_net_total += net
                div_cash_month += net

        # --- DCA：買入市值型抵押品（資金來自薪資結餘，不足才動 buffer）---
        collateral_shares += profile.dca_monthly / px
        avail = salary_surplus - profile.dca_monthly
        if avail < 0.0:
            buffer += avail  # 薪資結餘不夠付 DCA，先扣 buffer
            avail = 0.0
            if buffer < 0.0:
                deficit = -buffer
                collateral_shares -= deficit / px
                units_sold_value += deficit
                buffer = 0.0

        # --- 質押利息（本月）：薪資結餘 → buffer → 配息 → 賣抵押品單位 ---
        interest_m = (
            L * annual_rate * _days_in_month(month_key) / 365.0 if L > 0 else 0.0
        )
        if L > 0 and salary_avail_for_interest > 0:
            worst_int_ratio = max(
                worst_int_ratio, interest_m / salary_avail_for_interest
            )
        elif L > 0:
            worst_int_ratio = float("inf")

        rem = interest_m
        pay_salary = min(rem, max(0.0, avail))
        avail -= pay_salary
        rem -= pay_salary
        if rem > 1e-9:
            salary_covers_all = False
        pay_buffer = min(rem, buffer)
        buffer -= pay_buffer
        rem -= pay_buffer
        pay_div = min(rem, div_cash_month)
        div_cash_month -= pay_div
        div_diverted += pay_div
        rem -= pay_div
        if rem > 1e-9:  # 仍不足：賣抵押品單位付息
            collateral_shares -= rem / px
            units_sold_value += rem
            rem = 0.0

        # --- 月結餘進 buffer（薪資剩餘 + 未被利息吃掉的配息）---
        buffer += max(0.0, avail)
        if reinvest_dividends and div_cash_month > 0.0 and px > 0.0:
            # 複利版：剩餘配息買回抵押品股，不進 buffer
            collateral_shares += div_cash_month / px
        else:
            buffer += div_cash_month

        # --- 維持率（月底，近似第 21 條：除息事件在 ~8 個日曆天內則扣權息值）---
        haircut_ps = 0.0
        md_d = date.fromisoformat(md)
        for exd in collat_ex_dates:
            exd_d = date.fromisoformat(exd)
            if 0 <= (exd_d - md_d).days <= 8:
                haircut_ps = collat_ex_value.get(exd, 0.0)
                break
        collat_mv = collateral_shares * (px - haircut_ps) + posted_cash
        maint_ratio = collat_mv / L if L > 0 else float("inf")

        # --- 追繳／單月補繳／強制處分（月頻近似）---
        # 單向近似：本月若跌破 130%，先用 buffer 現金補繳到 166.67%；補不上就當月強制
        # 處分，不在同一趟內回頭重解「追繳時點會不會因處分而位移」（見 findings 限制清單）。
        if not forced_disposal and L > 0 and maint_ratio < MARGIN_CALL_RATIO:
            n_margin_calls += 1
            need = CURE_TARGET_RATIO * L - collat_mv
            post = min(max(0.0, need), buffer)
            buffer -= post
            posted_cash += post
            collat_mv += post
            maint_ratio = collat_mv / L
            if maint_ratio < MARGIN_CALL_RATIO:
                loan_sale_cash, collateral_shares, loan_shares, L = _hh_dispose(
                    liquidation_mode,
                    px,
                    py,
                    collateral_shares,
                    loan_shares,
                    L,
                    loan_sale_cash,
                )
                forced_disposal = True
                disposal_month = month_key

        # --- 18 個月合約週期（第 4 條，月頻近似：每 6 個月一期，第 18 期門檻 166.67%）---
        m_ord = len(buffer_path) + 1  # 本月是第幾個月（1-based）
        if not forced_disposal and L > 0 and m_ord % 6 == 0:
            term_counter += 6
            thr = CURE_TARGET_RATIO if term_counter >= 18 else MARGIN_CALL_RATIO
            if maint_ratio < thr:
                term_forced_sale = True
                forced_disposal = True
                disposal_month = month_key
                loan_sale_cash, collateral_shares, loan_shares, L = _hh_dispose(
                    "all", px, py, collateral_shares, loan_shares, L, loan_sale_cash
                )
            if term_counter >= 18:
                term_counter = 0

        # --- 記錄路徑 ---
        invest_eq = (
            collateral_shares * px + loan_shares * py - L + posted_cash + loan_sale_cash
        )
        if peak_invest is None or invest_eq > peak_invest:
            peak_invest = invest_eq
        if peak_invest and peak_invest > 0:
            dd = (peak_invest - invest_eq) / peak_invest
            if dd > max_dd:
                max_dd = dd
        buffer_deltas.append(buffer - buffer_before)
        buffer_path.append(buffer)

    end_px = collat_prices[end_date]
    end_py = _asof_price(loan_dates, loan_prices, end_date)
    invest_final = (
        collateral_shares * end_px
        + loan_shares * end_py
        - L
        + posted_cash
        + loan_sale_cash
    )
    final_wealth = invest_final + buffer

    return HouseholdResult(
        onset_date=onset_date,
        end_date=end_date,
        onset_year=int(onset_date[:4]),
        holding_years=holding_years,
        pledge_rate=pledge_rate,
        loan_target=loan_target,
        dividend_income_tax_rate=dividend_income_tax_rate,
        annual_rate=annual_rate,
        initial_buffer=float(initial_buffer),
        final_net_wealth=final_wealth,
        invest_position_final=invest_final,
        max_drawdown_invest_equity=max_dd,
        n_margin_calls=n_margin_calls,
        forced_disposal=forced_disposal,
        disposal_month=disposal_month,
        term_forced_sale=term_forced_sale,
        monthly_buffer_delta_min=min(buffer_deltas) if buffer_deltas else 0.0,
        monthly_buffer_delta_median=median(buffer_deltas) if buffer_deltas else 0.0,
        monthly_buffer_delta_max=max(buffer_deltas) if buffer_deltas else 0.0,
        buffer_path_min=min(buffer_path) if buffer_path else 0.0,
        buffer_path_median=median(buffer_path) if buffer_path else 0.0,
        buffer_path_max=max(buffer_path) if buffer_path else 0.0,
        salary_covers_interest_all_months=salary_covers_all,
        worst_interest_to_salary_surplus=worst_int_ratio,
        dividends_diverted_to_interest_total=div_diverted,
        dividend_tax_drag_total=div_gross_total - div_net_total,
        collateral_units_sold_value_total=units_sold_value,
    )


def overlay_rolling_starts(collat_dates: list[str], holding_years: int) -> list[int]:
    """One onset per month from ``OVERLAY_START_MONTH`` while the window still fits."""
    starts: list[int] = []
    months = sorted({d[:7] for d in collat_dates if d[:7] >= OVERLAY_START_MONTH})
    for ym in months:
        oi = nearest_trading_day_idx(collat_dates, ym + "-01")
        if oi is None:
            continue
        od = collat_dates[oi]
        tgt = f"{int(od[:4]) + holding_years}{od[4:]}"
        if nearest_trading_day_idx(collat_dates, tgt) is None:
            break
        starts.append(oi)
    return starts


def run_overlay_matrix_cell(
    collat: StockInputs,
    loan: StockInputs,
    pledge_rate: float,
    loan_target: str,
    holding_years: int,
    *,
    broker: BrokerProfile = GENERIC,
    dividend_income_tax_rate: float = 0.0,
    initial_buffer: float = 0.0,
    profile: HouseholdProfile = HouseholdProfile(),
    reinvest_dividends: bool = False,
) -> list[HouseholdResult]:
    """One matrix cell: run the overlay across every rolling start."""
    cd = collat.dates
    out: list[HouseholdResult] = []
    for oi in overlay_rolling_starts(cd, holding_years):
        od = cd[oi]
        ei = nearest_trading_day_idx(cd, f"{int(od[:4]) + holding_years}{od[4:]}")
        out.append(
            run_household_overlay(
                collat,
                loan,
                oi,
                ei,
                pledge_rate,
                loan_target,
                broker=broker,
                dividend_income_tax_rate=dividend_income_tax_rate,
                initial_buffer=initial_buffer,
                profile=profile,
                reinvest_dividends=reinvest_dividends,
            )
        )
    return out


def forced_disposal_stats(results: list[HouseholdResult]) -> dict:
    """Forced-disposal probability over rolling starts, with the onset-year breakdown."""
    n = len(results)
    disp = [r for r in results if r.forced_disposal]
    by_year: dict[int, int] = {}
    for r in disp:
        by_year[r.onset_year] = by_year.get(r.onset_year, 0) + 1
    return {
        "n_starts": n,
        "n_forced_disposal": len(disp),
        "forced_disposal_prob": (len(disp) / n) if n else float("nan"),
        "by_onset_year": dict(sorted(by_year.items())),
        "worst_year": (
            min(disp, key=lambda r: r.final_net_wealth).onset_year if disp else None
        ),
    }


def solve_min_buffer(
    collat: StockInputs,
    loan: StockInputs,
    pledge_rate: float,
    loan_target: str,
    holding_years: int,
    *,
    broker: BrokerProfile = GENERIC,
    dividend_income_tax_rate: float = 0.0,
    profile: HouseholdProfile = HouseholdProfile(),
    buf_hi: float = 20_000_000.0,
    tol: float = 25_000.0,
) -> dict:
    """Smallest initial buffer such that no historical start is force-disposed.

    Disposals are non-increasing in the buffer, so bisection applies.  If even
    ``buf_hi`` leaves disposals, report that no finite buffer exists.
    """

    def cell(buffer: float) -> list[HouseholdResult]:
        return run_overlay_matrix_cell(
            collat,
            loan,
            pledge_rate,
            loan_target,
            holding_years,
            broker=broker,
            dividend_income_tax_rate=dividend_income_tax_rate,
            initial_buffer=buffer,
            profile=profile,
        )

    def n_disp(buffer: float) -> int:
        return sum(1 for r in cell(buffer) if r.forced_disposal)

    stats0 = forced_disposal_stats(cell(0.0))
    if stats0["n_forced_disposal"] == 0:
        return {"min_buffer": 0.0, "finite": True, "stats_at_zero": stats0}
    if n_disp(buf_hi) > 0:
        stats_hi = forced_disposal_stats(cell(buf_hi))
        return {
            "min_buffer": None,
            "finite": False,
            "residual_prob_at_hi": stats_hi["forced_disposal_prob"],
            "worst_year": stats_hi["worst_year"],
            "buf_hi": buf_hi,
        }
    lo, hi = 0.0, buf_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if n_disp(mid) == 0:
            hi = mid
        else:
            lo = mid
    return {"min_buffer": hi, "finite": True, "stats_at_zero": stats0}
