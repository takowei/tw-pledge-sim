"""Pledge engine: one security, one loan, one rolling-start path, daily margin logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .regulation import (
    CURE_TARGET_RATIO,
    CURE_WINDOW_DAYS,
    DEFAULT_PENALTY_RATE_MULT,
    EX_DIVIDEND_LOOKBACK_DAYS,
    LIQUIDATION_START_BUSINESS_DAY,
    MARGIN_CALL_RATIO,
    MAX_ROLLOVERS,
    NHI_SUPP_RATE,
    NHI_SUPP_SINGLE_CAP,
    NHI_SUPP_SINGLE_THRESHOLD,
)


def net_dividend_cash(gross: float, income_tax_rate: float = 0.0) -> float:
    """單筆配息現金的稅後金額。

    扣除兩項：
      1. 二代健保補充保費——單筆給付 >= NHI_SUPP_SINGLE_THRESHOLD 時，
         對 min(給付, NHI_SUPP_SINGLE_CAP) 課 NHI_SUPP_RATE；未達門檻不扣。
      2. 股利所得稅——以 income_tax_rate 有效稅率對整筆給付課徵，無門檻。
    不含投資建議，只是把已知稅負算進現金流。
    """
    if gross <= 0.0:
        return gross
    nhi = (
        NHI_SUPP_RATE * min(gross, NHI_SUPP_SINGLE_CAP)
        if gross >= NHI_SUPP_SINGLE_THRESHOLD
        else 0.0
    )
    income_tax = income_tax_rate * gross
    return gross - nhi - income_tax


@dataclass
class MarginCallEvent:
    trigger_date: str
    ratio_at_trigger: float
    cured: bool = False
    cure_date: Optional[str] = None
    liquidated: bool = False
    liquidation_date: Optional[str] = None


@dataclass
class SimResult:
    onset_date: str
    end_date: str
    stock_no: str
    pledge_rate: float
    liquidation_mode: str
    margin_call_events: list[MarginCallEvent] = field(default_factory=list)
    liquidated: bool = False
    liquidation_date: Optional[str] = None
    term_expired_forced_sale: bool = False
    final_equity: float = 0.0
    final_equity_unlevered: float = 0.0
    max_drawdown_equity: float = 0.0
    total_interest_paid: float = 0.0
    default_penalty_paid: float = 0.0  # 第 26 條：融通利率 × 10%，逾期未補繳期間
    n_margin_calls: int = 0
    # --- 早窗口質押組態（2026-09-09 第六輪，findings §11）---
    # exit_mode：'none'（沿用舊行為）／'hard_zero'（日曆式硬歸零，2034 年底把借款
    # 加碼部位一次全賣還款）／'winddown'（房屋簽約前 12 個月逐月等額賣出）。
    exit_mode: str = "none"
    hard_unwound: bool = False
    exit_date: Optional[str] = None
    loan_leg_value_at_exit: float = 0.0  # 出場時賣借款腿換得的現金（含再投入股）
    loan_principal_at_exit: float = 0.0  # 出場當下尚未清償的借款本金 L
    exit_realized_loss: bool = False  # loan_leg_value_at_exit < loan_principal_at_exit
    collateral_sold_to_cover_exit: float = 0.0  # 借款腿不足清償、被迫賣掉的擔保品金額
    proactive_topup_peak: float = (
        0.0  # 主動補繳門檻（>130% 前先補）曾同時墊入的最大現金
    )
    # --- 本金分段成長（2026-09-13，小本金早期啟動情境）---
    # 每日「補繳前」的原始維持率（mv/L）跑過的最低點；只在 L>0 且尚未處分/到期
    # 強制平倉的交易日更新。principal_injections 未使用時，此欄位就是單純的
    # 「這條路徑歷史最低維持率」，可回答「維持率最低點」而不需另跑一次。
    min_maintenance_ratio: float = float("inf")


class PledgeSimulator:
    """單一標的、單次融通、rolling-start 質押路徑模擬器。

    使用者用自有資金買入 `principal` 元的標的，同時以持有部位質押借出
    `pledge_rate × principal_market_value_at_t0` 買回同一標的（加碼）。
    之後逐營業日重算維持率，觸發追繳/展延/處分依 pledge-mechanics-spec 規則。
    """

    def __init__(
        self,
        dates: list[str],
        prices: dict[str, float],
        ex_dividend_dates: Optional[set[str]] = None,
        ex_dividend_value: Optional[dict[str, float]] = None,
        annual_rate: float = 0.035,
        liquidation_mode: str = "all",
        auto_cure: bool = True,
        allow_rollover: bool = True,
        dividend_income_tax_rate: float = 0.0,
        split_events: Optional[list[dict]] = None,
        loan_prices: Optional[dict[str, float]] = None,
        loan_ex_dividend_dates: Optional[set[str]] = None,
        loan_ex_dividend_value: Optional[dict[str, float]] = None,
        loan_split_events: Optional[list[dict]] = None,
        reinvest_dividends: bool = False,
    ):
        if liquidation_mode not in ("all", "partial"):
            raise ValueError("liquidation_mode must be 'all' or 'partial'")
        # 配息再投入複利分支（findings §10，第五輪）：True 時每筆稅後配息在除息日
        # 當天用「該部位對應標的的收盤價」買回更多股，取代「純累積現金」。
        # 買回的股不併入擔保品（不進維持率分子/分母，比照 loan 腳的模型設計），
        # 只在期末權益 mark-to-market。flag 關閉（預設）時完全不觸發任何新分支，
        # 與第四輪的純現金路徑逐位元相同。
        self.reinvest_dividends = reinvest_dividends
        self.dates = dates  # 已排序的營業日 ISO 字串
        self.prices = prices
        self.ex_dividend_dates = ex_dividend_dates or set()
        self.ex_dividend_value = dict(ex_dividend_value or {})
        self.annual_rate = annual_rate
        self.liquidation_mode = liquidation_mode
        self.auto_cure = auto_cure
        self.allow_rollover = allow_rollover
        self.dividend_income_tax_rate = dividend_income_tax_rate
        # --- 借款標的 ≠ 抵押品標的（2026-09-09 overlay 輪，Scope 2）---
        # 抵押品部位（own_shares，維持率分子唯一來源）永遠用 self.prices / self.ex_*。
        # 借款買入的加碼部位（loan_shares）若呼叫端提供 loan_prices，則改用「借款標的 Y」
        # 的價格與除息序列各自 mark-to-market 與入息；未提供則 Y == 抵押品標的（沿用舊行為，
        # 現有 1290 測試不受影響）。維持率/追繳/除息前 6 日凹陷（第 16/20/21 條）只看抵押品，
        # 不看 Y —— 規格書那三條不動。
        self.loan_prices = loan_prices  # None ⇒ 同抵押品
        self.loan_ex_dividend_dates = (
            set(loan_ex_dividend_dates) if loan_ex_dividend_dates is not None else None
        )
        self.loan_ex_dividend_value = (
            dict(loan_ex_dividend_value) if loan_ex_dividend_value is not None else None
        )
        # D2（2026-09-09）：分割單位調整下推。先前只在 `_cli_run_phase_c` 對事件列表
        # 呼叫 `adjust_dividend_events_for_splits`，繞過 CLI 直接餵 `ex_dividend_value`
        # 者會重蹈 findings §8.2（分割前息值高估 matched_ratio 倍）。改為：若呼叫端
        # 提供 `split_events`，引擎自己把分割日之前的每股息值 ÷matched_ratio，與
        # `build_split_adjusted_series` 對價格的回溯調整同步。未提供則維持原行為
        # （呼叫端負責，合成測試資料無分割即無需調整）。
        if split_events:
            real_splits = [e for e in split_events if e.get("is_split")]
            for ev in sorted(real_splits, key=lambda e: e["date"], reverse=True):
                ratio = ev["matched_ratio"]
                cutoff = ev["date"]
                for dt in list(self.ex_dividend_value.keys()):
                    if dt < cutoff:
                        self.ex_dividend_value[dt] = self.ex_dividend_value[dt] / ratio
        # 借款標的 Y 的除息值同樣做分割單位回溯調整（與 Y 的價格序列同步，見 D2）。
        if loan_split_events and self.loan_ex_dividend_value is not None:
            real_splits = [e for e in loan_split_events if e.get("is_split")]
            for ev in sorted(real_splits, key=lambda e: e["date"], reverse=True):
                ratio = ev["matched_ratio"]
                cutoff = ev["date"]
                for dt in list(self.loan_ex_dividend_value.keys()):
                    if dt < cutoff:
                        self.loan_ex_dividend_value[dt] = (
                            self.loan_ex_dividend_value[dt] / ratio
                        )

    def _loan_price(self, d: str) -> float:
        """借款買入部位（Y）在某營業日的收盤價；未設 Y 序列時退回抵押品收盤價。"""
        if self.loan_prices is None:
            return self.prices[d]
        return self.loan_prices[d]

    def _loan_div_per_share(self, d: str) -> float:
        """借款買入部位（Y）在某營業日的每股權息值；未設 Y 序列時退回抵押品的除息設定。"""
        if self.loan_prices is None:
            if d in self.ex_dividend_dates:
                return self.ex_dividend_value.get(d, 0.0)
            return 0.0
        if self.loan_ex_dividend_dates and d in self.loan_ex_dividend_dates:
            return (self.loan_ex_dividend_value or {}).get(d, 0.0)
        return 0.0

    def _haircut_per_share(self, d_idx: int) -> float:
        """第 21 條：除權息交易日前 6 個營業日起，「每股」收盤價扣除權息值。

        `self.ex_dividend_value` 存的是 TWT49U 的「權值+息值」欄，單位是元／股，
        不是總額——呼叫端必須先乘上股數才能從總市值扣除，不能直接當總額減。
        """
        for lookahead in range(0, EX_DIVIDEND_LOOKBACK_DAYS + 1):
            fi = d_idx + lookahead
            if fi >= len(self.dates):
                break
            fd = self.dates[fi]
            if fd in self.ex_dividend_dates:
                return self.ex_dividend_value.get(fd, 0.0)
        return 0.0

    def run(
        self,
        onset_idx: int,
        end_idx: int,
        pledge_rate: float,
        principal_own_shares_value: float = 1_000_000.0,
        hard_unwind_idx: Optional[int] = None,
        winddown_start_idx: Optional[int] = None,
        winddown_end_idx: Optional[int] = None,
        proactive_cure_ratio: Optional[float] = None,
        principal_injections: Optional[list[tuple[int, float]]] = None,
    ) -> SimResult:
        """``hard_unwind_idx``：到達此營業日索引時，把借款加碼部位（loan_shares＋
        loan_reinvest_shares）一次全數賣出、還清 L，之後不槓桿續抱到 end_idx（早窗口
        質押的「2034 年底日曆式硬歸零」，findings §11）。``winddown_start_idx`` /
        ``winddown_end_idx``：在這段區間內逐月等額賣出借款加碼部位（keep-running 版的
        「簽約前 12 個月機會式減碼」出場代理）。``proactive_cure_ratio``：主動補繳門檻
        ——每日若擔保維持率低於此值，先用現金墊到此門檻（在 130% 追繳線之前就補），
        墊入現金計入維持率分子、期末歸還權益（等同 2026-09-01 建議的「275-300% 現金
        地板」）。``principal_injections``：``[(injection_idx, cash_amount), ...]``
        （findings §13，小本金早期啟動情境）——在指定交易日索引當天，用現金按當日
        收盤價加買同標的股，計入擔保品（``own_shares``／``liquidated_shares_left``
        同步增加），不影響已在 t0 鎖定的借款本金 ``L``（借款金額固定、質押率隨本金
        成長自然下滑，即既定口徑）。四者皆 None ⇒ 與第五輪逐位元相同（新增值
        全為 0.0，``x + 0.0 == x``）。
        """
        onset_date = self.dates[onset_idx]
        end_date = self.dates[end_idx]
        d0 = self.dates[onset_idx]
        p0 = self.prices[d0]
        p0_loan = self._loan_price(d0)  # 借款標的 Y 的 t0 價（Y==抵押品時等於 p0）

        # own_shares＝質押擔保品（維持率計算基礎，數量固定於 t0）；
        # loan_shares＝借來的錢在 t0 買入的加碼部位（不併入擔保品，不影響維持率
        # 分母/分子）。2026-09-09 overlay 輪起：Y 可以是與抵押品不同的標的，故用
        # Y 的 t0 價換算股數、之後用 Y 的價格序列 mark-to-market。追繳處分時 Y 是否
        # 一起賣依 liquidation_mode（all：Y 全賣；partial：抵押品清不掉 L 才動用 Y）。
        own_shares = principal_own_shares_value / p0
        loan_principal = (
            pledge_rate * principal_own_shares_value if pledge_rate > 0 else 0.0
        )
        loan_shares = loan_principal / p0_loan if loan_principal > 0 else 0.0
        total_shares = own_shares + loan_shares

        # 本金分段成長（findings §13）：待處理的注資佇列，按交易日索引排序，
        # 逐日經過時彈出。onset_idx 當天已到期的注資（idx<=onset_idx）視為 t0
        # 就已到位，因此直接併入起始 own_shares/total_shares，不進佇列。
        pending_injections = sorted(principal_injections or [], key=lambda x: x[0])
        while pending_injections and pending_injections[0][0] <= onset_idx:
            _amt0 = pending_injections.pop(0)[1]
            if p0 > 0:
                _sh0 = _amt0 / p0
                own_shares += _sh0
                total_shares += _sh0

        result = SimResult(
            onset_date=onset_date,
            end_date=end_date,
            stock_no="",
            pledge_rate=pledge_rate,
            liquidation_mode=self.liquidation_mode,
        )
        if hard_unwind_idx is not None:
            result.exit_mode = "hard_zero"
        elif winddown_start_idx is not None:
            result.exit_mode = "winddown"

        if pledge_rate == 0.0 or loan_principal == 0.0:
            end_price = self.prices[end_date]
            # 2026-09-08 配息 bug 修正：無槓桿對照組同樣要把持有期間收到的配息現金
            # 加回期末權益，否則對照組本身就低估（total_shares==own_shares 此分支恆成立，
            # 因為 loan_principal==0 只在 pledge_rate==0 時發生，見上方賦值）。
            # 第五輪（§10）：reinvest_dividends=True 時配息買回更多同標的股（複利）。
            dividend_cash_unlevered = 0.0
            reinv_shares = 0.0
            running_shares = (
                own_shares  # 逐日累加注資（findings §13），供股利與回撤都用當日真實股數
            )
            max_eq = None
            max_dd = 0.0
            for i in range(onset_idx, end_idx + 1):
                d_i = self.dates[i]
                price_i = self.prices[d_i]
                while pending_injections and pending_injections[0][0] <= i:
                    _amt = pending_injections.pop(0)[1]
                    if price_i > 0:
                        running_shares += _amt / price_i
                if d_i in self.ex_dividend_dates:
                    dps = self.ex_dividend_value.get(d_i, 0.0)
                    if self.reinvest_dividends:
                        net = net_dividend_cash(
                            (running_shares + reinv_shares) * dps,
                            self.dividend_income_tax_rate,
                        )
                        if price_i > 0:
                            reinv_shares += net / price_i
                    else:
                        gross = running_shares * dps
                        dividend_cash_unlevered += net_dividend_cash(
                            gross, self.dividend_income_tax_rate
                        )
                # 無槓桿也要算真實 peak-to-trough 回撤，不能省略成 0（否則對照組看起來永遠零風險）
                eq_i = running_shares * price_i
                if max_eq is None or eq_i > max_eq:
                    max_eq = eq_i
                if max_eq and max_eq > 0:
                    dd_i = (max_eq - eq_i) / max_eq
                    if dd_i > max_dd:
                        max_dd = dd_i
            own_shares = running_shares  # 供下方 unlevered 計算用（已含全部注資）
            div_add = (
                reinv_shares * end_price
                if self.reinvest_dividends
                else dividend_cash_unlevered
            )
            equity = own_shares * end_price + div_add
            unlevered = own_shares * end_price + div_add
            result.final_equity = equity
            result.final_equity_unlevered = unlevered
            result.max_drawdown_equity = max_dd
            return result

        L = loan_principal  # 撥款當下鎖定，不隨市值變動（第 20 條第 5 項）
        term_start_idx = onset_idx
        term_deadline_idx = onset_idx  # 展延後動態更新
        rollovers_used = 0

        pending_cure_deadline_idx: Optional[int] = None
        pending_liquidation_idx: Optional[int] = None
        in_cured_grace = False  # 補繳窗期滿但回升至>=130%、第3日暫不處分的狀態
        default_start_idx: Optional[int] = (
            None  # 第 26 條違約金起算日（補繳窗期滿之日）
        )

        max_equity = None
        max_drawdown = 0.0
        total_interest = 0.0
        dividend_cash = (
            0.0  # 2026-09-08 bug 修正：配息現金流累加（見 findings 第 8 節）
        )
        dividend_cash_unlevered = 0.0  # 反事實「從未槓桿」基準線的配息累加
        prev_idx = onset_idx
        liquidated_shares_left = own_shares  # 只有「擔保品」本身會被監控/處分
        loan_sale_cash = (
            0.0  # 追繳處分時賣出借款標的 Y 部位換得的現金（見 liquidation_mode）
        )

        # 配息再投入複利分支的股數累加（findings §10）。reinvest_dividends=False
        # 時三者恆為 0.0，下面 _div_equity / _unlev_div 走「+ dividend_cash」原路徑，
        # 與第四輪逐位元相同（x + 0.0 == x，dividend_cash 非 NaN/非 −0.0）。
        own_reinvest_shares = 0.0
        loan_reinvest_shares = 0.0
        own_reinvest_shares_unlev = 0.0

        # 早窗口質押組態（findings §11）。三個出場參數皆 None 時 posted_cash 恆 0.0、
        # 出場分支永不進入，下面每處 `+ posted_cash` == `+ 0.0`，與第五輪逐位元相同。
        posted_cash = 0.0  # proactive_cure_ratio 墊入的現金（計入維持率分子）
        proactive_topup_peak = 0.0
        wd_captured = False  # winddown 起點的借款腿股數是否已擷取
        wd_loan_shares_0 = 0.0
        wd_reinvest_shares_0 = 0.0
        wd_L_0 = 0.0
        wd_total_proceeds = 0.0
        wd_prev_month: Optional[str] = None
        wd_months_done = 0
        WD_MONTHS = 12

        def _div_equity(price_c: float, d_c: str) -> float:
            """槓桿路徑期末/當期權益裡「配息貢獻」項：純現金版回 dividend_cash；
            複利版回再投入股的 mark-to-market（擔保品腳用 price_c、Y 腳用 Y 收盤價）。"""
            if not self.reinvest_dividends:
                return dividend_cash
            return (
                dividend_cash
                + own_reinvest_shares * price_c
                + loan_reinvest_shares * self._loan_price(d_c)
            )

        def _unlev_div(price_c: float) -> float:
            """不槓桿反事實基準的配息貢獻項：純現金版 dividend_cash_unlevered；
            複利版 own_reinvest_shares_unlev 的 mark-to-market。"""
            if not self.reinvest_dividends:
                return dividend_cash_unlevered
            return own_reinvest_shares_unlev * price_c

        # 動態展延到期日（以「交易日數」近似 6 個月 ≈ 126 個交易日）
        APPROX_TRADING_DAYS_PER_TERM = 126
        term_deadline_idx = min(term_start_idx + APPROX_TRADING_DAYS_PER_TERM, end_idx)

        for i in range(onset_idx, end_idx + 1):
            d = self.dates[i]
            price = self.prices[d]
            if i > onset_idx:
                days_elapsed = (
                    date.fromisoformat(d) - date.fromisoformat(self.dates[prev_idx])
                ).days
            else:
                days_elapsed = 0
            if L > 0 and i > onset_idx:
                daily_rate = self.annual_rate / 365.0
                total_interest += L * daily_rate * days_elapsed
            prev_idx = i

            # 本金分段成長（findings §13）：注資買同標的股，計入擔保品
            # own_shares/liquidated_shares_left（維持率分子跟著漲），不動 L
            # （借款金額固定於 t0，質押率隨本金成長自然下滑，既定口徑）。
            while pending_injections and pending_injections[0][0] <= i:
                _amt = pending_injections.pop(0)[1]
                if price > 0:
                    _sh = _amt / price
                    own_shares += _sh
                    liquidated_shares_left += _sh

            # 配息現金流入袋（2026-09-08 bug 修正）：除息日當天用「持有股數 × 每股
            # 息值」累加現金，擔保品部位（liquidated_shares_left）與借款加碼部位
            # （loan_shares）分開累加，因為前者可能因處分而減少、後者依模型設計恆定
            # 不被賣出。用當日開盤前持股數（即今天任何處分動作發生前的數量），
            # 與 haircut 邏輯（只影響維持率分子的前瞻性扣減，不影響實際現金）互不重疊。
            # 2026-09-09 第四輪：配息改「稅後」現金。三邊各自依自己該筆給付金額
            # （單一除息事件 × 當下持股數）判斷二代健保 2 萬門檻並套用（見
            # net_dividend_cash / findings 第 9 節）。
            # 第五輪（§10）：reinvest_dividends=True 時，稅後配息在除息日當天用對應
            # 標的收盤價買回更多股（含「已再投入股」也一起計息＝真正的複利），
            # 取代純累積現金。買回股不進維持率分子/分母（比照 loan 腳），只在權益
            # mark-to-market。flag 關閉時走原本的 dividend_cash 累加，逐位元不變。
            tax_rate = self.dividend_income_tax_rate
            if d in self.ex_dividend_dates:
                div_per_share = self.ex_dividend_value.get(d, 0.0)
                if self.reinvest_dividends:
                    own_net = net_dividend_cash(
                        (liquidated_shares_left + own_reinvest_shares) * div_per_share,
                        tax_rate,
                    )
                    if price > 0:
                        own_reinvest_shares += own_net / price
                    u_net = net_dividend_cash(
                        (own_shares + own_reinvest_shares_unlev) * div_per_share,
                        tax_rate,
                    )
                    if price > 0:
                        own_reinvest_shares_unlev += u_net / price
                    if self.loan_prices is None:
                        l_net = net_dividend_cash(
                            (loan_shares + loan_reinvest_shares) * div_per_share,
                            tax_rate,
                        )
                        if price > 0:
                            loan_reinvest_shares += l_net / price
                else:
                    dividend_cash += net_dividend_cash(
                        liquidated_shares_left * div_per_share, tax_rate
                    )
                    # final_equity_unlevered 是「假設從未槓桿」的反事實比較基準，
                    # own_shares 在該反事實裡從不因追繳處分而減少，故用固定值另外累加。
                    dividend_cash_unlevered += net_dividend_cash(
                        own_shares * div_per_share, tax_rate
                    )
                    if self.loan_prices is None:
                        # Y == 抵押品標的：借款加碼部位配息用同一筆除息事件
                        dividend_cash += net_dividend_cash(
                            loan_shares * div_per_share, tax_rate
                        )
            if self.loan_prices is not None:
                # 借款標的 Y ≠ 抵押品：Y 用自己的除息序列入息
                loan_div_ps = self._loan_div_per_share(d)
                if loan_div_ps:
                    if self.reinvest_dividends:
                        lp = self._loan_price(d)
                        l_net = net_dividend_cash(
                            (loan_shares + loan_reinvest_shares) * loan_div_ps,
                            tax_rate,
                        )
                        if lp > 0:
                            loan_reinvest_shares += l_net / lp
                    else:
                        dividend_cash += net_dividend_cash(
                            loan_shares * loan_div_ps, tax_rate
                        )

            if result.term_expired_forced_sale:
                break

            if result.liquidated:
                # 已處分：debt 已依處分結果結清（或留有 L_after 餘額），
                # 剩餘擔保品（partial 模式）繼續隨市價波動，不再觸發新一輪追繳流程。
                equity_now = (
                    liquidated_shares_left * price
                    - L
                    + loan_shares * self._loan_price(d)
                    + loan_sale_cash
                    + posted_cash
                    + _div_equity(price, d)
                )
                if max_equity is None or equity_now > max_equity:
                    max_equity = equity_now
                if max_equity and max_equity > 0:
                    dd = (max_equity - equity_now) / max_equity
                    if dd > max_drawdown:
                        max_drawdown = dd
                continue

            # --- 早窗口質押出場（findings §11）：硬歸零 / 簽約前逐月減碼 ---
            if (
                hard_unwind_idx is not None
                and i >= hard_unwind_idx
                and not result.hard_unwound
                and L > 0.0
            ):
                lp_exit = self._loan_price(d)
                leg_val = (loan_shares + loan_reinvest_shares) * lp_exit
                result.loan_principal_at_exit = L
                result.loan_leg_value_at_exit = leg_val
                result.exit_realized_loss = leg_val < L - 1e-6 * max(1.0, L)
                applied = min(leg_val, L)
                loan_sale_cash += leg_val - applied
                residual = L - applied
                if residual > 0.0 and price > 0.0:
                    # 借款腿賣光仍不足清償 ⇒ 賣擔保品補足（實務上由專款補繳 buffer 承接）
                    liquidated_shares_left -= residual / price
                    result.collateral_sold_to_cover_exit = residual
                L = 0.0
                loan_shares = 0.0
                loan_reinvest_shares = 0.0
                result.hard_unwound = True
                result.exit_date = d

            if (
                winddown_start_idx is not None
                and i >= winddown_start_idx
                and (loan_shares > 1e-12 or loan_reinvest_shares > 1e-12)
            ):
                if not wd_captured:
                    wd_loan_shares_0 = loan_shares
                    wd_reinvest_shares_0 = loan_reinvest_shares
                    wd_L_0 = L
                    wd_prev_month = d[:7]
                    wd_captured = True  # 起點當月不賣，之後每逢新月賣 1/WD_MONTHS
                at_end = winddown_end_idx is not None and i >= winddown_end_idx
                if d[:7] != wd_prev_month or at_end:
                    wd_prev_month = d[:7]
                    wd_months_done += 1
                    lp_wd = self._loan_price(d)
                    if at_end or wd_months_done >= WD_MONTHS:
                        sell_loan, sell_reinv = loan_shares, loan_reinvest_shares
                    else:
                        sell_loan = min(loan_shares, wd_loan_shares_0 / WD_MONTHS)
                        sell_reinv = min(
                            loan_reinvest_shares, wd_reinvest_shares_0 / WD_MONTHS
                        )
                    proceeds = (sell_loan + sell_reinv) * lp_wd
                    wd_total_proceeds += proceeds
                    loan_shares -= sell_loan
                    loan_reinvest_shares -= sell_reinv
                    applied = min(proceeds, L)
                    loan_sale_cash += proceeds - applied
                    L = max(0.0, L - applied)
                    if loan_shares <= 1e-12 and loan_reinvest_shares <= 1e-12:
                        result.hard_unwound = True
                        result.exit_date = d
                        result.loan_principal_at_exit = wd_L_0
                        result.loan_leg_value_at_exit = wd_total_proceeds
                        result.exit_realized_loss = (
                            wd_total_proceeds < wd_L_0 - 1e-6 * max(1.0, wd_L_0)
                        )

            mv = liquidated_shares_left * (price - self._haircut_per_share(i))
            maint_ratio = mv / L if L > 0 else float("inf")
            if L > 0.0:
                result.min_maintenance_ratio = min(
                    result.min_maintenance_ratio, maint_ratio
                )

            # --- 主動補繳門檻（findings §11）：跌破 proactive_cure_ratio 就用現金墊到該線 ---
            if proactive_cure_ratio is not None:
                if L > 0.0:
                    posted_cash = max(0.0, proactive_cure_ratio * L - mv)
                    proactive_topup_peak = max(proactive_topup_peak, posted_cash)
                    mv = mv + posted_cash
                    maint_ratio = mv / L
                else:
                    posted_cash = 0.0

            equity_now = (
                liquidated_shares_left * price
                - L
                + loan_shares * self._loan_price(d)
                + loan_sale_cash
                + posted_cash
                + _div_equity(price, d)
            )
            if max_equity is None or equity_now > max_equity:
                max_equity = equity_now
            if max_equity and max_equity > 0:
                dd = (max_equity - equity_now) / max_equity
                if dd > max_drawdown:
                    max_drawdown = dd

            # --- 到期展延／借新還舊／強制平倉（第 4 條，獨立於追繳維持率狀態機）---
            if self.allow_rollover:
                if i >= term_deadline_idx:
                    if rollovers_used < MAX_ROLLOVERS:
                        # 6 個月一般展延：整戶維持率門檻 130%（broker practice, medium confidence）
                        can_continue = maint_ratio >= MARGIN_CALL_RATIO
                        rollovers_used += 1
                    else:
                        # 18 個月到期須辦「借新還舊」（新合約）：單筆維持率門檻 166.67%（§5.2）。
                        # 只要通過，視為新合約重新起算 18 個月（rollovers_used 歸零），
                        # 這是長期槓桿策略「持續質押」的常態，不是異常；只有不通過才強制平倉。
                        can_continue = maint_ratio >= CURE_TARGET_RATIO
                        rollovers_used = 0
                    if can_continue:
                        term_start_idx = i
                        term_deadline_idx = min(
                            i + APPROX_TRADING_DAYS_PER_TERM, end_idx
                        )
                    else:
                        # 維持率不足以展延／借新還舊 -> 次一交易日強制賣出
                        ni = min(i + 1, end_idx)
                        nd = self.dates[ni]
                        nprice = self.prices[nd]
                        proceeds = liquidated_shares_left * nprice
                        # 扣除累積利息 total_interest：這是已發生但尚未在其他地方
                        # 從權益中減去的現金流出（2026-09-08 bug 修正，見 findings）。
                        remaining_cash = (
                            proceeds
                            - L
                            + loan_shares * self._loan_price(nd)
                            + loan_sale_cash
                            + posted_cash
                            - total_interest
                            + _div_equity(nprice, nd)
                        )
                        result.term_expired_forced_sale = True
                        result.final_equity = remaining_cash
                        result.final_equity_unlevered = (
                            own_shares * nprice + _unlev_div(nprice)
                        )
                        result.max_drawdown_equity = max_drawdown
                        result.total_interest_paid = total_interest
                        result.n_margin_calls = len(result.margin_call_events)
                        result.proactive_topup_peak = proactive_topup_peak
                        return result
            else:
                if i >= term_deadline_idx:
                    ni = min(i + 1, end_idx)
                    nd = self.dates[ni]
                    nprice = self.prices[nd]
                    proceeds = liquidated_shares_left * nprice
                    # 扣除累積利息 total_interest（同上，2026-09-08 bug 修正）。
                    remaining_cash = (
                        proceeds
                        - L
                        + loan_shares * self._loan_price(nd)
                        + loan_sale_cash
                        + posted_cash
                        - total_interest
                        + _div_equity(nprice, nd)
                    )
                    result.term_expired_forced_sale = True
                    result.final_equity = remaining_cash
                    result.final_equity_unlevered = own_shares * nprice + _unlev_div(
                        nprice
                    )
                    result.max_drawdown_equity = max_drawdown
                    result.total_interest_paid = total_interest
                    result.n_margin_calls = len(result.margin_call_events)
                    result.proactive_topup_peak = proactive_topup_peak
                    return result

            # --- 追繳/處分狀態機（第 20、25 條）---
            if pending_liquidation_idx is not None and i >= pending_liquidation_idx:
                if maint_ratio < MARGIN_CALL_RATIO:
                    ev = result.margin_call_events[-1]
                    ev.liquidated = True
                    ev.liquidation_date = d
                    if self.liquidation_mode == "all":
                        sell_shares = liquidated_shares_left
                    else:
                        # partial：只賣到足以清償融通本金（common broker practice: stop selling once the loan is covered）
                        sell_shares = min(
                            liquidated_shares_left,
                            L / price if price > 0 else liquidated_shares_left,
                        )
                    proceeds = sell_shares * price
                    liquidated_shares_left -= sell_shares
                    overdue_days = (
                        i - default_start_idx if default_start_idx is not None else 0
                    )
                    penalty = (
                        L
                        * self.annual_rate
                        * DEFAULT_PENALTY_RATE_MULT
                        * overdue_days
                        / 365.0
                    )
                    L = max(
                        0.0, L - proceeds
                    )  # 已處分清償後的剩餘本金（partial 模式常為 0）
                    # 借款買入標的 Y 部位的處分（Scope 2，2026-09-09）：
                    #   all     -> Y 全數賣出，Y 收入先抵剩餘 L、餘額進現金
                    #   partial -> 只有抵押品賣光仍清不掉 L 時才動用 Y，且賣到 L 歸零為止
                    lp = self._loan_price(d)
                    if loan_shares > 0.0 and lp > 0.0:
                        if self.liquidation_mode == "all":
                            loan_proceeds = loan_shares * lp
                            loan_shares = 0.0
                            applied = min(loan_proceeds, L)
                            L = max(0.0, L - applied)
                            loan_sale_cash += loan_proceeds - applied
                        elif L > 0.0:  # partial 且抵押品清償不足
                            sell_val = min(L, loan_shares * lp)
                            loan_shares -= sell_val / lp
                            L = max(0.0, L - sell_val)
                    result.liquidated = True
                    result.liquidation_date = d
                    result.default_penalty_paid = penalty
                    pending_liquidation_idx = None
                    # 不在此處提前 return：partial 模式剩餘擔保品之後仍隨市價波動，
                    # 交給迴圈其餘天數與迴圈後的收尾區塊算出真正的期末權益。
                    continue
                else:
                    # 第 3 日暫不處分（已回升 >=130%）
                    pending_liquidation_idx = None
                    in_cured_grace = True
                    result.margin_call_events[-1].cured = True
                    result.margin_call_events[-1].cure_date = d

            if pending_cure_deadline_idx is not None:
                if maint_ratio >= CURE_TARGET_RATIO and self.auto_cure:
                    result.margin_call_events[-1].cured = True
                    result.margin_call_events[-1].cure_date = d
                    pending_cure_deadline_idx = None
                elif i >= pending_cure_deadline_idx:
                    # 第3營業日開市起處分：通知日 D+0, D+1,D+2 補繳窗, D+3 處分
                    pending_liquidation_idx = (
                        i - CURE_WINDOW_DAYS + LIQUIDATION_START_BUSINESS_DAY
                    )
                    default_start_idx = i  # 補繳窗期滿之日起算逾期（第 26 條）
                    pending_cure_deadline_idx = None
                continue

            if in_cured_grace:
                if maint_ratio < MARGIN_CALL_RATIO:
                    # 嗣後再破130% -> 次一營業日起處分
                    pending_liquidation_idx = min(i + 1, end_idx)
                    default_start_idx = i
                    ev = MarginCallEvent(trigger_date=d, ratio_at_trigger=maint_ratio)
                    result.margin_call_events.append(ev)
                    in_cured_grace = False
                continue

            if maint_ratio < MARGIN_CALL_RATIO:
                ev = MarginCallEvent(trigger_date=d, ratio_at_trigger=maint_ratio)
                result.margin_call_events.append(ev)
                pending_cure_deadline_idx = i + CURE_WINDOW_DAYS

        end_d = self.dates[min(end_idx, len(self.dates) - 1)]
        end_price = self.prices[end_d]
        final_mv = liquidated_shares_left * end_price
        # 扣除累積利息 total_interest：2026-09-08 bug 修正——原本三處 final_equity
        # 賦值都漏了這一項，導致整條槓桿路徑「用免費的錢」跑出來（見 findings）。
        # 2026-09-08 bug 修正（第二輪）：加回持有期間收到的配息現金（見 findings 第 8 節）。
        # 2026-09-09：借款部位改用 Y 的期末價，並加回處分 Y 換得的現金 loan_sale_cash。
        result.final_equity = (
            final_mv
            - L
            - result.default_penalty_paid
            - total_interest
            + loan_shares * self._loan_price(end_d)
            + loan_sale_cash
            + posted_cash
            + _div_equity(end_price, end_d)
        )
        result.final_equity_unlevered = own_shares * end_price + _unlev_div(end_price)
        result.max_drawdown_equity = max_drawdown
        result.total_interest_paid = total_interest
        result.n_margin_calls = len(result.margin_call_events)
        result.proactive_topup_peak = proactive_topup_peak
        return result


# ---------------------------------------------------------------------------
# 前瞻漂移模式（2026-09-09 第五輪，findings §10）
# ---------------------------------------------------------------------------
