# tw-pledge-sim

A historical-path simulator for Taiwan securities-backed loans (the "unrestricted-purpose loan" a broker grants against pledged listed shares or ETFs). It replays real price history through the loan's margin mechanics and counts how often a borrower would have been force-sold.

Standard library only, Python 3.10+.

## The problem

Pledging a stock portfolio to borrow money and buy more of the same asset raises expected return and adds a tail risk: a drawdown pushes the maintenance ratio under the call line, and the broker sells the collateral at the bottom. Most calculators answer with a normal-distribution Monte Carlo or a single "how far can the price fall" figure. Neither says how often a borrower who started in an arbitrary month of the past would actually have been sold out, nor which start years were the dangerous ones.

## Method

- Real paths. For every month in the sample, start a loan on that day and replay daily closes for the holding period (5, 10 or 15 years in the defaults). Each start is one observation; the output is the share of starts that ended in forced disposal, plus the distribution of final equity relative to not borrowing.
- Price data from TWSE public endpoints, cached on disk. Splits are detected from gap jumps that match an integer ratio, and prices and per-share ex-rights values before the split are divided by the ratio so that dividend cash is not overstated.
- Ex-dividend handling. Dividends are paid as cash (optionally reinvested), after the supplementary health-insurance premium and a configurable income-tax rate. The collateral value is reduced per share in the 6 business days before an ex-date, as the regulation allows.
- Margin mechanics follow the regulation: loan value 60% of market value (Art. 16), maintenance ratio computed daily on the drawn amount, call below 130% with 2 business days to cure back to 166.67% (Art. 20), disposal from the third business day (Art. 25), a default penalty of 10% of the loan rate (Art. 26), and the 6-month term with two extensions, after which a new contract needs 166.67% (Art. 4).
- Household overlay. A monthly layer where salary pays living costs and a fixed mortgage first, then pledge interest (salary surplus, then a cash buffer, then dividends, then selling collateral units). Calls are cured from the buffer. `solve_min_buffer` bisects for the smallest starting buffer that avoids every historical disposal.
- Break-even rate. `solve_breakeven_rate` bisects the constant loan rate at which the median (or the win rate) of leveraged-versus-unleveraged final equity crosses parity.

Confidence is layered. Everything in `tw_pledge_sim/regulation.py` is statute, with the article noted beside each constant, and brokers can be stricter but not looser. Everything that varies by institution sits in `BrokerProfile` (loan rate, whether disposal sells everything or stops once the loan is covered, whether cures are automatic, whether extensions are granted). The default profile is `generic`. Those fields describe common practice, which is a weaker claim than the law, and you should replace them with your own broker's published terms.

## Why real paths

A normal model with constant volatility can understate how clustered crashes are, and a call is a path event: what matters is the lowest ratio reached on any day, plus whether the cure window and the 18-month renewal land near a trough. Replaying history keeps volatility clustering, gaps, and the order of dividends and drawdowns intact. The cost is a short sample (the earliest ETF starts in 2003), so the result is a count of what happened, with few independent crises behind it.

## Layout

    src/tw_pledge_sim/
        regulation.py   statutory constants with article numbers
        broker.py       BrokerProfile (generic defaults)
        data.py         TWSE fetch + cache, StockInputs
        adjust.py       split detection, split and dividend adjustment
        engine.py       PledgeSimulator (one security, one loan, one path)
        rolling.py      rolling-start scan, break-even solver
        household.py    monthly cash-flow overlay, disposal statistics, buffer solver
        synthetic.py    seeded synthetic prices for demos and offline tests
    examples/demo.py    one-command demo with a result table (output in demo_output.txt)
    tests/              pytest suite, offline

## Running

    pip install -e ".[dev]"
    python examples/demo.py          # offline, synthetic prices, no network
    python examples/demo.py --live   # downloads TWSE prices into ./data (network, a few minutes)
    pytest

The cache directory defaults to `./data` and can be set with `TW_PLEDGE_SIM_DATA`. It is git-ignored; raw exchange data is re-fetched, never committed. All household numbers in the demo are invented (capital 1,000,000, monthly income 50,000).

`tests/test_consistency.py` compares this package with the reference engine it was extracted from, using a fixture of values computed once on deterministic synthetic data (rolling-start scan, break-even solver, household matrix, buffer solver). The tests run offline.

## Limitations

- Daily closes only. No intraday trading, bid-ask spread or transaction costs.
- Overlapping rolling windows are strongly correlated, so the number of starts is far larger than the number of independent episodes.
- The loan rate is constant within a run, while real broker rates are repriced periodically. The break-even rate is a statement about a constant rate.
- The household overlay steps monthly and applies a one-pass approximation: after a disposal it does not re-solve call timing within the month. Margin calls are checked at month end there, and with a coarser calendar than the daily engine.
- Taxes are reduced to the supplementary premium and one effective income-tax rate.
- The regulation text is modelled as of the 2026-08 version. Check the current text before relying on any threshold.
- The forward-return drift option of the original research code was left out of this package.
- Output describes a model and gives no investment advice.

## 中文簡介

問題：以股票或 ETF 質押借款、再買進同一標的，期望報酬變高，但下跌時維持率可能跌破追繳線而被強制處分。常見試算多半用常態分配的蒙地卡羅，或只給一個「最多跌多少」的靜態數字。

方法：把歷史上每個月都當成一個起點，沿真實日收盤價回放整個持有期，統計有多少起點會被強制處分、期末淨值相對於不借款的分布，以及損益兩平利率。價格取自證交所公開資料，並處理分割與除權息調整。追繳、補繳、處分、18 個月展延等條件依《證券商辦理不限用途款項借貸業務操作辦法》建模，條號寫在 `regulation.py`；各家券商不同的部分（利率、處分範圍、展延核准）集中在 `BrokerProfile`，預設為 generic。家計現金流 overlay 以月為單位，加入薪資、房貸、生活費與現金緩衝。

為什麼用真實路徑：追繳取決於路徑上最低的那一天，而且補繳期限與 18 個月續約是否落在谷底附近，常態假設抓不到。代價是樣本短（最早 2003 年），獨立的危機事件不多。

已知限制：只用日收盤價、利率在每次模擬內固定、月度 overlay 為單向近似、稅負簡化、法規版本為 2026 年 8 月，請以現行條文為準。

怎麼跑：`pip install -e ".[dev]"` 之後，`python examples/demo.py` 離線跑合成資料示範，`--live` 才會連網抓取證交所資料；`pytest` 離線執行。示範中的金額全為虛構。
