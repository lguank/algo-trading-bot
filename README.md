# Production-Grade Algorithmic Trading Engine 📈

An automated, systematic trading engine engineered to capture momentum and trend convexity across highly liquid equities. Built entirely from scratch in Python, this project bridges applied computing and quantitative finance via vectorized technical analytics, persistent state tracking, and rigorous double-entry portfolio accounting.


---

### 📊 Quantitative Performance (1-Year Backtest: May 2025 – May 2026)
Tested across a diversified universe (`AAPL`, `GOOGL`, `MSFT`, `TSLA`, `SPY`) strictly accounting for real-world transaction costs (**$9.99 flat fee per trade**) and capital depletion limits:

| Performance Metric | Result | Quantitative Implication |
| :--- | :--- | :--- |
| **Initial Capital** | `$10,000.00` | Standard baseline allocation |
| **Final Portfolio NAV** | **`$13,156.25`** | Demonstrates clear positive expected value |
| **Total Absolute Return** | **`+31.56%`** | Significant outperformance vs. benchmark equities |
| **Annualized Sharpe Ratio** | **`1.479`** | Institutional-grade risk-adjusted return efficiency |
| **Maximum Drawdown** | **`-17.04%`** | Controlled capital preservation during regime shifts |
| **Payoff Profile** | Convex Alpha | Aggressively rides open winners while truncating losses |

> **Quant Note on Win Rate (12.5%):** The strategy exhibits highly convex trend-following characteristics. In institutional quant management, trend-following models typically run on low win rates; profitability is driven by an asymmetric payoff profile (maximizing the magnitude of unrealised gains while strictly limiting realised downside risk).

---

### 🧠 Core Architecture & Confluence Matrix
Rather than relying on isolated, lagging technical triggers, the engine executes decisions via a **weighted confluence voting matrix** analyzing 7 simultaneous indicators across 4 distinct market dimensions:

1. **Macro Trend Confirmation:** Moving Average Crossovers (`SMA`, `EMA`) to confirm broad regime direction.
2. **Momentum Dynamics:** Wilder's Smoothed `RSI` and `Stochastic Oscillators` to identify structural entry zones.
3. **Volume Microstructure:** On-Balance Volume (`OBV`) and Institutional **`VWAP`** to track institutional block accumulation.
4. **Volatility Sizing:** `Bollinger Bands` and Average True Range (`ATR`) to govern entries and prevent overbought exposure.
