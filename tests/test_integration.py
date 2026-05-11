# test_integration.py
# Run this last: python test_integration.py
# Purpose: test the full pipeline end to end with real market data

from datetime import datetime, timedelta

import pandas as pd

print("=" * 50)
print("INTEGRATION TEST - Full Pipeline")
print("=" * 50)

from src.data.data_fetcher import DataFetcher
from src.portfolio.portfolio_tracker import PortfolioTracker
from src.strategies.base_strategy import MovingAverageCrossoverStrategy
from src.strategies.multi_indicator_strategy import MultiIndicatorStrategy
from src.utils.helpers import format_currency

fetcher = DataFetcher()
portfolio = PortfolioTracker(initial_capital=100_000.0)
strategy = MultiIndicatorStrategy(buy_threshold=2.5, sell_threshold=-2.5)

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"    ✅ {name}")
        passed += 1
    else:
        print(f"    ❌ {name} {detail}")
        failed += 1


# ── Test 1: Data fetching ─────────────────────────────────────────
print("\n[1] Data Fetching")

data = fetcher.get_historical_data("AAPL", period="3mo", interval="1d")

check("Data is not None", data is not None)
check("Data is not empty", not data.empty)
check("Has required columns", all(c in data.columns for c in ["Open", "High", "Low", "Close", "Volume"]))
check("Has enough rows for strategy", len(data) >= 60, f"got {len(data)} rows")
check("No negative prices", (data["Close"] > 0).all())

print(f"    ℹ️  Fetched {len(data)} rows for AAPL")
print(f"    ℹ️  Date range: {data.index[0].date()} → {data.index[-1].date()}")
print(f"    ℹ️  Latest close: {format_currency(float(data['Close'].iloc[-1]))}")


# ── Test 2: Signal generation ─────────────────────────────────────
print("\n[2] Signal Generation")

portfolio_info = {
    "cash": 100_000.0,
    "total_value": 100_000.0,
    "positions": {},
    "current_prices": {},
}

signal = strategy.generate_signal(
    symbol="AAPL",
    data=data,
    current_price=float(data["Close"].iloc[-1]),
    portfolio_info=portfolio_info,
)

check("Signal is not None", signal is not None)
check("Signal has valid action", signal.action in ["BUY", "SELL", "HOLD"], f"got '{signal.action}'")
check("Signal has valid confidence", 0.0 <= signal.confidence <= 1.0, f"got {signal.confidence}")
check("Signal has positive price", signal.price > 0, f"got {signal.price}")
check("Signal has non-negative quantity", signal.quantity >= 0, f"got {signal.quantity}")
check("Signal has reason text", len(signal.reason) > 0)

print(f"    ℹ️  Signal: {signal.action} | " f"Confidence: {signal.confidence:.0%} | " f"Reason: {signal.reason}")


# ── Test 3: get_full_analysis (web API method) ────────────────────
print("\n[3] Full Analysis (Web API)")

analysis = strategy.get_full_analysis("AAPL", data)

check("Analysis returns dict", isinstance(analysis, dict))
check("Has signal key", "signal" in analysis)
check("Has score key", "score" in analysis)
check("Has votes key", "votes" in analysis)
check("Has price_history key", "price_history" in analysis)
check("Has stop_loss key", "stop_loss" in analysis)
check("Price history has 30 points", len(analysis.get("price_history", [])) == 30)
check("Signal is valid string", analysis.get("signal") in ["BUY", "SELL", "HOLD"])

print(f"    ℹ️  Signal: {analysis['signal']} | Score: {analysis['score']}")
print(f"    ℹ️  Stop loss: {format_currency(analysis['stop_loss'])}")
print(f"    ℹ️  Votes cast: {len(analysis['votes'])}")


# ── Test 4: Backtesting ───────────────────────────────────────────
print("\n[4] Backtesting")

# Fetch data for multiple symbols
backtest_data = {}
for symbol in ["AAPL", "MSFT"]:
    d = fetcher.get_historical_data(symbol, period="1y", interval="1d")
    if d is not None and not d.empty:
        backtest_data[symbol] = d
        print(f"    ℹ️  Loaded {len(d)} rows for {symbol}")

check("Got data for backtest", len(backtest_data) > 0)

if backtest_data:
    try:
        result = strategy.backtest(backtest_data)

        check("Backtest returned result", result is not None)
        check("Has strategy name", len(result.strategy_name) > 0)
        check("Final value is positive", result.final_portfolio_value > 0, f"got {result.final_portfolio_value}")
        check("Has portfolio history", len(result.portfolio_history) > 0)
        check("Win rate between 0-100", 0 <= result.win_rate_pct <= 100, f"got {result.win_rate_pct}")
        check("Profit factor non-negative", result.profit_factor >= 0, f"got {result.profit_factor}")
        check("to_dict() works", isinstance(result.to_dict(), dict))

        print(f"    ℹ️  Return: {result.total_return_pct:+.2f}%")
        print(f"    ℹ️  Trades: {result.total_trades}")
        print(f"    ℹ️  Sharpe: {result.sharpe_ratio:.3f}")

    except Exception as e:
        print(f"    ❌ Backtest crashed: {e}")
        import traceback

        traceback.print_exc()
        failed += 1


# ── Test 5: Compare both strategies ──────────────────────────────
print("\n[5] Strategy Comparison")

if backtest_data:
    try:
        ma_strategy = MovingAverageCrossoverStrategy(short_window=20, long_window=50)
        ma_result = ma_strategy.backtest(backtest_data)
        mi_result = result  # already computed above

        print(f"\n    Strategy Comparison:")
        print(f"    {'Metric':<25} {'MA Cross':>12} {'Multi-Indicator':>16}")
        print(f"    {'─'*55}")
        print(
            f"    {'Total Return':<25} "
            f"{ma_result.total_return_pct:>+11.2f}% "
            f"{mi_result.total_return_pct:>+15.2f}%"
        )
        print(f"    {'Sharpe Ratio':<25} " f"{ma_result.sharpe_ratio:>12.3f} " f"{mi_result.sharpe_ratio:>16.3f}")
        print(
            f"    {'Max Drawdown':<25} "
            f"{ma_result.max_drawdown_pct:>11.2f}% "
            f"{mi_result.max_drawdown_pct:>15.2f}%"
        )
        print(f"    {'Total Trades':<25} " f"{ma_result.total_trades:>12} " f"{mi_result.total_trades:>16}")
        print(f"    {'Win Rate':<25} " f"{ma_result.win_rate_pct:>11.1f}% " f"{mi_result.win_rate_pct:>15.1f}%")
        print(f"    {'Profit Factor':<25} " f"{ma_result.profit_factor:>12.2f} " f"{mi_result.profit_factor:>16.2f}")

        check("Both backtests completed", True)

    except Exception as e:
        print(f"    ❌ Comparison failed: {e}")
        failed += 1


print(f"\n{'=' * 50}")
print(f"RESULTS: {passed} passed | {failed} failed")
print("=" * 50)
