# test_smoke.py
# Run this first: python test_smoke.py
# Purpose: catch any import errors or broken connections before anything else

print("=" * 50)
print("SMOKE TEST - Checking all imports and connections")
print("=" * 50)

# Test 1: Imports
print("\n[1/5] Testing imports...")
try:
    from config.config import get_config
    from src.data.data_fetcher import DataFetcher
    from src.portfolio.portfolio_tracker import PortfolioTracker
    from src.strategies.base_strategy import BacktestResult, BaseStrategy, MovingAverageCrossoverStrategy, Signal
    from src.strategies.multi_indicator_strategy import IndicatorVote, MultiIndicatorStrategy, SignalReport
    from src.utils.helpers import (
        calculate_atr,
        calculate_bollinger_bands,
        calculate_exponential_moving_average,
        calculate_macd,
        calculate_max_drawdown,
        calculate_moving_average,
        calculate_obv,
        calculate_profit_factor,
        calculate_rsi,
        calculate_sharpe_ratio,
        calculate_stochastic_oscillator,
        calculate_vwap,
        calculate_win_rate,
        format_currency,
        format_percentage,
    )

    print("    ✅ All imports successful")
except ImportError as e:
    print(f"    ❌ Import failed: {e}")
    print("    Fix this before running any other tests")
    exit(1)

# Test 2: Config loads correctly
print("\n[2/5] Testing config...")
try:
    config = get_config()
    print(f"    ✅ Config loaded | ENV: {config.ENV}")
    print(f"    ✅ Symbols: {config.SYMBOLS}")
    print(f"    ✅ Capital: {format_currency(config.INITIAL_CAPITAL)}")
except Exception as e:
    print(f"    ❌ Config error: {e}")

# Test 3: Strategy instantiates
print("\n[3/5] Testing strategy instantiation...")
try:
    ma_strategy = MovingAverageCrossoverStrategy(short_window=20, long_window=50)
    print(f"    ✅ MA Strategy created: {ma_strategy.name}")

    mi_strategy = MultiIndicatorStrategy(buy_threshold=2.5, sell_threshold=-2.5)
    print(f"    ✅ Multi-Indicator Strategy created: {mi_strategy.name}")
except Exception as e:
    print(f"    ❌ Strategy error: {e}")

# Test 4: Portfolio tracker instantiates
print("\n[4/5] Testing portfolio tracker...")
try:
    portfolio = PortfolioTracker(initial_capital=100_000.0)
    print(f"    ✅ Portfolio created | Cash: {format_currency(portfolio.cash)}")
except Exception as e:
    print(f"    ❌ Portfolio error: {e}")

# Test 5: Data fetcher connects
print("\n[5/5] Testing data connection (fetching AAPL)...")
try:
    fetcher = DataFetcher()
    data = fetcher.get_historical_data("AAPL", period="5d", interval="1d")
    if data is not None and not data.empty:
        print(f"    ✅ Data fetched | Rows: {len(data)}")
        print(f"    ✅ Latest close: {format_currency(float(data['Close'].iloc[-1]))}")
    else:
        print("    ❌ Data came back empty")
except Exception as e:
    print(f"    ❌ Data fetch error: {e}")

print("\n" + "=" * 50)
print("SMOKE TEST COMPLETE")
print("=" * 50)
