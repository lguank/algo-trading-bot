"""Test script for trading strategies"""

from datetime import datetime, timedelta

from src.data.data_fetcher import DataFetcher
from src.strategies.base_strategy import MovingAverageCrossoverStrategy


def test_moving_average_strategy():
    """Test the moving average crossover strategy"""
    print("🤖 Testing Moving Average Crossover Strategy")
    print("=" * 60)

    # Create strategy
    strategy = MovingAverageCrossoverStrategy(
        short_window=10, long_window=20, initial_capital=50000  # Shorter windows for faster testing
    )

    # Fetch historical data for backtesting
    print("\n📊 Fetching historical data...")
    fetcher = DataFetcher()

    symbols = ["AAPL", "GOOGL"]
    backtest_data = {}

    for symbol in symbols:
        data = fetcher.get_stock_data(symbol, period="6mo")  # 6 months
        if not data.empty:
            backtest_data[symbol] = data
            print(f"✅ Fetched {len(data)} days of {symbol} data")

    if not backtest_data:
        print("❌ No data fetched for backtesting")
        return

    # Run backtest
    print(f"\n🔄 Running backtest on {len(backtest_data)} symbols...")

    try:
        result = strategy.backtest(
            backtest_data, start_date=datetime.now() - timedelta(days=120), end_date=datetime.now()  # 4 months
        )

        print("\n✅ Backtest completed successfully!")

        # Test signal generation on recent data
        print("\n🧪 Testing signal generation...")
        for symbol, data in backtest_data.items():
            current_price = data["Close"].iloc[-1]

            mock_portfolio = {
                "cash": 25000,
                "total_value": 50000,
                "positions": {symbol: 50},  # Own 50 shares
                "current_prices": {symbol: current_price},
            }

            signal = strategy.generate_signal(symbol, data, current_price, mock_portfolio)

            print(f"📡 {symbol}: {signal.action} - {signal.reason}")
            print(f"   Confidence: {signal.confidence:.2f}, Quantity: {signal.quantity}")

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback

        traceback.print_exc()


def test_strategy_signals():
    """Test signal generation with mock data"""
    print("\n🧪 Testing Signal Generation with Mock Data")
    print("=" * 50)

    # Create simple test data
    dates = pd.date_range("2024-01-01", periods=60, freq="D")

    # Create mock price data with clear trend
    prices = [100]
    for i in range(59):
        # Simulate trending price data
        if i < 30:
            prices.append(prices[-1] + np.random.normal(0.5, 1))  # Upward trend
        else:
            prices.append(prices[-1] + np.random.normal(-0.3, 1))  # Downward trend

    mock_data = pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": [1000000] * 60,
        },
        index=dates,
    )

    strategy = MovingAverageCrossoverStrategy(5, 15, 10000)

    # Test signal generation
    current_price = mock_data["Close"].iloc[-1]

    mock_portfolio = {"cash": 5000, "total_value": 10000, "positions": {}, "current_prices": {"TEST": current_price}}

    signal = strategy.generate_signal("TEST", mock_data, current_price, mock_portfolio)

    print(f"📡 Generated signal: {signal.action}")
    print(f"🔍 Reason: {signal.reason}")
    print(f"🎯 Confidence: {signal.confidence:.2f}")
    print(f"📊 Quantity: {signal.quantity}")

    print("\n✅ Signal generation test complete!")


if __name__ == "__main__":
    # Import here to avoid circular imports
    import numpy as np
    import pandas as pd

    test_strategy_signals()
    test_moving_average_strategy()
