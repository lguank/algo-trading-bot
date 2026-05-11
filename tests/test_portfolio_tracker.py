"""Test script for PortfolioTracker module"""

from src.data.data_fetcher import DataFetcher
from src.portfolio.portfolio_tracker import PortfolioTracker


def test_portfolio_basic_operations():
    """Test basic portfolio operations"""
    print("🧪 Testing Portfolio Tracker")
    print("=" * 50)

    # Create portfolio with $50,000
    portfolio = PortfolioTracker(50000, "data/test_portfolio.json")

    print(f"✅ Initial portfolio created")
    portfolio.print_portfolio_summary()

    # Test buying stocks
    print("\n📈 Testing BUY orders...")
    success1 = portfolio.add_trade("AAPL", "BUY", 100, 150.00, 5.00)
    success2 = portfolio.add_trade("GOOGL", "BUY", 20, 2500.00, 5.00)

    if success1 and success2:
        print("✅ Buy orders executed successfully")
        portfolio.print_portfolio_summary()

    # Update with current prices
    print("\n💰 Updating current prices...")
    fetcher = DataFetcher()
    current_prices = fetcher.get_latest_prices(["AAPL", "GOOGL"])

    if current_prices:
        portfolio.update_prices(current_prices)
        print("✅ Prices updated")
        portfolio.print_portfolio_summary()

    # Test selling
    print("\n📉 Testing SELL order...")
    success3 = portfolio.add_trade("AAPL", "SELL", 50, 155.00, 5.00)

    if success3:
        print("✅ Sell order executed successfully")
        portfolio.print_portfolio_summary()

    # Show trade history
    print("\n📋 Trade History:")
    trade_history = portfolio.get_trade_history()
    print(trade_history)

    print("\n🎉 Portfolio Tracker tests complete!")


if __name__ == "__main__":
    test_portfolio_basic_operations()
