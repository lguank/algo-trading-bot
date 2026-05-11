"""
Main Application Entry Point for the Algorithmic Trading Bot.

This module orchestrates all components:
1. DataFetcher: Gets current market data
2. PortfolioTracker: Manages positions and cash
3. TradingStrategy: Generates buy/sell signals
4. Risk Management: Validates and controls trades

DESIGN PHILOSOPHY:
- Modular: Each component is independent and testable
- Configurable: Strategy parameters can be easily changed
- Defensive: Every operation is validated before execution
- Professional: Clear logging and reporting

FINANCIAL CONCEPTS DEMONSTRATED:
- Portfolio management and position tracking
- Technical analysis with moving averages
- Risk management with position sizing
- Performance measurement and backtesting
"""

import time
import traceback
from datetime import datetime
from math import e
from pickle import NONE
from tokenize import ContStr
from typing import Dict, Optional
from unittest import result

import pandas as pd
from numpy import true_divide

# Import our custom modules
from config import config
from config.config import get_config
from src.data.data_fetcher import DataFetcher
from src.portfolio.portfolio_tracker import PortfolioTracker
from src.strategies.base_strategy import BacktestResult, BaseStrategy, MovingAverageCrossoverStrategy
from src.strategies.multi_indicator_strategy import MultiIndicatorStrategy
from src.utils.helpers import format_currency, format_percentage


class TradingBot:
    """
    Main trading bot that orchestrates all system components.

    WHY A SINGLE CONTROLLER CLASS?
    - Single point of control for the trading loop
    - Manages lifecycle of all components
    - Provides clean interface for different modes (live, backtest, simulation)
    - Follows the "Facade" design pattern (simplifies complex subsystems)

    COMPONENT INTERACTIONS:
    DataFetcher ──prices──> TradingBot ──data──> Strategy
                                                   │
                                              Signal
                                                   │
    PortfolioTracker <────trade──────────┬──────────┘
    """

    def __init__(self, mode: str = "simulation"):
        """
        Initialise the trading bot

        Args:
            mode: Operating mode (simulation, live or backtest)

        Why different modes?
        - Simulation : Virtual money, real data, safe testing
        - Live: Real money (Not implemented in this project)
        - Backtest: Historical data analysis

        Mode implications
        - Simulation uses real prices with virtual money
        - Different configs for differnt risk levels
        - Backtest mode disable price updates from API
        """

        self.mode = mode

        # Loading of the appropriate configuration
        # get_config() select DevelopmentConfig or ProductionConfig
        if mode == "simulation":
            self.config = get_config("development")
        else:
            self.config = get_config("production")

        # validate() check values within acceptable ranges
        self.config.validate()

        # Print active config summary to console
        self.config.display()

        # Initialise core components
        self.data_fetcher = DataFetcher()
        self.portfolio = PortfolioTracker(self.config.INITIAL_CAPITAL, self.config.PORTFOLIO_DATA_PATH)

        # Create trading strategy
        self.strategy = MultiIndicatorStrategy(self.config.INITIAL_CAPITAL, buy_threshold=2.5, sell_threshold=-2.5)

        # Bot state
        self.is_running = False
        self.last_update_time = None
        self.update_count = 0

        print(f"🤖 Trading Bot Initialised")
        print(f"   Mode: {self.mode.upper()}")
        print(f"   Strategy: {self.strategy.name}")
        print(f"   Capital: {format_currency(self.config.INITIAL_CAPITAL)}")
        print(f"   Symbols: {self.config.SYMBOLS}")
        print("=" * 60)

    def fetch_current_prices(self) -> Dict[str, float]:
        """
        Fetching current prices for all tracked symbols

        Why separate method?
        - Can add retry logic later
        - Easy to switch between real and mock data
        - Central point for price data validation
        """

        try:
            prices = self.data_fetcher.get_latest_prices(self.config.DEFAULT_SYMBOLS)
            if prices:
                print(f"📊 Current Prices: ", end="")
                for symbol, price in list(prices.items())[:3]:
                    print(f"{symbol}: {format_currency(price)}", end="")
                print()
            return prices
        except Exception as e:
            print(f"Error fetching prices: {e}")
            return {}

    def update_portfolio_prices(self, prices: Dict[str, float]) -> None:
        """
        Update portfolio with current market prices

        Why update before trading?
        - Need accurate portfolio value for position sizing
        - Unrealised PnL depends on current prices
        - Risk management requires up-to-date values
        """

        self.portfolio.update_prices(prices)

    def generate_trading_signals(self, prices: Dict[str, float]) -> list:
        """
        Generate trading signals for all tracked symbols

        Why iterate thorugh all symbols?
        - Each symbol gets an independent analysis
        - Different symbols can have different signals
        - Portfolio diversification needs multiple signals

        Signal generation process
        - Fetch historical data for indicators
        - Call strategy generate_signal() with current context
        - Validate signal quality (confidence, quantity)
        - Return only valid signals for execution
        """

        signals = []

        portfolio_summary = self.portfolio.get_portfolio_summary()

        for symbol in self.config.DEFAULT_SYMBOLS:
            try:
                # Fetch data for analysis
                data = self.data_fetcher.get_stock_data(symbol, period="3mo")

                if data.empty:
                    print(f"⚠️  No data for {symbol} - skipping")
                    continue

                # Get current price
                current_price = prices.get(symbol)
                if current_price is None:
                    print(f"⚠️  No price for {symbol} - skipping")
                    continue

                # Prepare portfolio info for strategy (FIXED FORMAT)
                portfolio_info = {
                    "cash": portfolio_summary["cash"],
                    "total_value": portfolio_summary["total_portfolio_value"],
                    "positions": {},  # ← Start with empty dict
                    "current_prices": prices,
                }

                # Safely add positions data if they exist
                positions_data = portfolio_summary.get("positions", {})
                for pos_symbol, pos_data in positions_data.items():
                    portfolio_info["positions"][pos_symbol] = pos_data["quantity"]

                # Debug output to see what we have
                # print(f"Debug: Portfolio has {len(positions_data)} positions")

                # Generate signal
                signal = self.strategy.generate_signal(symbol, data, current_price, portfolio_info)

                # Only act on strong signals
                if signal.action in ["BUY", "SELL"] and signal.confidence > 0.3:
                    signals.append(signal)
                    print(f"📡 {symbol}: {signal.action} signal " f"(confidence: {signal.confidence:.0%})")
                else:
                    print(f"📡 {symbol}: {signal.action} (confidence: {signal.confidence:.0%})")

            except Exception as e:
                print(f"❌ Error generating signal for {symbol}: {e}")
                import traceback

                traceback.print_exc()  # ← Add this for debugging
                continue

        return signals

    def execute_signal(self, signal) -> bool:
        """
        Execute trading signal by placing a trade

        Execution process
        - Validate signal parameters
        - Check portfolio constraints
        - Execute trade through PortfolioTracker
        - Log the result

        Why validation required?
        - Insufficient funds check
        - Position size check
        - Price reasonableness check

        Risk controls
        - Transaction cost included
        - Max position limits enforced
        - Stop-loss logic can be added here
        """

        try:
            # Price validation
            if signal.price <= 0:
                print(f"Invalid price for {signal.symbol}: {signal.price}")
                return False

            # Quantity validation
            if signal.quantity <= 0:
                print(f"Invalid quantity for {signal.symbol}: {signal.quantity}")
                return False

            # Check portfolio constraints for BUY orders
            if signal.action == "BUY":
                total_cost = (signal.quantity * signal.price) + self.config.TRANSACTION_COST

                portfolio_summary = self.portfolio.get_portfolio_summary()

                if portfolio_summary["cash"] < total_cost:
                    print(f"Insufficient funds for {signal.symbol} BUY")
                    print(
                        f"Need: {format_currency(total_cost)}, " f"Have: {format_currency(portfolio_summary['cash'])}"
                    )
                    return False

            # Check portfolio constraints for SELL orders
            elif signal.action == "SELL":
                portfolio_summary = self.portfolio.get_portfolio_summary()

                if signal.symbol not in portfolio_summary["positions"]:
                    print(f"No position in {signal.symbol} to sell")
                    return False

                current_position = portfolio_summary["positions"][signal.symbol]

                if signal.quantity > current_position["quantity"]:
                    print(f"Insufficient shares for {signal.symbol} SELL")
                    print(f"Want: {signal.quantity}, Have: {current_position['quantity']}")
                    return False

            # Execute the trade
            success = self.portfolio.add_trade(
                symbol=signal.symbol,
                action=signal.action,
                quantity=signal.quantity,
                price=signal.price,
                transaction_cost=self.config.TRANSACTION_COST,
            )

            if success:
                self.strategy.add_signal(signal)
                return True
            else:
                return False

        except Exception as e:
            print(f"Error executing signal: {e}")
            return False

    def run_single_update(self) -> None:
        """
        Executing one complete update cycle of trading bot

        Trading cycle
        - Fetch current market data
        - Update portfolio valuations
        - Generate trading signals
        - Execute valid signals
        - Display updated portfolio status

        Why structure it as a cycle?
        - Real trading systems run in a continuous loop
        - Each step isolated for clarity
        - Easy to add new steps (risk checks, logging)
        """
        print(f"\n{'='*60}")
        print(f"🔄 Update #{self.update_count + 1}")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # Step 1: Fetch current prices
        print("\n📡 Step 1: Fetching current prices...")
        prices = self.fetch_current_prices()

        if not prices:
            print("❌ No prices available - skipping update")
            return

        # Step 2: Update portfolio with current prices
        print("\n💰 Step 2: Updating portfolio values...")
        self.update_portfolio_prices(prices)

        # Step 3: Generate trading signals
        print("\n🧠 Step 3: Generating trading signals...")
        signals = self.generate_trading_signals(prices)

        # Step 4: Execute valid signals
        print(f"\n⚡ Step 4: Executing {len(signals)} signals...")
        executed_count = 0
        for signal in signals:
            if self.execute_signal(signal):
                executed_count += 1
        print(f"   Executed: {executed_count}/{len(signals)} signals")

        # Step 5: Display portfolio status
        print("\n📊 Step 5: Portfolio Status")
        self.portfolio.print_portfolio_summary()

        # Update bot state
        self.update_count += 1
        self.last_update_time = datetime.now()

    def run_backtest(self, days_back: int = 120) -> Optional[BacktestResult]:
        """
        Run a comprehensive backtest of the strategy.

        YFINANCE PERIOD MAPPING:
        - days_back <= 7: Use "1mo" period
        - days_back <= 90: Use "3mo" period
        - days_back <= 180: Use "6mo" period
        - days_back > 180: Use "1y" period

        Why these mappings?
        - yfinance only accepts predefined periods
        - We get more data than needed, then filter
        - Ensures we have enough data for technical indicators
        """
        print(f"\n{'='*60}")
        print(f"📊 Running Backtest (approximately {days_back} days)")
        print(f"{'='*60}")

        # Convert days to valid yfinance period
        if days_back <= 7:
            period = "1mo"
            print(f"Using 1 month of data (≥{days_back} days)")
        elif days_back <= 90:
            period = "3mo"
            print(f"Using 3 months of data (≥{days_back} days)")
        elif days_back <= 180:
            period = "6mo"
            print(f"Using 6 months of data (≥{days_back} days)")
        else:
            period = "1y"
            print(f"Using 1 year of data (≥{days_back} days)")

        # Fetch backtest data
        print("\n📡 Fetching historical data...")
        backtest_data = {}

        for symbol in self.config.DEFAULT_SYMBOLS:
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period)

                # Validate data type and content
                if isinstance(data, str):
                    print(f"   ❌ {symbol}: API Error - {data}")
                    continue
                elif data.empty:
                    print(f"   ❌ {symbol}: No data returned")
                    continue
                else:
                    # Optionally limit to requested days
                    if len(data) > days_back:
                        data = data.tail(days_back)

                    backtest_data[symbol] = data
                    print(f"   ✅ {symbol}: {len(data)} days")

            except Exception as e:
                print(f"   ❌ {symbol}: Exception - {e}")
                continue

        if not backtest_data:
            print("❌ No data available for backtesting")
            return None

        # Run the backtest
        print(f"\n🔄 Running backtest simulation on {len(backtest_data)} symbols...")

        try:
            # Calculate start date from actual data
            all_dates = []
            for df in backtest_data.values():
                all_dates.extend(df.index)

            if all_dates:
                earliest_date = min(all_dates)
                latest_date = max(all_dates)
                print(f"Data range: {earliest_date.date()} to {latest_date.date()}")

            result = self.strategy.backtest(backtest_data)
            return result

        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            traceback.print_exc()
            return None

    def run_simulation(self, max_updates: int = 3, delay_seconds: int = 2) -> None:
        """
        Run simulation with multiple update cycles

        Args:
            max_updates: Number of update cycles to run before it shuts down
            delay_seconds: Simulate time passing

        Why multiple cycles?
        - Demostrate the bot running over time
        - Show how portfolio changes with market
        - Test strategy consistency

        Delay between updates
        - Simulate real-world waiting periods
        - Prevents excessive API calls
        - Allow time for market to update

        Why shut it down?
        - Check if code logic works without creating massive log file
        - Ensure bot stops if there are infinite loops due to bugs
        - Ensure bot finishes and saves frequently
        - Potential API credit limits
        """

        print(f"\n🚀 Starting {self.mode} with {max_updates} updates")
        print(f"   Delay between updates: {delay_seconds} seconds")

        self.is_running = True

        for update_num in range(max_updates):
            if not self.is_running:
                print("Bot stopped")

            self.run_single_update()

            if update_num < max_updates - 1:
                print(f"\n⏳ Waiting {delay_seconds}s before next update...")
                time.sleep(delay_seconds)

        print(f"\n✅ Simulation complete after {self.update_count} updates")

    def display_final_report(self) -> None:
        """
        Display comprehensive final report of bot's performance

        Why final report?
        - Summarise bot activity
        - Show performance metrics
        - Professional presentation of results
        - Good for documentation and learning
        """

        print(f"\n{'='*60}")
        print(f"📋 FINAL TRADING REPORT")
        print(f"{'='*60}")

        # Bot statistics
        print(f"\n🤖 Bot Statistics:")
        print(f"   Mode: {self.mode.upper()}")
        print(f"   Strategy: {self.strategy.name}")
        print(f"   Updates Run: {self.update_count}")
        print(f"   Last Update: {self.last_update_time}")

        # Portfolio performance
        print(f"\n💼 Portfolio Performance:")
        summary = self.portfolio.get_portfolio_summary()

        # Print the key values from summary dict
        print(f"   Cash Remaining : {format_currency(summary.get('cash', 0))}")
        print(f"   Portfolio Value: {format_currency(summary.get('total_portfolio_value', 0))}")

        print(f"   Unrealised PnL : {format_currency(summary.get('total_unrealised_pnl', 0))}")
        print(f"   Unrealised %   : {summary.get('unrealised_pnl_pct', 0):.1f}%")

        # Show each open position
        positions = summary.get("positions", {})
        print(f"   Total Positions: {len(positions)}")
        if positions:
            print(f"\n   Open Positions:")
            for sym, pos in positions.items():
                print(f"     {sym:<6} qty={pos['quantity']:>3} avg=${pos['avg_cost']:>8.2f}")

        # Signal history
        if self.strategy.signals_history:
            print(f"\n📡 Signal History ({len(self.strategy.signals_history)} total):")
            buy_count = sum(1 for s in self.strategy.signals_history if s.action == "BUY")
            sell_count = sum(1 for s in self.strategy.signals_history if s.action == "SELL")
            hold_count = sum(1 for s in self.strategy.signals_history if s.action == "HOLD")
            print(f" BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")

            # Recent signals
            print(f"\n Recent signals:")
            for signal in self.strategy.signals_history[-3:]:
                print(
                    f"   - {signal.symbol}: {signal.action} @ "
                    f"{format_currency(signal.price)} "
                    f"(confidence: {signal.confidence:.0%})"
                )

        # Trade history
        trade_df = self.portfolio.get_trade_history()
        if not trade_df.empty:
            print(f"\n Recent trades:")
            print(trade_df.tail(5))

        print(f"\n{'='*60}")


def main():
    """
    Main entry point for the trading bot application

    Application flow
    - Load configuration
    - Initialise trading bot
    - Run backtest (Optional)
    - Run simulation
    - Display final report
    """

    print("🚀 Algorithmic Trading Bot")
    print("=" * 60)
    print("Portfolio Project")
    print("=" * 60)

    try:
        # Initialise the trading bot
        bot = TradingBot(mode="simulation")

        # 1) Run backtest first for analysis
        print(f"\n STEP 1: Strategy Backtesting")
        print(f"\n{'='*60}")
        backtest_result = bot.run_backtest(days_back=365)

        # 2) Run simulation with live data
        print(f"\n STEP 2: Running Trading Simulation")
        print(f"\n{'='*60}")
        bot.run_simulation(max_updates=2, delay_seconds=2)

        # 3) Display final report
        print(f"\n STEP 3: Final Report")
        bot.display_final_report()

        print("\n Trading bot execution completed")
    except KeyboardInterrupt:
        print("\n\n Bot stopped by user")

    except Exception as e:
        print(f"\n Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
