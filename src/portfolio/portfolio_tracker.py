"""
Portfolio Management Module for the Algorithmic Trading Bot.

PANDAS/NUMPY CONCEPTS USED:
- Series: For price calculations and P&L tracking
- DateTime: For trade timestamps and portfolio history
- JSON: For saving/loading portfolio state persistently
- Dictionary comprehensions: For efficient portfolio summary generation

FINANCIAL CONCEPTS:
- Position: A holding in a particular stock (quantity, average cost)
- Trade: A single buy or sell transaction
- Portfolio Value: Cash + Market Value of all positions
- Unrealized P&L: Profit/Loss on positions you still hold
- Realized P&L: Profit/Loss from completed trades

Design Philosophy:
- Immutable trade records (audit trail)
- Defensive programming (validate all inputs)
- Separation of concerns (Position vs Trade vs Portfolio)
- Professional financial calculations
"""

import grp
import json
import os
from argparse import Action
from dataclasses import asdict, dataclass
from datetime import datetime
from doctest import FAIL_FAST
from pickle import FALSE
from shutil import ExecError
from symtable import Symbol
from turtle import pos
from typing import Dict, List, Optional, Tuple

import pandas as pd
from numpy import true_divide

from config.config import get_config

asdict

"""
Why do we use @dataclass?
- Automatically creates the __init__, __repr__, __eq__ methods

Instead of writing
class Position:
    def __init__(self, symbol, quantity, avg_cost):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_cost = avg_cost

We can just write:
@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float

Much cleaner and less error-prone!
"""


@dataclass
class Position:
    """
    Represent stock position in portfolio

    Why dataclass?
    - Easy to convert to dictionary for JSON storage
    - Built in comparison function (__eq__)
    - Automatic validation through type hints

    """

    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        """
        Current market value of position

        Formula = current_price x quantity

        Why use @property
        - Calculated dynamically
        - Accessed like an attribute
        - Cannot be accidentally modified
        """

        return self.current_price * self.quantity

    @property
    def cost_basis(self) -> float:
        """
        Total amount originally invested in this position

        Formula = quantity x average_cost

        Potnetial improvement
        - Accounting for commisions / fees incurred in transactions

        Why needed?
        - Needed for tax calculations
        - Required for P&L calculations
        """

        return self.quantity * self.avg_cost

    @property
    def unrealised_pnl(self) -> float:
        """
        Unrealised profit/loss (still holding this position)

        Formula = market_value - cost_basis
        """

        return self.market_value - self.cost_basis

    @property
    def unrealised_pnl_pct(self) -> float:
        """
        Unrealsied P&L as percentage

        Formula = (unrealised P&L / cost_basis) x 100

        Why percentage?
        - Ease of comparison between different positions
        - Independence of position size
        """

        if self.cost_basis == 0:
            return 0.0
        return (self.unrealised_pnl / self.cost_basis) * 100


@dataclass
class Trade:
    """
    Represents a single buy or sell transaction

    Why separate Trade from Position?
    - Trades are immutable for the Audit trail
    - Position = Current state (what you own)
    - Trade = Historical event (what happened when)
    - Position changes when price moves
    """

    timestamp: datetime
    symbol: str
    quantity: int
    action: str  # 'BUY' or 'SELL'
    price: float
    transaction_cost: float

    @property
    def total_value(self) -> float:
        """
        Total cash impact of this trade

        For BUY: Cash outflow = Negative
        For SELL: Cash inflow = Positive

        Inclusion of transaction costs
        - Real trading has commisions
        - Accuracy in backtesting
        """

        base_value = self.quantity * self.price
        if self.action.upper() == "BUY":
            # Buying = Pay the price and cost
            return -(base_value + self.transaction_cost)
        else:  # Sell
            return base_value - self.transaction_cost


class PortfolioTracker:
    """
    Main portfolio management class.

    Responsibilities:
    - Track cash balance
    - Manage stock positions
    - Execute trades (virtually)
    - Calculate portfolio performance
    - Save/load portfolio state

    Why a class instead of functions?
    - Maintains state (cash, positions, trades)
    - Encapsulates portfolio logic
    - Easier to test and extend
    """

    def __init__(self, initial_capital: float, data_file_path: Optional[str] = None):
        """
        Initialisation of portfolio tracker

        Why optional file path?
        - Flexibility: Allows for default or custom paths to be used
        - Testing: Create temporary portfolios
        - Configuration: Different paths for dev/prod (Testing new features etc.)
        """

        self.initial_capital = initial_capital

        # Set up file path
        if data_file_path is None:
            config = get_config()
            self.data_file_path = config.PORTFOLIO_DATA_PATH
        else:
            self.data_file_path = data_file_path

        # Initialise portfolio state
        self.position: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.cash = initial_capital

        self.load_portfolio_data()

        print(f"Portfolio initialised with ${initial_capital:,.2f}")

    def add_trade(self, symbol: str, action: str, quantity: int, price: float, transaction_cost: float = 0.0) -> bool:
        """
        Execute a trade and updating portfolio

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Price per share
            transaction_cost: Trading fees

        Returns
            True if trade executed successfully

        Why return bool instead of raising exceptions?
        - Graceful handling of insufficient funds
        - Allow trading algorithms to continue
        - Clear success and failure indication

        """

        # Input validation
        if quantity <= 0:
            print(f"Invalid quantity: {quantity}")
            return False

        if price <= 0:
            print(f"Invalid Price: {price}")
            return False

        # Creating the object
        trade = Trade(
            timestamp=datetime.now(),
            symbol=symbol.upper(),
            quantity=quantity,
            action=action.upper(),
            price=price,
            transaction_cost=transaction_cost,
        )

        if action.upper() == "BUY":
            return self._execute_buy(trade)
        elif action.upper() == "SELL":
            return self._execute_sell(trade)
        else:
            print(f"Invalid action: {action}")
            return False

    def _execute_buy(self, trade: Trade) -> bool:
        """
        Execute a buy order

        Why use a private method?
        - Internal implementation detail
        - Not meant to be called directly
        - Allows add_trade to validate the purchase
        """

        total_cost = (trade.quantity * trade.price) + trade.transaction_cost

        # Check if we have enough cash
        if self.cash < total_cost:
            print(f"Insufficient funds for {trade.symbol}")
            print(f"Need: ${total_cost:,.2f}, Have: ${self.cash:,.2f}")
            return False

        # Update cash
        self.cash -= total_cost

        # Update position
        if trade.symbol in self.position:

            # Add to exisitng position
            pos = self.position[trade.symbol]

            total_shares = pos.quantity + trade.quantity

            # Total cost basis will be the total price paid, those from original and those from the new 'trade'
            total_cost_basis = (pos.quantity * pos.avg_cost) + (trade.quantity * trade.price)

            new_avg_cost = total_cost_basis / total_shares

            # Update position
            pos.quantity = total_shares
            pos.avg_cost = new_avg_cost

        else:

            # Create new position
            self.position[trade.symbol] = Position(
                symbol=trade.symbol, quantity=trade.quantity, avg_cost=trade.price, current_price=trade.price
            )

        # Record trade
        self.trades.append(trade)
        self.save_portfolio_data()

        print(f"BUY: {trade.quantity} {trade.symbol} @ {trade.price:,.2f}")
        return True

    def _execute_sell(self, trade: Trade) -> bool:
        """
        Execute sell order

        Selling more complex than buying
        - Check if we own enough shares
        - May fully or partially close a position
        - Need to calculate realised P&L
        """

        # Check if we have the position
        if trade.symbol not in self.position:
            print(f"No position in {trade.symbol} to sell")
            return False

        pos = self.position[trade.symbol]

        # Check if we have enough shares
        if pos.quantity < trade.quantity:
            print(f"Insufficient funds to trade in {trade.symbol}")
            print(f"Want to sell {trade.quantity}. Own {pos.quantity}")
            return False

        # Calculate cash we will receive
        gross_proceeds = trade.quantity * trade.price
        net_proceeds = gross_proceeds - trade.transaction_cost

        # Update cash
        self.cash += net_proceeds

        # Update position
        pos.quantity -= trade.quantity

        # Remove position if fully sold
        if pos.quantity == 0:
            del self.position[trade.symbol]

        self.trades.append(trade)
        self.save_portfolio_data()

        print(f"SELL: {trade.quantity} {trade.symbol} @ {trade.price:,.2f}")
        return True

    def update_prices(self, price_data: Dict[str, float]) -> None:
        """
        Update current prices for all positions

        Why a separate method?
        - Portfolio position don't automatically know current prices
        - Data fetcher provides prices, portfolio uses them
        """

        for symbol, price in price_data.items():
            if symbol in self.position:
                self.position[symbol].current_price = price

        self.save_portfolio_data()

    def get_portfolio_summary(self) -> Dict:
        """
        Generate complete portfolio performance summary

        Why return dictionary?
        - Easy to convert into JSON for saving
        - Can be passed to differnt display functions
        - Can add new metrics without changing interface
        """

        # Calculate portfolio totals
        total_market_value = sum(pos.market_value for pos in self.position.values())
        total_cost_basis = sum(pos.cost_basis for pos in self.position.values())
        total_unrealised_pnl = total_market_value - total_cost_basis
        total_portfolio_value = self.cash + total_market_value

        # Calculate performance metrics
        total_return_pct = ((total_portfolio_value - self.initial_capital) / self.initial_capital) * 100
        unrealised_pnl_pct = (total_unrealised_pnl / total_cost_basis) * 100 if total_cost_basis > 0 else 0.0

        return {
            "cash": self.cash,
            "total_market_value": total_market_value,
            "total_cost_basis": total_cost_basis,
            "total_portfolio_value": total_portfolio_value,
            "total_unrealised_pnl": total_unrealised_pnl,
            "unrealised_pnl_pct": unrealised_pnl_pct,
            "total_return_pct": total_return_pct,
            "number_of_position": len(self.position),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "cost_basis": pos.cost_basis,
                    "unrealised_pnl": pos.unrealised_pnl,
                    "unrealised_pnl_pct": pos.unrealised_pnl_pct,
                }
                for symbol, pos in self.position.items()
            },
        }

    def save_portfolio_data(self) -> None:
        """
        Save portfolio state to JSON file

        Why JSON?
        - Human readable
        - Built in python support
        - Cross-platform compatibility

        JSON Structure
        {
            "initial_capital": 100000,
            "cash": 85000,
            "position": {...},
            "trades": [...]
        }
        """

        data = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": {symbol: asdict(pos) for symbol, pos in self.position.items()},
            "trades": [
                {
                    "timestamp": trade.timestamp.isoformat(),
                    "symbol": trade.symbol,
                    "action": trade.action,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "transaction_cost": trade.transaction_cost,
                }
                for trade in self.trades
            ],
        }

        # Make sure the directory exists
        os.makedirs(os.path.dirname(self.data_file_path), exist_ok=True)

        # Saving data into a permanenet file (Take python object and write into file using JSON format)
        try:
            with open(self.data_file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving portfolio: {e}")

    def load_portfolio_data(self) -> None:
        """
        Load portfolio state from JSON file

        Why handle errors gracefully?
        - First time running: file will not exist
        - Corrupted file: should start fresh instead of crash
        - Permission issues: informative error message
        """

        if not os.path.exists(self.data_file_path):
            print(f"No existing portfolio data found at {self.data_file_path}")
            return

        try:
            with open(self.data_file_path, "r") as f:
                data = json.load(f)

            # Restore cash value
            self.cash = data.get("cash", self.initial_capital)

            # Restore position
            self.position = {}
            for symbol, pos_data in data.get("positions", {}).items():
                self.position[symbol] = Position(
                    symbol=pos_data["symbol"],
                    quantity=pos_data["quantity"],
                    avg_cost=pos_data["avg_cost"],
                    current_price=pos_data.get("current_price", 0.0),
                )

            # Restore trades
            self.trades = []
            for trade_data in data.get("trades", []):
                trade = Trade(
                    timestamp=datetime.fromisoformat(trade_data["timestamp"]),
                    symbol=trade_data["symbol"],
                    action=trade_data["action"],
                    quantity=trade_data["quantity"],
                    price=trade_data["price"],
                    transaction_cost=trade_data.get("transaction_cost", 0.0),
                )
                self.trades.append(trade)

            print(f"Loaded portfolio with {len(self.position)} positions and {len(self.trades)} trades")

        except Exception as e:
            print(f"Error loading portfolio data: {e}")
            print("Starting with fresh portfolio")

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as panda Dataframe

        Returns:
        - Dataframe with columns: timestamp, symbol, action, quantity, price, value

        Why return Dataframe?
        - Easy to analyse with panda
        - Natural for financial data analysis
        - Integrates well with plotting libraries
        """

        if not self.trades:
            return pd.DataFrame

        # Convert trades to list of dictionaries
        trade_dicts = []
        for trade in self.trades:
            trade_dict = asdict(trade)
            # Creating new key called 'total_value' and using method total_value to obtain the value
            trade_dict["total_value"] = trade.total_value
            trade_dicts.append(trade_dict)

        # Creating the dataframe
        df = pd.DataFrame(trade_dicts)

        # Basically set timestamp as the row labels instead of standard indexing
        df.set_index("timestamp", inplace=True)

        return df

    def print_portfolio_summary(self) -> None:
        """
        Print formatted portfolio summary

        Why separate print method?
        - get_portfolio_summary() returns data
        - get_portfolio_summary() formats for humans
        - Separation of display and data logic
        """

        summary = self.get_portfolio_summary()

        print("\n" + "=" * 50)
        print("📊 PORTFOLIO SUMMARY")
        print("=" * 50)
        print(f"💰 Cash:                 ${summary['cash']:,.2f}")
        print(f"📈 Market Value:         ${summary['total_market_value']:,.2f}")
        print(f"💼 Total Portfolio:      ${summary['total_portfolio_value']:,.2f}")
        print(f"📊 Total Return:         {summary['total_return_pct']:.2f}%")
        print(f"🎯 Unrealised P&L:       ${summary['total_unrealised_pnl']:,.2f}")
        print(f"📍 Positions:            {summary['number_of_position']}")

        if summary.get("positions"):
            print("\n📋 INDIVIDUAL POSITIONS:")
            print("-" * 70)
            for symbol, pos in summary["positions"].items():
                pnl_sign = "+" if pos["unrealised_pnl"] >= 0 else ""
                print(
                    f"{symbol:6} | {pos['quantity']:4d} shares | "
                    f"${pos['avg_cost']:7.2f} avg | "
                    f"${pos['current_price']:7.2f} now | "
                    f"{pnl_sign}${pos['unrealised_pnl']:8.2f} "
                    f"({pnl_sign}{pos['unrealised_pnl_pct']:5.1f}%)"
                )

        print("=" * 50)
