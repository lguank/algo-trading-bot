"""
Data Fetching Module for the Algorithmic Trading Bot.

PANDAS/NUMPY CONCEPTS USED:
- DataFrame: Like a spreadsheet, rows = dates, columns = OHLCV prices
- Index: The row labels (in finance, usually dates)
- .iloc[]: Selection by position (e.g., get the last row)
- Dictionary: Map stock symbols (AAPL) to their DataFrames

Design Philosophy:
- Encapsulate API logic (separation of concerns)
- Handle errors gracefully (defensive programming)
- Use caching to speed up development (efficiency)
- Respect API limits (professionalism)
"""

import time
from datetime import datetime, timedelta
from traceback import print_exception
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from pandas import period_range
from yfinance import ticker

from src import data

"""
Handles data fetching opreations from Yahoo Finance

Why a class?
- Maintain internal state (like cache)
- Encapsulate the yfinance logic
- Ease in testing
"""


class DataFetcher:

    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.last_fetch_time: Dict[str, datetime] = {}
        self.cache_duration_seconds: int = 300

    """
     - Handle data fetching operations from Yahoo Finance

     Why set up this cache?
     - Avoid downloading same data 10 times in 10 seconds
     - Speed up development
    """

    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch histrocial stock data for single symbol

        Args:
        - symbol (str): Stock ticker symbol
        - period: Time period which Defaults to '1y'.

        Returns:
        - Dataframe with columns: Open, High, Low, Close, Volume, Adj Close
        - Return empty Dataframe if there is an error

        Cache logic:
        - Check if we fetched this symbol+period recently
        - "Recently" = within cache_duration_seconds (5 min default)
        - If yes: return cached copy instantly (no API call)
        - If no: fetch from yfinance, store in cache, return
        """

        cache_key = f"{symbol}_{period}"
        now = datetime.now()

        # If we fetched this recently, return cache
        if cache_key in self.cache and cache_key in self.last_fetch_time:
            time_since_fetch = (now - self.last_fetch_time[cache_key]).total_seconds()
            if time_since_fetch < self.cache_duration_seconds:  # 5 minutes
                print(f"📦 Using cached data for {symbol}")
                return self.cache[cache_key]

        try:
            ticker = yf.Ticker(symbol)

            # Returns a Panda Dataframe
            data = ticker.history(period=period)

            # Empty dataframe would mean that the symbol likely invalid
            if data.empty:
                print(f"No data found for symbol {symbol}")
                return pd.DataFrame()

            data.columns = data.columns.str.replace(" ", "")

            # Storage of data in the cache
            self.cache[cache_key] = data
            self.last_fetch_time[cache_key] = now

            print(f"  [DataFetcher] ✅ {symbol}: {len(data)} rows fetched ({period})")
            return data

        except Exception as e:
            print(f"  [DataFetcher] ❌ Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Alias for get_stock_data with interval parameter

        Why do we need this?
        - get_stock_data() only accepts period
        - this method adds interval support and keeps both names working

        Interval controls the bar size:
        - "1d"  = daily bars   (default, most indicators use this)
        - "1h"  = hourly bars  (more data, more noise)
        - "5m"  = 5-min bars   (intraday, needs recent period only)

        Note: yfinance restricts interval/period combinations
        - "1m" interval: only available for last 7 days
        - "1h" interval: only available for last 730 days
        - "1d" interval: available for any period
        """

        # If interval is daily, we can use get_stock_data
        if interval == "1d":
            return self.get_stock_data(symbol, period)

        # For non daily-intervals, fetch directly without caching
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                print(f"  [DataFetcher] No data for {symbol} ({period}, {interval})")
                return pd.DataFrame()

            print(f"  [DataFetcher] ✅ {symbol}: {len(data)} rows " f"({period}, {interval})")
            return data

        except Exception as e:
            print(f"  [DataFetcher] ❌ Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the most recent closing price for a symbol

        PANDAS CONCEPT: .iloc[location]
        - .iloc[-1]: Selects the LAST row
        - data['Close']: Selects the column named 'Close'
        - data['Close'].iloc[-1]: The value at the intersection of Last Row and Close Column
        """

        data = self.get_stock_data(symbol, period="5d")

        if data.empty:
            return None

        try:
            # Obtaining the closing price
            current_price = data["Close"].iloc[-1]
            return float(current_price)

        except Exception as e:
            print(f"Error getting the current price: {e}")
            return None

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols in one call.

        Why a dedicated method?
        - Single entry point for bulk fetching of prices
        - main.py and TradingBot can call this once per update cycle

        Args:
            symbols: List of ticker symbols

        Returns:
            Dictionary: { 'AAPL': 150.50, 'GOOGL': 2800.00 }
        """
        prices = {}

        for symbol in symbols:
            try:
                # Get 5 days of data to ensure we get latest price
                data = self.get_stock_data(symbol, period="5d")

                # Safety check - ensure data is DataFrame and not empty
                if isinstance(data, pd.DataFrame) and not data.empty:
                    latest_price = float(data["Close"].iloc[-1])
                    prices[symbol] = latest_price
                else:
                    print(f"⚠️ No valid price data for {symbol}")

            except Exception as e:
                print(f"❌ Error getting price for {symbol}: {e}")
                continue

        return prices

    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks simultaneously

        Why dictionary?
        - Map keys (symbols) to complex values (Dataframes)
        - Easy for looking up via data['APPL']

        Why delay?
        - Prevents one from getting blocked
        - SOP for APIs
        """
        stock_data: Dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            data = self.get_stock_data(symbol, period)

            if not data.empty:
                stock_data[symbol] = data

            # Small delay
            time.sleep(0.5)

        return stock_data
