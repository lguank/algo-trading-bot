"""
Utility functions for the Algorithmic Trading Bot.

Design Philosophy:
- Pure functions (input → output, no side effects)
- Well-documented mathematical formulas
- Error handling for edge cases
- Reusable across different modules

Indicator Categories Covered:
1. Trend Indicators     - SMA, EMA, MACD
2. Momentum Indicators  - RSI, Stochastic Oscillator
3. Volatility Indicators- Bollinger Bands, ATR
4. Volume Indicators    - OBV, VWAP
5. Performance Metrics  - Sharpe Ratio, Max Drawdown, Returns
6. Portfolio Utilities  - Position sizing
7. Helper Utilities     - Formatting, validation, cleaning
"""

from datetime import date, datetime, timedelta
from math import nan
from multiprocessing import Value
from pdb import run
from pickle import FLOAT, LIST
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.pyplot import flag

# === TREND INDICATORS ===
# === Tells us what direction the market is moving ===


def calculate_moving_average(data: pd.Series, window: int) -> pd.Series:
    """
    Calculation of the Simple Moving Average

    Args:
    - data : Price data (usually the closing prices) - pd.Series
    - window = Number of periods for average

    Return
    - Series with moving averages
    """
    if len(data) < window:
        raise ValueError(f"Not enough data points. Need {window}, got {len(data)}")
    moving_avg = data.rolling(window=window).mean()
    return moving_avg


"""
Calculation of Exponential Moving Average

Formula: EMA = (Price * α) + (Previous_EMA * (1 - α))
    where α = 2/(window+1) ← smoothing factor

Why EMA over SMA?
- Reacts faster to recent price changes
- Reduction of lag in signals
- Help in identification of new trends earlier

Explanation of ewm function
- span=window controls how much weight current values get
 - E.g Larger Span = Smaller Decay Rate = Recent values get smaller weightage
- adjust=False means using the standard EMA formula
 - Most charting platforms and technical indicators use recursive formula
 - Lighter on memory since only need current price and previous EMA value
"""


def calculate_exponential_moving_average(data: pd.Series, window: int) -> pd.Series:
    """
    Calculation of Exponential Moving Average

    Formula: EMA = (Price * α) + (Previous_EMA * (1 - α))
        where α = 2/(window+1) ← smoothing factor

    Why EMA over SMA?
    - Reacts faster to recent price changes
    - Reduction of lag in signals
    - Help in identification of new trends earlier

    Explanation of ewm function
    - span=window controls how much weight current values get
     - E.g Larger Span = Smaller Decay Rate = Recent values get smaller weightage
    - adjust=False means using the standard EMA formula
     - Most charting platforms and technical indicators use recursive formula
     - Lighter on memory since only need current price and previous EMA value
    """

    if len(data) < window:
        raise ValueError(f"Not enough data points. Need {window}, got {len(data)}")
    ema = data.ewm(span=window, adjust=False).mean()
    return ema


def calculate_macd(
    data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence (MACD)

    Components of MACD
    - MACD Line = EMA(12) - EMA(26), measuring momentum of trend
    - Signal Line = EMA(9)
    - Histogram = MACD Line - Signal Line

    Args:
        data          : Price series (closing prices)
        fast_period   : Short EMA period (default 12)
        slow_period   : Long EMA period (default 26)
        signal_period : Signal line EMA period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram) as pd.Series
    """

    min_required = slow_period + signal_period
    if len(data) < min_required:
        raise ValueError(f"MACD requires {min_required} data points, only {len(data)} provided")

    ema_fast = calculate_exponential_moving_average(data, fast_period)
    ema_slow = calculate_exponential_moving_average(data, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# === MOMENTUM INDICATORS ===
# === Tells us how STRONG the move is and if it is exhausted ===


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculation of Relative Strength Index (RSI)

    RSI measurement and formula
    - Measures overbought or oversold conditions
    - Formula
     - RSI = 100 - (100 / (1 + RS))
     - RS = Average Gain / Average Loss
    - Window set as 14 as it represents half a 28-day cycle
     - Fast enough to catch move, Slow enough to filter out daily "noise"
     - Represent half od a 28 day lunar cycle

    Using Wilder's EWM smoothing (com = window - 1):
    - com stands for "center of mass" in pandas
    - This is equivalent to α = 1/window in Wilder's original formula
    - com=13 for window=14 means α = 1/14 ≈ 0.071
    - More stable than simple rolling average RSI

    Edge cases to be handled
    1) avg_loss = 0 (pure uptrend) -> RSI = 100
    2) avg_gain = 0 (pure downtrend) -> RSI = 0
    3) insufficient data -> RSI = 50


    Explanation of where function
    - Example
     - delta.where(delta>0, 0)
      - If delta > 0, keep the value
      - If delta <= 0, replace the value with 0
    """

    if len(data) < window + 1:
        raise ValueError(f"Not enough data for RSI. Need {window+1}, got {len(data)}")

    delta = data.diff()

    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    avg_gains = gains.ewm(com=window - 1, adjust=False).mean()
    avg_losses = losses.ewm(com=window - 1, adjust=False).mean()

    # Edge case handling
    rsi = pd.Series(index=data.index, dtype=float)

    # Tolerance for floating point zero comparison
    # Using == 0 fails since ewm([NaN,0,0,0...]) produces tiny non-zero values
    ZERO_TOLERANCE = 1e-10

    for i in range(len(data)):
        ag = avg_gains.iloc[-1]
        al = avg_losses.iloc[-1]

        if pd.isna(ag) or pd.isna(al):
            rsi.iloc[i] = 50.0  # insufficient data
        elif al < ZERO_TOLERANCE and ag < ZERO_TOLERANCE:
            rsi.iloc[i] = 50.0
        elif al < ZERO_TOLERANCE:
            rsi.iloc[i] = 100.0
        elif ag < ZERO_TOLERANCE:
            rsi.iloc[i] = 0.0
        else:
            rs = ag / al
            rsi.iloc[i] = 100 - (100 / (1 + rs))

    return rsi


def calculate_stochastic_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator

    Args:
        high (pd.Series): high price series
        low (pd.Series): low price series
        close (pd.Series): close price series
        k_period (int, optional): look back window for raw %K
        d_period (int, optional): smoothing window for %D signal line

    Returns:
        Tuple of (%K series, %D Series), both 0-100

    Why use stochastic alongside RSI?
    - RSI measure speed of price changes
    - Stochastic measures position within recent range
    - Together they confirm momentum
    """

    if len(close) < k_period:
        raise ValueError(f"Stochastic requires {k_period} data points, only {len(close)} provided")

    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    price_range = (highest_high - lowest_low).replace(0, np.nan)
    k_line = ((close - lowest_low) / (price_range)) * 100
    d_line = k_line.rolling(window=d_period).mean()

    return k_line, d_line


# === VOLATILITY INDICATORS ===
# === Tells us how much risk exist in the current market ===


def calculate_bollinger_bands(
    data: pd.Series, window: int = 20, std_dev: float = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculation of Bollinger Bands

    Components of the Bollinger Bands
    Middle Band: Simple Moving Average
    Upper Band: SMA + (standard deviation * multiplier)
    Lower Band: SMA - (standard deviation * multiplier)

    Assumptions
    Multiplier = 2
    - Covers 95% of data within upper and lower bands
    - Helps in identifying true outliers when bands are broken or touched

    Window = 20
    - Balance between responsiveness and noise reduction
    """

    if len(data) < window:
        raise ValueError(f"Not enough data for Bollinger Bands. Need {window}, got {len(data)}")

    middle_band = calculate_moving_average(data, window)
    rolling_std = data.rolling(window=window).std()

    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)

    return upper_band, middle_band, lower_band


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range

    Why not High - Low?
    - If a stock gaps up 5% overnight, that gap is real volatility
    - High - Low ignores the gap entirely
    - ATR captures the full risk, including overnight moves

    True Range = Max of these three;
    - High - Low (today's intraday range)
    - |High - Prev Close| (gap up + today's high)
    - |Low - Prev Close| (gap down + today's low)

    Args:
        high (pd.Series): High price series
        low (pd.Series): Low price series
        close (pd.Series): Close price series
        window (int, optional): Smoothing period Defaults to 14.

    Returns:
        pd.Series of ATR values in price units
    """

    if len(close) < window + 1:
        raise ValueError(f"ATR requires {window+1} data points, only {len(close)} provided")

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(com=window, adjust=False).mean()

    return atr


# === VOLUME INDICATORS ===
# === These CONFIRM whether prices moves are backed by real activity ===


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-balance volume

    Formula:
    - If Close > Previous Close: OBV = Previous OBV + Volume
    - If Close < Previous Close: OBV = Previous OBV - Volume
    - If Close = Previous Close: OBV = Previous OBV

    The core insight:
    - Smart money (institutions) accumulate shares quietly
    - Volume starts rising before price does (institutions buying)
    - OBV rises → institutional buying → price will follow UP
    - OBV falls → institutional selling → price will follow DOWN

    Args:
        close (pd.Series): close price series
        volume (pd.Series): volume series

    Returns:
        pd.Series: culmulative OBV values
    """

    if len(close) != len(volume):
        raise ValueError(f"Close and Volume series must have the same length")
    if len(close) < 2:
        raise ValueError(f"OBV requires more than 2 data points")

    # +1 if price went up, -1 if price went down, 0 if it remained
    direction = np.sign(close.diff()).fillna(0)

    obv = (direction * volume).cumsum()
    return obv


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Volume weighted average price (VWAP)

    Formula:
        Typical Price = (High + Low + Close) / 3
        VWAP = Σ(Typical Price × Volume) / Σ(Volume)

    Why Typical Price and not just Close?
    - Captures the full price range of each period
    - More representative of where "fair value" was during the day
    - High + Low + Close = triangle center of each candle

    Pairings
    - MACD (Momentum filter)
    - OBV (Smart Money filter)
    """

    if len(close) < 1:
        raise ValueError(f"VWAP requires at least one data point")

    typical_price = (high + low + close) / 3
    culmulative_tp_vol = (typical_price * volume).cumsum()
    culmulative_vol = volume.cumsum()

    vwap = culmulative_tp_vol / culmulative_vol.replace(0, np.nan)

    return vwap


# === PERFORMANCE METRICS ===


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculation of percentage returns from price data

    Formula: (Today's Price - Yesterday's Price) / Yesterday's Price

    Why calculate returns?
    - Standaridation of comparison across different price levels
    - Essential in risk analysis
    - Industry standard metric
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 price points to calculate returns")
    returns = prices.pct_change().dropna()

    return returns


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculaion of Sharpe Ratio

    Formula: (Annual returns - Risk free rate) / Annual Volatility
     - Annualised Sharpe Ratio calcualted using Annualised volatility and returns

    Assumptions
    - Risk free rate of 2%
    - 252 trading days every year

    Explanation of codes
    - np.sqrt: Used for the annualisation of the sharpe ratio
    - returns.mean(): To find the average daily returns
    - returns.mean() * 252 = Find the annual returns
    - returns.std(): To find the volatility of daily returns
    - returns.std() * np_sqrt(252) = Find the annual volatility

    Improvement
    - Use sortiono ratio instead of sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)

    if annual_volatility == 0:
        return 0.0

    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

    return sharpe_ratio


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Calulate Maximum Drawdown

    Formula
    drawdown = (Trough Value - Peak Value) / Peak Value
    MDD = min(drawdown) for all values

    Explanation of codes
    - .expanding().max()
     - Window will expand, and the largest value seen at each point is found
    """
    if len(portfolio_values) == 0:
        return 0.0

    running_max = portfolio_values.expanding().max()

    drawdown = (portfolio_values - running_max) / running_max

    max_drawdown = drawdown.min()

    return max_drawdown


def calculate_win_rate(trades: list) -> float:
    """
    Win Rate (Hit Rate)

    Formula
    - Win rate = (Profitable trades / Total trades)

    Additional tools to complement
    - Profit factor

    Args:
    - trades: List of trade PnL values
    """
    if not trades:
        return 0.0
    winning_trades = sum(1 for t in trades if t > 0)
    return winning_trades / len(trades)


def calculate_profit_factor(trades: List) -> float:
    """
    Profit factor

    Formula
    - Profit factor = Total Gross profit / Total Gross Loss
    """
    if not trades:
        return 0.0

    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))

    return safe_divide(gross_profit, gross_loss, default=0.0)


# === DATE AND TIME UTILITIES ===


def is_market_hours(timestamp: datetime) -> bool:
    """
    Check if the timestamp is during market hours

    US Stock Market Hours: 9:30 AM - 4:00 PM ET, Monday-Friday

    Days in the week represented by Indexs where Monday = 0 and Sunday = 6

    Why check market hours
    - Avoid trading when market is closed
    - Filter data to relevant time periods
    - Professional trading practice
    """
    if timestamp.weekday() >= 5:
        return False

    market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= timestamp <= market_close


def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """

    Explanation of code
    - current_date += timedelta(days=1)
     - timedelta(days=1) basically creates a fixed, temporary duration of time

    Why seaprate trading days?
    - Markets closed on weekends
    - Some strategies only need buisness days
    - Accurate backtesting requires correct dates
    """
    trading_days = []
    current_date = start_date

    while current_date <= end_date:
        if current_date.weekday() < 5:
            trading_days.append(current_date)
        current_date += timedelta(days=1)

    return trading_days


# === DATA VALIDATION AND CLEANING ===


def validate_stock_data(data: pd.DataFrame) -> bool:
    """
    Validate that stock data has required columns and reasonable values

    Typical stock data structure:
                        Open    High    Low     Close   Volume
        2024-01-01     100.0   102.0   99.0    101.0   1000000
        2024-01-02     101.0   103.0   100.0   102.0   1200000

    Why validate data?
    - API Data could be incomplete
    - Prevent calculation on bad data
    - Better error messages for users
    - Defensive programming practice
    """
    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    # CHECKING FOR CORRECT COLUMNS:
    if not all(col in data.columns for col in required_columns):
        return False

    # True if Datafram has no rows or columns
    if data.empty:
        return False

    price_columns = ["Open", "High", "Low", "Close"]
    for col in price_columns:
        if (data[col] <= 0).any():
            return False

    if (data["High"] < data["Low"]).any():
        return False

    if (data["Volume"] < 0).any():
        return False

    return True


def clean_stock_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare stock data for analysis

     Why clean data?
    - Remove or fix corrupted entries
    - Standardize data format
    - Handle missing values appropriately
    - Improve calculation accuracy

    Why forward fill and not interpolate?
    - Markets have "last traded price" concept
    - If data missing for a period, last price is still valid
    - Interpolation creates artificial price that never existed

    Explanation of code
    - "if col in cleaned_data.columns:"
     - Check whether column with such a name exist in the table
    - "pd.to_numeric"
     - Converting data that looks like a number into a format Python can do math in
    - , errors='coerce'
     - When unable to convert the data, Python replaces bad data with NaN
    """

    cleaned_data = data.copy()

    # Dropping rows where all values are NaN
    cleaned_data = cleaned_data.dropna(how="all")

    # Forward Filling
    cleaned_data = cleaned_data.ffill()

    # Dropping any remaining NaN values
    cleaned_data = cleaned_data.dropna()

    # Conversion to format Python can math
    price_columns = ["Open", "High", "Low", "Close"]
    for col in price_columns:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors="coerce")

    return cleaned_data


# === PORTFOLIO UTILITIES ===


def calculate_position_size(
    portfolio_value: float, target_percentage: float, stock_price: float, max_shares: Optional[int] = None
) -> int:
    """
    Calculate appropriate position size based on portfolio allocation

    Args:
     - portfolio_value: Total portfolio value
     - target_percentage: Desired allocation (e.g., 0.1 for 10%)
     - stock_price: Current stock price
     - max_shares: Hard cap on shares

    Returns:
    - Number of shares to buy

    Why position sizing?
     - Risk management (don't put all money in one stock)
     - Portfolio diversification
     - Professional money management
     - Prevents overconcentration

    """
    if stock_price == 0:
        return 0

    target_amount = portfolio_value * target_percentage

    shares = int(target_amount / stock_price)

    if max_shares is not None and shares > max_shares:
        shares = max_shares

    return max(0, shares)


def calculate_stop_loss_price(entry_price: float, atr_value: float, multiplier: float = 2.0) -> float:
    """
    ATR-Dynamic Stop Loss

    Formula
    - Stop Loss = Entry Price - (ATR * Multiplier)

    Args:
        entry_price (float): Price at which position was entered
        atr_value (float): Current ATR value for stock
        multiplier (float, optional): How many units of ATR to risk, Defaults to 2.0.

    Returns:
        float: Stop loss price level
    """

    return entry_price - (atr_value * multiplier)


# === LOGGING AND ERROR HANDLING ===

"""
3 MAIN LOGGING AND ERROR HANDLING

1) Safely dividing 2 numbers and avoiding division by 0
2) Formatting of number as currency string
 - Reason being computers are bad at math with decimals
 - String formatting adds context
3) Formatting of decimals as percentage string

Explanation of code
- return f"${amount:,.2f}" for currency string
 - :, = Tell python to add thousands separator
 - .2 = Forces exactly 2 decimal places
 -f = Fixed-point notation
- return f"{value * 100:.2f}%"
 - value * 100 = Move decimal point 2 places to right since we are converting decimal to percentage
 - :.2f = Rounding result to 2 decimal places

"""


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"
