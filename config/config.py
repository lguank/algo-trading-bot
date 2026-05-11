"""
Configuration settings for the Algorithmic Trading Bot.

Design Philosophy:
- Centralise all configurable parameters in one place
- Class-based configuration for easy inheritance and modification
- Validation to catch configuration errors early
- Separate concerns: portfolio, risk, data, API settings

Structure:
    Config              ← base class with all default values
    DevelopmentConfig   ← overrides for local development/testing
    ProductionConfig    ← overrides for live/paper trading
    get_config()        ← factory function that selects the right one
"""

import os
from typing import List


class Config:
    """
    Base configuration class.

    Why a class instead of a dictionary or plain variables?
    - Inheritance: DevelopmentConfig can override just the values it needs
    - Methods: can add validate(), get_data_dir() alongside the settings
    - Type safety: attributes have clear names and expected types
    - Single source of truth: every other module imports from here
    """

    # ── Environment Identity ───────────────────────────────────────
    # Each subclass overrides this so you always know which config is active
    # This is what caused your smoke test failure - it was missing entirely
    ENV = "base"

    # ── Portfolio Settings ─────────────────────────────────────────
    INITIAL_CAPITAL: float = 100_000.0
    # Why $100k?
    # - Realistic for a paper trading account
    # - Large enough to buy meaningful quantities of most stocks
    # - Round number makes percentage calculations easy to verify
    # - Common starting point in quant finance courses

    PORTFOLIO_DATA_PATH: str = "data/portfolio_data.json"
    # Why JSON?
    # - Human readable: you can open it in any text editor
    # - Native Python support via the json module (no extra libraries)
    # - Easy to inspect and debug during development
    # - Simple enough for a single-user bot (database overkill here)

    # ── Risk Management ────────────────────────────────────────────
    MAX_POSITION_SIZE: float = 0.20
    # Why 20%?
    # - Hard limit: never put more than 20% of portfolio in one stock
    # - Allows minimum 5 positions (basic diversification)
    # - Professional portfolio management rarely exceeds 10-15% per name
    # - 20% is generous for a learning/demo bot

    TRANSACTION_COST: float = 9.99
    # Why $9.99 flat fee instead of a percentage?
    # - Matches real broker fees (many charge $0-$10 per trade)
    # - Flat fee penalises small trades more (realistic behaviour)
    # - Percentage (0.001) would be $0.10 on a $100 trade - unrealistically cheap
    # - Makes the backtest engine calculation unambiguous:
    #   cost = (quantity × price) + TRANSACTION_COST

    STOP_LOSS_PCT: float = 0.05
    # Why 5%?
    # - Cuts losing positions before they become serious losses
    # - 5% is within normal daily volatility for most stocks
    #   so it avoids being triggered by random noise
    # - Professional traders often use ATR-based stops (we do this too
    #   in helpers.py calculate_stop_loss_price) but a percentage
    #   fallback is useful for simple strategies

    # ── Universe of Stocks ─────────────────────────────────────────
    SYMBOLS: List[str] = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]
    # Why these symbols?
    # - All are highly liquid (easy to get clean data from yfinance)
    # - Mix of sectors: tech (AAPL, GOOGL, MSFT), growth (TSLA), index (SPY)
    # - SPY as benchmark: lets you compare bot performance vs market
    # - High daily volume means prices are less susceptible to manipulation

    DEFAULT_SYMBOLS: List[str] = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]
    # Alias kept for backward compatibility with any code that
    # references DEFAULT_SYMBOLS instead of SYMBOLS

    # ── Data Settings ──────────────────────────────────────────────
    DATA_PERIOD: str = "1y"
    # Why 1 year?
    # - Enough history for all indicators (SMA50 needs 50+ days minimum)
    # - Covers different market conditions (bull runs, corrections)
    # - Within yfinance free tier limits
    # - Balances computation time vs data richness

    DATA_INTERVAL: str = "1d"
    # Why daily ("1d") bars?
    # - Matches the frequency of most technical indicators (RSI-14 = 14 days)
    # - Reduces noise compared to intraday (1m, 5m, 1h) data
    # - Manageable data volume for development
    # - Most academic backtesting research uses daily data
    # Options: "1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"

    # ── API Settings ───────────────────────────────────────────────
    MAX_API_CALLS_PER_MINUTE: int = 100
    # Why rate limit?
    # - yfinance scrapes Yahoo Finance - aggressive calling can get you blocked
    # - Good defensive programming practice for any external API
    # - Prevents accidental infinite loops from hammering the server

    CACHE_DURATION_MINUTES: int = 5
    # Why cache for 5 minutes?
    # - Stock prices update every second during market hours
    # - But for daily-bar strategies, a 5-minute old price is fine
    # - Reduces API calls during development (run bot multiple times quickly)
    # - Respects the data provider's infrastructure

    # ── File Paths ─────────────────────────────────────────────────
    LOG_FILE_PATH: str = "data/trading_log.txt"

    # ── Class Methods ──────────────────────────────────────────────

    @classmethod
    def get_data_dir(cls) -> str:
        """
        Return the data directory path, creating it if it does not exist.

        Why @classmethod instead of a plain string?
        - Can contain logic (directory creation)
        - Works correctly even when called from a subclass
          e.g. DevelopmentConfig.get_data_dir() still works
        - Handles cross-platform path differences via os module
        """
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"[Config] Created data directory: {data_dir}")
        return data_dir

    @classmethod
    def validate(cls) -> None:
        """
        Validate that all settings are within acceptable ranges.

        Why validate at startup?
        - Catches mistakes early (misconfigured INITIAL_CAPITAL = -1000)
        - Gives a clear error message instead of a cryptic crash later
        - Professional practice: fail fast with a helpful message
        - Especially important when config is loaded from environment
          variables (which are always strings and need type checking)

        Call this in main.py at startup:
            config = get_config()
            config.validate()
        """
        errors = []

        if cls.INITIAL_CAPITAL <= 0:
            errors.append(f"INITIAL_CAPITAL must be positive, got {cls.INITIAL_CAPITAL}")

        if cls.TRANSACTION_COST < 0:
            errors.append(f"TRANSACTION_COST cannot be negative, got {cls.TRANSACTION_COST}")

        if not 0 < cls.MAX_POSITION_SIZE <= 1:
            errors.append(f"MAX_POSITION_SIZE must be between 0 and 1, " f"got {cls.MAX_POSITION_SIZE}")

        if not cls.SYMBOLS:
            errors.append("SYMBOLS list cannot be empty")

        if cls.DATA_PERIOD not in ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]:
            errors.append(f"DATA_PERIOD '{cls.DATA_PERIOD}' is not a valid yfinance period")

        if cls.DATA_INTERVAL not in ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]:
            errors.append(f"DATA_INTERVAL '{cls.DATA_INTERVAL}' is not a valid yfinance interval")

        # If any errors found, report all of them at once
        if errors:
            error_msg = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(f"[Config] Validation failed:\n{error_msg}")

        print(f"[Config] ✅ Validation passed ({cls.ENV} environment)")

    @classmethod
    def display(cls) -> None:
        """
        Print a summary of all active settings.
        Useful to call at bot startup so logs show exact config used.
        """
        print(f"\n{'─' * 45}")
        print(f"  ACTIVE CONFIGURATION: {cls.ENV.upper()}")
        print(f"{'─' * 45}")
        print(f"  Initial Capital   : ${cls.INITIAL_CAPITAL:,.2f}")
        print(f"  Max Position Size : {cls.MAX_POSITION_SIZE:.0%}")
        print(f"  Transaction Cost  : ${cls.TRANSACTION_COST:.2f}")
        print(f"  Stop Loss         : {cls.STOP_LOSS_PCT:.0%}")
        print(f"  Symbols           : {cls.SYMBOLS}")
        print(f"  Data Period       : {cls.DATA_PERIOD}")
        print(f"  Data Interval     : {cls.DATA_INTERVAL}")
        print(f"{'─' * 45}\n")


# ================================================================
# ENVIRONMENT-SPECIFIC CONFIGS
# Each subclass only overrides what is different
# Everything not listed here is inherited from Config above
# ================================================================


class DevelopmentConfig(Config):
    """
    Settings for local development and testing.

    Philosophy: make it easy to test quickly
    - Smaller capital so mistakes are obvious but not alarming
    - Larger position size to force more trades in backtests
    - Shorter data period for faster test runs
    - Dev-specific file paths to avoid overwriting real data

    Why inherit from Config instead of rewriting everything?
    - Only need to specify what CHANGES
    - If you add a new setting to Config, it automatically
      appears in DevelopmentConfig without any extra work
    - Classic object-oriented inheritance benefit
    """

    ENV = "dev"

    INITIAL_CAPITAL: float = 10_000.0  # smaller for testing
    PORTFOLIO_DATA_PATH: str = "data/portfolio_dev.json"  # separate file from prod
    MAX_POSITION_SIZE: float = 0.25  # slightly larger to force more signals
    DATA_PERIOD: str = "6mo"  # shorter for faster test runs
    TRANSACTION_COST: float = 9.99  # same as prod (realistic)


class ProductionConfig(Config):
    """
    Settings for live paper trading or real trading.

    Philosophy: be conservative and protect capital
    - Full capital amount
    - Smaller position sizes (tighter risk management)
    - Tighter stop loss
    - Full data period for richer indicator calculations
    """

    ENV = "prod"

    INITIAL_CAPITAL: float = 100_000.0
    PORTFOLIO_DATA_PATH: str = "data/portfolio_prod.json"
    MAX_POSITION_SIZE: float = 0.10  # more conservative than dev
    STOP_LOSS_PCT: float = 0.03  # tighter stop loss in prod
    DATA_PERIOD: str = "1y"  # full year of data


# ================================================================
# FACTORY FUNCTION
# ================================================================


def get_config(environment: str = "development") -> Config:
    """
    Return the correct Config instance based on environment string.

    Why a factory function instead of just importing the class directly?
    - Single decision point: change environment in one place
    - Supports environment variables (twelve-factor app pattern)
    - Easy to add new environments later (StagingConfig, TestConfig)
    - Other modules never need to know which class to instantiate

    Usage:
        from config.config import get_config
        config = get_config()           # uses "development" by default
        config = get_config("production")

    Or via environment variable (useful for deployment):
        TRADING_ENV=production python main.py

    Args:
        environment : "development" or "production" (case insensitive)

    Returns:
        Config instance with appropriate settings
    """
    # Also check environment variable so you can set it externally
    # os.getenv() reads from your terminal/system environment
    # The second argument is the fallback if the variable is not set
    env = os.getenv("TRADING_ENV", environment).lower()

    if env == "production" or env == "prod":
        config = ProductionConfig()
    else:
        config = DevelopmentConfig()

    return config
