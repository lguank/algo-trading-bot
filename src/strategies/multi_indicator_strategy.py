"""
Multi-Indicator Confluence Strategy

Relationship to base_strategy.py:
- Imports BaseStrategy, Signal, BacktestResult FROM base_strategy.py
- MultiIndicatorStrategy INHERITS from BaseStrategy
- This means it gets backtest(), calculate_position_size(),
  should_buy(), should_sell() FOR FREE
- Only needs to implement generate_signal() (the @abstractmethod)

Design pattern used: Template Method Pattern
- BaseStrategy defines the SKELETON (backtest loop, position sizing)
- MultiIndicatorStrategy fills in the DETAILS (how to generate signals)

Why not put everything in one file?
- base_strategy.py = the CONTRACT (stable, rarely changes)
- multi_indicator_strategy.py = the IMPLEMENTATION (you will tune this often)
- Separating them means you can add new strategies without ever
  touching the backtesting engine
"""

from dataclasses import dataclass, field
from datetime import datetime
from tkinter import N
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from cycler import K

# Import contract we must fufil
from src import data
from src.strategies.base_strategy import BaseStrategy, Signal

# Import pure math functions from helpers
from src.utils.helpers import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_exponential_moving_average,
    calculate_macd,
    calculate_moving_average,
    calculate_obv,
    calculate_rsi,
    calculate_stochastic_oscillator,
    calculate_stop_loss_price,
    calculate_vwap,
    format_currency,
    safe_divide,
)

# === SUPPORTING DATA CLASSES ===
# These are specific to multi-indicator system
# Do not belong in base_strategy.py becacuse simpler strategies like MA crossover do need them


@dataclass
class IndicatorVote:
    """
    Indicator verdict on market direction

    Mechanism
    1) Each indicator in confluence system casts a vote
    - Bullish = +1
    - Bearish = -1
    - Neutral = 0
    2) Vote multipied by weight of indicator to get contribution to final score

    Difference in weight of indicators
    - MACD and RSI combines multiple concepts like trend and momentum
    - OBV and VWAP can be noisy on daily data
    - Allows us to alter weight via backtesting
    """

    vote: int
    weight: float
    reason: str  # Human-readable explanations
    value: float = 0.0

    @property
    def weighted_vote(self) -> float:
        """
        Actual contribution of indicator to final score.
        Gets summed to produce the confluence score
        """
        return self.vote * self.weight


@dataclass
class SignalReport:
    """
    Breakdown of indicator's vote for one analysis
    """

    symbol: str
    price: float
    votes: Dict[str, IndicatorVote] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        """
        Sum of all weighted votes

        Range depends on which indicators are active:
        With all 8 indicators,
            Max = +9 (All Bullish)
            Min = -9 (All Bearish)
        """
        return sum(v.weighted_vote for v in self.votes.values())

    @property
    def max_possible_score(self) -> float:
        """
        Score if every indicator voted bullish
        Used to express score as a percentage
        """
        return sum(v.weight for v in self.votes.values())

    @property
    def bullish_percentage(self) -> float:
        """
        How bullish is the overall picture expressed as 0-100%
        50% = perfectly neutral, 100% = maximum bullish

        To map score from [-max, +max] range to [0,100]
        - Add 1 to shift range from [-1,1] to [0,2]
        - Divide by 2 to shift from [0,2] to [0,1]
        - Multiply by 100 for a percentage
        """

        if self.max_possible_score == 0:
            return 50.0
        ratio = self.total_score / self.max_possible_score
        return ((ratio + 1) / 2) * 100

    def get_signal(self, buy_threshold: float, sell_threshold: float) -> str:
        """
        Convert numeric score into BUY / SELL / HOLD string
        """
        if self.total_score >= buy_threshold:
            return "BUY"
        elif self.total_score <= sell_threshold:
            return "SELL"
        return "HOLD"

    def to_dict(self) -> dict:
        """
        Serialise for web API JSON response
        Called by flask route
        """
        return {
            "symbol": self.symbol,
            "price": self.price,
            "total_score": round(self.total_score, 2),
            "max_score": round(self.max_possible_score, 2),
            "bullish_pct": round(self.bullish_percentage, 1),
            "votes": {
                name: {
                    "vote": v.vote,
                    "weight": v.weight,
                    "weighted_vote": round(v.weighted_vote, 2),
                    "reason": v.reason,
                    "value": round(v.value, 4),
                }
                for name, v in self.votes.items()
            },
        }

    def print_report(self) -> None:
        """
        Print the full breakdown to console
        Called during live trading and backtesting
        """
        print(f"\n  {'─' * 55}")
        print(f"  SIGNAL REPORT │ {self.symbol} │ {format_currency(self.price)}")
        print(f"  {'─' * 55}")
        print(f"  {'Indicator':<16} {'Vote':>5}  {'Weight':>7}  " f"{'Score':>7}  Reason")
        print(f"  {'─' * 55}")

        for name, v in self.votes.items():
            icon = "🟢" if v.vote > 0 else "🔴" if v.vote < 0 else "⚪"
            print(f"  {icon} {name:<14} {v.vote:>+5}  " f"{v.weight:>7.1f}  {v.weighted_vote:>+7.2f}  " f"{v.reason}")

        print(f"  {'─' * 55}")
        print(
            f"  TOTAL SCORE: {self.total_score:+.2f} / "
            f"±{self.max_possible_score:.1f}  "
            f"({self.bullish_percentage:.0f}% bullish)"
        )
        print(f"  {'─' * 55}")


# === THE STRATEGY CLASS ===
# Inherits from Base Strategy, implements generate_signal()


class MultiIndicatorStrategy(BaseStrategy):
    """
    Confluence strategy combining 8 indicators across 4 categories

     Inheritance chain:
        MultiIndicatorStrategy
            └── BaseStrategy (ABC)
                    └── provides: backtest(), calculate_position_size(),
                                  should_buy(), should_sell()

    The only method this class MUST implement is generate_signal()
    because that is the @abstractmethod in BaseStrategy.

    Indicator weights(Total max score is 9):

    │   Indicator     │ Weight │            Rationale             │
    ├─────────────────┼────────┼──────────────────────────────────┤
    │ MACD            │  1.5   │ Trend + momentum combined        │
    │ RSI             │  1.5   │ Universally reliable momentum    │
    │ EMA Crossover   │  1.2   │ Faster trend signal than SMA     │
    │ Stochastic      │  1.2   │ Short-term momentum, RSI confirm │
    │ SMA Crossover   │  1.0   │ Classic trend, slightly lagging  │
    │ Bollinger Bands │  1.0   │ Volatility + price extreme check │
    │ OBV             │  0.8   │ Volume confirmation              │
    │ VWAP            │  0.8   │ Institutional price level        │

    Thresholds
    - BUY when score >= 2.5 (need about 28% of bullish score to buy)
    - SELL when score <= 2.5 (need about 28% of bearish score to sell)
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        buy_threshold: float = 2.5,
        sell_threshold: float = -2.5,
        # Trend parameters
        sma_short: int = 20,
        sma_long: int = 50,
        ema_short: int = 12,
        ema_long: int = 26,
        # Momentum parameters
        rsi_period: int = 14,
        rsi_overbought: float = 65.0,
        rsi_oversold: float = 35.0,
        # Volatility parameters
        bb_window: int = 20,
        atr_period: int = 14,
    ):
        # Call BaseStrategy.__init__() to set up name, capital and config
        super().__init__("MultiIndicator", initial_capital)

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bb_window = bb_window
        self.atr_period = atr_period

    # === REQUIRED IMPLEMENTATION OF ABSTRACT METHOD ===
    def generate_signal(self, symbol: str, data: pd.DataFrame, current_price: float, portfolio_info: Dict) -> Signal:
        """
        Implements @abstractmethod from BaseStrategy

        Flow
        1) Check if we have enough data for all indicators
        2) Build SignalReport (evaluate every indicator)
        3) Print breakdown
        4) Convert score to BUY/SELL/HOLD
        5) Package into Signal dataclass and return
        """

        # Most demanding indicator is SMA(50) + buffer of a few bars
        min_bars = self.sma_long + 10

        if len(data) < min_bars:
            return Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                price=current_price,
                quantity=0,
                reason=(f"Insufficient data: Have {len(data)} bars, need {min_bars}"),
            )

        # Build the report
        report = self._build_report(symbol, data, current_price)

        # Only print during live trading
        # Backtesting loops over many days, leading to too much output
        if not self.is_backtesting:
            report.print_report()

        signal_str = report.get_signal(self.buy_threshold, self.sell_threshold)

        # Confidence = how far the score is from zero, capped at 1
        # Example: score=5.0, while max=9.0. confidence = 5/9 = 0.56

        confidence = (
            min((report.total_score / report.max_possible_score), 1.0) if report.max_possible_score > 0.0 else 0.0
        )

        # Determine quantity
        if signal_str == "BUY":
            quantity = self.calculate_position_size(symbol, current_price, portfolio_info["total_value"])
        elif signal_str == "SELL":
            # Sell entire existing position
            quantity = portfolio_info.get("positions", {}).get(symbol, 0)
        else:
            quantity = 0

        if quantity > 0:
            quantity = max(quantity, 1)  # Force min 1 share if any calculated

        if not self.is_backtesting:
            print(f"➤  DECISION: {signal_str} | Confidence: {confidence:.0%}")

        return Signal(
            timestamp=datetime.now(),
            symbol=symbol,
            action=signal_str,
            confidence=confidence,
            price=current_price,
            quantity=quantity,
            reason=(
                f"Confluence score {report.total_score:+.2f} "
                f"(threshold: buy≥{self.buy_threshold}, "
                f"sell≤{self.sell_threshold})"
            ),
        )

    def get_full_analysis(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Extended method used by web API
        Returns everything needed to render analysis dashboard

        Exists only to give web layer rich data
        """

        if data is None or data.empty:
            return {"error": f"No data avaliable for {symbol}"}

        current_price = float(data["Close"].iloc[-1])
        report = self._build_report(symbol, data, current_price)
        signal_str = report.get_signal(self.buy_threshold, self.sell_threshold)

        # ATR-Based stop loss for risk management display
        atr = calculate_atr(data["High"], data["Low"], data["Close"], self.atr_period)
        atr_val = float(atr.iloc[-1])

        return {
            "symbol": symbol,
            "signal": signal_str,
            "score": round(report.total_score, 2),
            "bullish_pct": round(report.bullish_percentage, 1),
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "atr": round(atr_val, 2),
            "stop_loss": round(calculate_stop_loss_price(current_price, atr_val), 2),
            "votes": report.to_dict()["votes"],
            "price_history": data["Close"].tail(30).round(2).tolist(),
        }

    # === Orchestrating of all indicator evaluations ===
    def _build_report(self, symbol: str, data: pd.DataFrame, current_price: float) -> SignalReport:
        """
        Evaluate every indicator and collect their vote into SignalReport

        Private method used to ensure generate_signal() is clean
        Individual indicators are also private, allowing one function to only do one thing
        """

        report = SignalReport(symbol=symbol, price=current_price)

        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        # Check if volume data exists
        has_volume = "Volume" in data.columns and not data["Volume"].isna().all()
        volume = data["Volume"] if has_volume else None

        # Trend Indicators
        report.votes["SMA Cross"] = self._eval_sma(close)
        report.votes["EMA Cross"] = self._eval_ema(close)
        report.votes["MACD"] = self._eval_macd(close)

        # Momentum Indicators
        report.votes["RSI"] = self._eval_rsi(close)
        report.votes["Stochastic"] = self._eval_stochastic(high, low, close)

        # Volatility Indicators
        report.votes["Bollinger"] = self._eval_bollinger(close)

        # Volume indicators (Only if there is volume data)
        if has_volume:
            report.votes["OBV"] = self._eval_obv(close, volume)
            report.votes["VWAP"] = self._eval_vwap(high, low, close, volume)

        return report

    # === INDIVIDUAL INDICATOR EVALUATORS ===
    # Each method returns one IndicatorVote
    # Each method is private and self-contained

    def _eval_sma(self, close: pd.Series) -> IndicatorVote:
        """
        SMA Crossover Vote

        Bullish when fast SMA is above slow SMA
        """
        sma_s = calculate_moving_average(close, self.sma_short)
        sma_l = calculate_moving_average(close, self.sma_long)

        val_s = float(sma_s.iloc[-1])
        val_l = float(sma_l.iloc[-1])

        if val_s > val_l:
            return IndicatorVote(
                vote=1,
                weight=1.0,
                value=val_s,
                reason=f"SMA{self.sma_short} ({val_s:.2f}) > " f"SMA{self.sma_long} ({val_l:.2f}) → Uptrend",
            )
        return IndicatorVote(
            vote=-1,
            weight=1.0,
            value=val_s,
            reason=f"SMA{self.sma_short} ({val_s:.2f}) < " f"SMA{self.sma_long} ({val_l:.2f}) → Downtrend",
        )

    def _eval_ema(self, close: pd.Series) -> IndicatorVote:
        """
        EMA Crossover Vote

        Same logic as SMA but react faster to price changes = Higher weight given
        """
        ema_s = calculate_exponential_moving_average(close, self.ema_short)
        ema_l = calculate_exponential_moving_average(close, self.ema_long)

        val_s = float(ema_s.iloc[-1])
        val_l = float(ema_l.iloc[-1])

        if val_s > val_l:
            return IndicatorVote(
                vote=1,
                weight=1.2,
                value=val_s,
                reason=f"EMA{self.ema_short} ({val_s:.2f}) > " f"EMA{self.ema_long} ({val_l:.2f}) → Bullish Momentum",
            )
        return IndicatorVote(
            vote=-1,
            weight=1.2,
            value=val_s,
            reason=f"EMA{self.ema_short} ({val_s:.2f}) < " f"EMA{self.ema_long} ({val_l:.2f}) → Bearish Momentum",
        )

    def _eval_macd(self, close: pd.Series) -> IndicatorVote:
        """
        MACD Vote

        Check for histogram crossover (strongest signal)

        Why prioritise crossover?
        - Crossover = Momentum shift direction right now
        - Above/below zero = momentum HAS BEEN in this direction
        """

        macd_line, signal_line, histogram = calculate_macd(close)

        hist_now = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2])
        macd_val = float(macd_line.iloc[-1])

        # Histogram crossed from negative to positive
        if hist_prev < 0 and hist_now > 0:
            return IndicatorVote(
                vote=1, weight=1.5, value=macd_val, reason=f"MACD histogram bullish crossover ({hist_now:+.4f})"
            )

        # Histogram crossed from postive to negative
        if hist_prev > 0 and hist_now < 0:
            return IndicatorVote(
                vote=-1, weight=1.5, value=macd_val, reason=f"MACD histogram bearish crossover ({hist_now:+.4f})"
            )

        # No crossover - use current histogram position
        if hist_now > 0:
            return IndicatorVote(
                vote=1, weight=1.5, value=macd_val, reason=f"MACD histogram positive ({hist_now:+.4f})"
            )

        if hist_now < 0:
            return IndicatorVote(
                vote=-1, weight=1.5, value=macd_val, reason=f"MACD histogram negative({hist_now:+.4f})"
            )

    def _eval_rsi(self, close: pd.Series) -> IndicatorVote:
        """
        RSI vote

        Extreme readings like overbought and oversold are strongest signals
        Above/below 50 midline gives a weak directional bias

        Note: Current oversold=35 and overbought=65 intentionally to generate more signals comapred
        to class 30/70 thresholds
        """

        rsi_series = calculate_rsi(close, self.rsi_period)
        rsi_val = float(rsi_series.iloc[-1])

        if rsi_val <= self.rsi_oversold:
            return IndicatorVote(
                vote=1,
                weight=1.5,
                value=rsi_val,
                reason=(f"RSI {rsi_val:.1f} ≤ {self.rsi_oversold} " f"→ Oversold, expect bounce"),
            )

        if rsi_val >= self.rsi_overbought:
            return IndicatorVote(
                vote=-1,
                weight=1.5,
                value=rsi_val,
                reason=(f"RSI {rsi_val:.1f} ≥ {self.rsi_overbought} " f"→ Overbought, expect pullback"),
            )

        if rsi_val >= 50:
            return IndicatorVote(
                vote=1,
                weight=1.5,
                value=rsi_val,
                reason=(f"RSI {rsi_val:.1f} in bullish zone (≥ 50)"),
            )

        if rsi_val < 50:
            return IndicatorVote(
                vote=-1,
                weight=1.5,
                value=rsi_val,
                reason=(f"RSI {rsi_val:.1f} in bearish zone (<50)"),
            )

    def _eval_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series) -> IndicatorVote:
        """
        Stochastic Oscillator Vote

        Tiers of signal strength (in order)
        - %K%D crossover in extreme zones (80 and 20 zones)
        - %K in extreme zone without crossover
        - %K above/below 50 midline (only shows directional bias)
        """

        k_line, d_line = calculate_stochastic_oscillator(high, low, close)

        k_now = float(k_line.iloc[-1])
        k_prev = float(k_line.iloc[-2])
        d_now = float(d_line.iloc[-1])
        d_prev = float(d_line.iloc[-1])

        # Crossover in oversold zone (strongest BUY)
        if k_now < 20 and k_now > d_now and k_prev <= d_prev:
            return IndicatorVote(
                vote=1,
                weight=1.2,
                value=k_now,
                reason=(f"Stochastic %K ({k_now:.1f}) crossed above %D " f"in oversold zone → Strong reversal signal"),
            )

        # Crossover in overbought zone (Strongest SELL)
        if k_now > 80 and k_now < d_now and k_prev >= d_prev:
            return IndicatorVote(
                vote=-1,
                weight=1.2,
                value=k_now,
                reason=(
                    f"Stochastic %K ({k_now:.1f}) crossed below %D " f"in overbought zone → Strong reversal signal"
                ),
            )

        # Midline position
        if k_now >= 50:
            return IndicatorVote(
                vote=1, weight=1.2, value=k_now, reason=(f"Stochastic %K ({k_now:.1f}) above 50 → Bullish")
            )

        if k_now < 50:
            return IndicatorVote(
                vote=-1, weight=1.2, value=k_now, reason=(f"Stochastic %K ({k_now:.1f}) below 50 → Bearish")
            )

    def _eval_bollinger(self, close: pd.Series) -> IndicatorVote:
        """
        Bollinger Band Vote

        Price at the bands = Volatility extreme = Potential reversal
        Price above/below midline = trend direction confirmed
        """

        upper, middle, lower = calculate_bollinger_bands(close, self.bb_window)

        price = float(close.iloc[-1])
        upper_val = float(upper.iloc[-1])
        lower_val = float(lower.iloc[-1])
        middle_val = float(middle.iloc[-1])

        # Extreme BUY Signal (should check with RSI)
        if price <= lower_val:
            return IndicatorVote(
                vote=1,
                weight=1.0,
                value=price,
                reason=(f"Price ({price:.2f}) at/below lower BB ({lower_val:.2f}) " f"→ Oversold extreme"),
            )

        # Extreme SELL signal
        elif price >= upper_val:
            return IndicatorVote(
                vote=-1,
                weight=1.0,
                value=price,
                reason=(f"Price ({price:.2f}) at/above upper BB ({upper_val:.2f}) " f"→ Overbought extreme"),
            )

        # Price above midline
        elif price >= middle_val:
            return IndicatorVote(
                vote=1,
                weight=1.0,
                value=price,
                reason=(f"Price ({price:.2f}) above BB midline ({middle_val:.2f}) " f"→ Mild bullish"),
            )

        # Price below midline
        else:
            return IndicatorVote(
                vote=-1,
                weight=1.0,
                value=price,
                reason=(f"Price ({price:.2f}) below BB midline ({middle_val:.2f}) " f"→ Mild bearish"),
            )

    def _eval_obv(self, close: pd.Series, volume: pd.Series) -> IndicatorVote:
        """
        On balance volume

        Compares OBV now to 5 bars ago to determine the volume trend

        Why 5 bars?
        - 1 bar is too noisy due to possible single day volume spike
        - 10+ too slow to catch recent shifts
        - 5 bars represent a single trading week (Balance responsiveness and noise)

        Potential improvement
        - Combining OBV checks with whether the price remain relatively flat or has increased
        """

        obv = calculate_obv(close, volume)
        obv_now = float(obv.iloc[-1])
        obv_ago = float(obv.iloc[-5])

        if obv_now > obv_ago:
            return IndicatorVote(
                vote=1,
                weight=0.8,
                value=obv_now,
                reason=(f"OBV rising ({obv_ago:,.0f} → {obv_now:,.0f}) " f"→ Accumulation (buying pressure)"),
            )

        if obv_now < obv_ago:
            return IndicatorVote(
                vote=-1,
                weight=0.8,
                value=obv_now,
                reason=(f"OBV falling ({obv_ago:,.0f} → {obv_now:,.0f}) " f"→ Distribution (selling pressure)"),
            )

    def _eval_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> IndicatorVote:
        """
        VWAP Vote

        Price above VWAP = Institutions willing to pay above average = Bullish
        Price below VWAP = price below institutional average = Bearish
        """

        vwap = calculate_vwap(high, low, close, volume)
        price = float(close.iloc[-1])
        vwap_val = float(vwap.iloc[-1])

        if price > vwap_val:
            return IndicatorVote(
                vote=1,
                weight=0.8,
                value=vwap_val,
                reason=(f"Price ({price:.2f}) above VWAP ({vwap_val:.2f}) " f"→ Bullish institutional sentiment"),
            )

        if price < vwap_val:
            return IndicatorVote(
                vote=-1,
                weight=0.8,
                value=vwap_val,
                reason=(f"Price ({price:.2f}) below VWAP ({vwap_val:.2f}) " f"→ Bearish institutional sentiment"),
            )
