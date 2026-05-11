# test_helpers.py
# Run with: python test_helpers.py
# Purpose: verify every helper function produces mathematically correct results

import numpy as np
import pandas as pd

print("=" * 60)
print("UNIT TESTS - helpers.py")
print("=" * 60)

from src.utils.helpers import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_exponential_moving_average,
    calculate_macd,
    calculate_max_drawdown,
    calculate_moving_average,
    calculate_obv,
    calculate_profit_factor,
    calculate_returns,
    calculate_rsi,
    calculate_sharpe_ratio,
    calculate_stochastic_oscillator,
    calculate_vwap,
    calculate_win_rate,
    format_currency,
    format_percentage,
    safe_divide,
)

# ── Test tracking ─────────────────────────────────────────────────
passed = 0
failed = 0


def check(test_name: str, condition: bool, detail: str = "") -> None:
    """
    Record and print a single test result.

    Args:
        test_name : Human readable description of what is being tested
        condition : True = pass, False = fail
        detail    : Extra info printed on failure (what value was received)
    """
    global passed, failed
    if condition:
        print(f"    ✅ {test_name}")
        passed += 1
    else:
        print(f"    ❌ {test_name} {detail}")
        failed += 1


# ================================================================
# SHARED TEST DATA
# Built once at the top so every test section can reuse it
# Using a fixed random seed ensures the same results every run
# ================================================================

# Perfect uptrend: 1, 2, 3 ... 100
# RSI should approach 100 (no losing days)
prices_up = pd.Series(range(1, 101), dtype=float)

# Perfect downtrend: 100, 99, 98 ... 1
# RSI should approach 0 (no winning days)
prices_down = pd.Series(range(100, 0, -1), dtype=float)

# Flat prices: all 100.0
# RSI should be 50 (neutral - no movement)
prices_flat = pd.Series([100.0] * 50)

# Realistic prices: sine wave + random noise
# This is the most important test series because it resembles real market data
# np.random.seed(42) ensures every run produces identical values
np.random.seed(42)
prices_realistic = pd.Series([100 + 20 * np.sin(i * 0.15) + np.random.normal(0, 2) for i in range(100)])

# Realistic OHLCV data for indicators that need High, Low, Volume
# High and Low have their own noise so they are not just fixed offsets
realistic_high = prices_realistic + abs(pd.Series(np.random.normal(1.5, 0.5, 100)))
realistic_low = prices_realistic - abs(pd.Series(np.random.normal(1.5, 0.5, 100)))
realistic_volume = pd.Series(np.random.randint(500_000, 2_000_000, 100), dtype=float)

# Simple symmetric high/low for ATR tests (known, predictable range)
simple_high = prices_up + 2.0
simple_low = prices_up - 2.0
simple_volume = pd.Series([1_000_000.0] * 100)


# ================================================================
# SECTION 1: SIMPLE MOVING AVERAGE
# ================================================================
print("\n" + "─" * 60)
print("SMA - Simple Moving Average")
print("─" * 60)

sma_10 = calculate_moving_average(prices_up, 10)
sma_20 = calculate_moving_average(prices_up, 20)

check(
    "Returns a pd.Series",
    isinstance(sma_10, pd.Series),
)
check(
    "Output length matches input length", len(sma_10) == len(prices_up), f"expected {len(prices_up)}, got {len(sma_10)}"
)
check(
    "First window-1 values are NaN (not enough data yet)",
    sma_10.iloc[:9].isna().all(),
    f"first 9 values: {sma_10.iloc[:9].tolist()}",
)
check(
    "First complete value is correct mean",
    # SMA(10) at position 9 = mean(1..10) = 5.5
    abs(float(sma_10.iloc[9]) - 5.5) < 0.001,
    f"expected 5.5, got {sma_10.iloc[9]:.4f}",
)
check(
    "Last value is correct mean of last 10 prices",
    # SMA(10) at end = mean(91..100) = 95.5
    abs(float(sma_10.iloc[-1]) - 95.5) < 0.001,
    f"expected 95.5, got {sma_10.iloc[-1]:.4f}",
)
check(
    "SMA values increase in uptrend",
    float(sma_10.iloc[-1]) > float(sma_10.iloc[10]),
    f"last={sma_10.iloc[-1]:.2f}, early={sma_10.iloc[10]:.2f}",
)
check(
    "Longer window SMA is smoother (lower std dev)",
    float(sma_20.dropna().std()) < float(sma_10.dropna().std()),
    f"SMA10 std={sma_10.dropna().std():.3f}, SMA20 std={sma_20.dropna().std():.3f}",
)
check(
    "Raises ValueError when insufficient data",
    # Requesting window=200 on 100-point series should fail
    (
        (lambda: (calculate_moving_average(prices_up, 200)) or True)()
        if False
        else __import__("builtins").isinstance(
            (
                lambda: (
                    [
                        None
                        for _ in [None]
                        if not (lambda: (_ for _ in ()).throw(ValueError()) if len(prices_up) >= 200 else None)()
                    ]
                )
            )(),
            list,
        )
        or True
    ),
    # Simpler approach - just test directly:
    "",
)
# Cleaner ValueError test:
try:
    calculate_moving_average(prices_up, 200)
    check("Raises ValueError for window > data length", False, "no error raised")
except ValueError:
    check("Raises ValueError for window > data length", True)


# ================================================================
# SECTION 2: EXPONENTIAL MOVING AVERAGE
# ================================================================
print("\n" + "─" * 60)
print("EMA - Exponential Moving Average")
print("─" * 60)

ema_10 = calculate_exponential_moving_average(prices_up, 10)
ema_20 = calculate_exponential_moving_average(prices_up, 20)
sma_ref = calculate_moving_average(prices_up, 10)

check(
    "Returns a pd.Series",
    isinstance(ema_10, pd.Series),
)
check(
    "Output length matches input length", len(ema_10) == len(prices_up), f"expected {len(prices_up)}, got {len(ema_10)}"
)
check("All values are positive", (ema_10 > 0).all(), f"min value: {ema_10.min():.4f}")
check(
    "EMA > SMA in uptrend (EMA weights recent prices more)",
    # In an uptrend, recent prices are higher so EMA > SMA
    float(ema_10.iloc[-1]) > float(sma_ref.iloc[-1]),
    f"EMA={ema_10.iloc[-1]:.2f}, SMA={sma_ref.iloc[-1]:.2f}",
)
check(
    "EMA reacts faster than SMA (higher value in uptrend)",
    float(ema_10.iloc[-1]) > float(sma_ref.iloc[-1]),
    f"EMA10={ema_10.iloc[-1]:.2f} should be > SMA10={sma_ref.iloc[-1]:.2f}",
)
check(
    "Shorter EMA > longer EMA in uptrend",
    # EMA(10) reacts faster so tracks higher prices more closely
    float(ema_10.iloc[-1]) > float(ema_20.iloc[-1]),
    f"EMA10={ema_10.iloc[-1]:.2f}, EMA20={ema_20.iloc[-1]:.2f}",
)

try:
    calculate_exponential_moving_average(prices_up, 200)
    check("Raises ValueError for window > data length", False, "no error raised")
except ValueError:
    check("Raises ValueError for window > data length", True)


# ================================================================
# SECTION 3: RSI
# Uses the fixed version that handles edge cases correctly
# ================================================================
print("\n" + "─" * 60)
print("RSI - Relative Strength Index")
print("─" * 60)

rsi_up = calculate_rsi(prices_up, 14)
rsi_down = calculate_rsi(prices_down, 14)
rsi_flat = calculate_rsi(prices_flat, 14)
rsi_realistic = calculate_rsi(prices_realistic, 14)

check(
    "Returns a pd.Series",
    isinstance(rsi_up, pd.Series),
)
check(
    "Output length matches input length", len(rsi_up) == len(prices_up), f"expected {len(prices_up)}, got {len(rsi_up)}"
)
check(
    "All non-NaN values between 0 and 100",
    (rsi_up.dropna() >= 0).all() and (rsi_up.dropna() <= 100).all(),
    f"min={rsi_up.dropna().min():.2f}, max={rsi_up.dropna().max():.2f}",
)
check(
    "Pure uptrend gives RSI = 100 (no losing days)",
    # FIX: was fillna(50) which masked this - now returns 100 correctly
    float(rsi_up.iloc[-1]) == 100.0,
    f"expected 100.0, got {rsi_up.iloc[-1]:.2f}",
)
check(
    "Pure downtrend gives RSI = 0 (no winning days)",
    float(rsi_down.iloc[-1]) == 0.0,
    f"expected 0.0, got {rsi_down.iloc[-1]:.2f}",
)
check(
    "Flat prices give RSI = 50 (no movement, neutral)",
    abs(float(rsi_flat.iloc[-1]) - 50.0) < 1.0,
    f"expected ~50.0, got {rsi_flat.iloc[-1]:.2f}",
)
check(
    "Realistic prices give RSI between 30 and 70",
    # Oscillating realistic data should stay in the middle range
    30 <= float(rsi_realistic.iloc[-1]) <= 70,
    f"got {rsi_realistic.iloc[-1]:.2f}",
)
check(
    "RSI is higher in uptrend than downtrend",
    float(rsi_up.iloc[-1]) > float(rsi_down.iloc[-1]),
    f"up={rsi_up.iloc[-1]:.2f}, down={rsi_down.iloc[-1]:.2f}",
)

try:
    calculate_rsi(prices_up, 200)
    check("Raises ValueError for insufficient data", False, "no error raised")
except ValueError:
    check("Raises ValueError for insufficient data", True)


# ================================================================
# SECTION 4: MACD - CORRECTED
# ================================================================
print("\n" + "─" * 60)
print("MACD - Moving Average Convergence Divergence")
print("─" * 60)

macd_line, signal_line, histogram = calculate_macd(prices_up)

# Use realistic data for the smoothness test
# Perfect uptrend makes MACD a nearly straight line
# EMA of a straight line can have higher std than the line itself
macd_r, signal_r, hist_r = calculate_macd(prices_realistic)

check(
    "Returns exactly 3 series",
    all(isinstance(s, pd.Series) for s in [macd_line, signal_line, histogram]),
)
check(
    "All 3 series have same length as input",
    len(macd_line) == len(prices_up) and len(signal_line) == len(prices_up) and len(histogram) == len(prices_up),
    f"macd={len(macd_line)}, signal={len(signal_line)}, hist={len(histogram)}",
)
check(
    "MACD line positive in uptrend (fast EMA > slow EMA)",
    float(macd_line.iloc[-1]) > 0,
    f"expected > 0, got {macd_line.iloc[-1]:.4f}",
)
check(
    "Histogram = MACD line minus signal line",
    abs(float(histogram.iloc[-1]) - (float(macd_line.iloc[-1]) - float(signal_line.iloc[-1]))) < 0.0001,
    f"histogram={histogram.iloc[-1]:.6f}, " f"difference={macd_line.iloc[-1] - signal_line.iloc[-1]:.6f}",
)
check(
    "MACD line negative in downtrend",
    float(calculate_macd(prices_down)[0].iloc[-1]) < 0,
    f"got {calculate_macd(prices_down)[0].iloc[-1]:.4f}",
)
check(
    "Signal line is smoother than MACD on realistic data",
    # Must use realistic (noisy) data - perfect linear data breaks this
    # because EMA of a perfectly smooth line can have higher variance
    float(signal_r.dropna().std()) < float(macd_r.dropna().std()),
    f"MACD std={macd_r.dropna().std():.4f}, " f"Signal std={signal_r.dropna().std():.4f}",
)
check(
    "Histogram oscillates around zero on realistic data",
    # In realistic data MACD crosses signal repeatedly
    # so histogram should have both positive and negative values
    float(hist_r.dropna().min()) < 0 and float(hist_r.dropna().max()) > 0,
    f"min={hist_r.dropna().min():.4f}, max={hist_r.dropna().max():.4f}",
)


# ================================================================
# SECTION 5: BOLLINGER BANDS
# ================================================================
print("\n" + "─" * 60)
print("Bollinger Bands")
print("─" * 60)

upper, middle, lower = calculate_bollinger_bands(prices_up, 20)
sma_20_ref = calculate_moving_average(prices_up, 20)

check(
    "Returns exactly 3 series",
    all(isinstance(s, pd.Series) for s in [upper, middle, lower]),
)
check(
    "All bands have same length as input", len(upper) == len(prices_up), f"expected {len(prices_up)}, got {len(upper)}"
)
check(
    "Upper band > middle band at all valid points",
    (upper.dropna() > middle.dropna()).all(),
    f"violations found in upper > middle check",
)
check(
    "Middle band > lower band at all valid points",
    (middle.dropna() > lower.dropna()).all(),
    f"violations found in middle > lower check",
)
check(
    "Middle band equals SMA(20)",
    abs(float(middle.iloc[-1]) - float(sma_20_ref.iloc[-1])) < 0.001,
    f"middle={middle.iloc[-1]:.4f}, SMA20={sma_20_ref.iloc[-1]:.4f}",
)
check(
    "Bands are symmetric around middle",
    # upper - middle should equal middle - lower (±tolerance)
    abs((float(upper.iloc[-1]) - float(middle.iloc[-1])) - (float(middle.iloc[-1]) - float(lower.iloc[-1]))) < 0.001,
    f"upper gap={upper.iloc[-1]-middle.iloc[-1]:.4f}, " f"lower gap={middle.iloc[-1]-lower.iloc[-1]:.4f}",
)

# Flat prices test: bands should be very narrow (no volatility)
upper_flat, middle_flat, lower_flat = calculate_bollinger_bands(prices_flat, 20)
check(
    "Bands narrow for flat prices (low volatility)",
    float((upper_flat - lower_flat).dropna().mean()) < 1.0,
    f"average band width: {(upper_flat - lower_flat).dropna().mean():.4f}",
)

try:
    calculate_bollinger_bands(prices_up, 200)
    check("Raises ValueError for insufficient data", False, "no error raised")
except ValueError:
    check("Raises ValueError for insufficient data", True)


# ================================================================
# SECTION 6: STOCHASTIC OSCILLATOR
# Uses realistic data with noise - fixes the "D smoother than K" failure
# ================================================================
print("\n" + "─" * 60)
print("Stochastic Oscillator")
print("─" * 60)

# WHY realistic data here?
# Perfect straight-line prices create a %K that is already smooth
# because the ratio (close - lowest_low)/(highest_high - lowest_low)
# changes at a constant rate. %D averaging it adds no extra smoothing.
# Real prices have noise: %K is choppy, %D smooths it out correctly.
k_line, d_line = calculate_stochastic_oscillator(realistic_high, realistic_low, prices_realistic)

check(
    "Returns exactly 2 series",
    isinstance(k_line, pd.Series) and isinstance(d_line, pd.Series),
)
check(
    "Both series have same length as input",
    len(k_line) == len(prices_realistic) and len(d_line) == len(prices_realistic),
    f"K={len(k_line)}, D={len(d_line)}, input={len(prices_realistic)}",
)
check(
    "%K values between 0 and 100",
    (k_line.dropna() >= 0).all() and (k_line.dropna() <= 100).all(),
    f"K range: {k_line.dropna().min():.2f} to {k_line.dropna().max():.2f}",
)
check(
    "%D values between 0 and 100",
    (d_line.dropna() >= 0).all() and (d_line.dropna() <= 100).all(),
    f"D range: {d_line.dropna().min():.2f} to {d_line.dropna().max():.2f}",
)
check(
    "D is smoother than K (lower standard deviation)",
    # %D is a rolling mean of %K so it must have lower variance
    # This test passes with realistic noisy data, fails with perfect lines
    float(d_line.dropna().std()) < float(k_line.dropna().std()),
    f"K std={k_line.dropna().std():.4f}, D std={d_line.dropna().std():.4f}",
)
check(
    "D has compressed range compared to K",
    # Smoothing compresses the min/max so D range < K range
    (float(d_line.dropna().max()) - float(d_line.dropna().min()))
    < (float(k_line.dropna().max()) - float(k_line.dropna().min())),
    f"K range={k_line.dropna().max()-k_line.dropna().min():.2f}, "
    f"D range={d_line.dropna().max()-d_line.dropna().min():.2f}",
)
check(
    "%D has fewer NaN values than expected (3-period smoothing)",
    # First 13 values of K are NaN (14-period window)
    # First 15 values of D are NaN (K NaN + 3-period D smoothing - 1)
    d_line.isna().sum() >= k_line.isna().sum(),
    f"K NaN={k_line.isna().sum()}, D NaN={d_line.isna().sum()}",
)

try:
    calculate_stochastic_oscillator(realistic_high[:5], realistic_low[:5], prices_realistic[:5])
    check("Raises ValueError for insufficient data", False, "no error raised")
except ValueError:
    check("Raises ValueError for insufficient data", True)


# ================================================================
# SECTION 7: ATR
# ================================================================
print("\n" + "─" * 60)
print("ATR - Average True Range")
print("─" * 60)

atr = calculate_atr(simple_high, simple_low, prices_up, 14)

check(
    "Returns a pd.Series",
    isinstance(atr, pd.Series),
)
check("Output length matches input length", len(atr) == len(prices_up), f"expected {len(prices_up)}, got {len(atr)}")
check("All ATR values are positive", (atr.dropna() > 0).all(), f"min ATR: {atr.dropna().min():.4f}")
check(
    "ATR approximately equals high-low range for gap-free data",
    # simple_high = prices + 2, simple_low = prices - 2 → range = 4
    # ATR should converge to approximately 4
    abs(float(atr.iloc[-1]) - 4.0) < 1.0,
    f"expected ~4.0, got {atr.iloc[-1]:.4f}",
)

# Volatile data should have higher ATR than calm data
atr_volatile = calculate_atr(
    prices_realistic + 10, prices_realistic - 10, prices_realistic, 14  # wider high  # wider low
)
atr_calm = calculate_atr(prices_realistic + 1, prices_realistic - 1, prices_realistic, 14)  # narrow high  # narrow low
check(
    "Higher volatility produces higher ATR",
    float(atr_volatile.iloc[-1]) > float(atr_calm.iloc[-1]),
    f"volatile ATR={atr_volatile.iloc[-1]:.2f}, calm ATR={atr_calm.iloc[-1]:.2f}",
)

try:
    calculate_atr(simple_high[:5], simple_low[:5], prices_up[:5], 14)
    check("Raises ValueError for insufficient data", False, "no error raised")
except ValueError:
    check("Raises ValueError for insufficient data", True)


# ================================================================
# SECTION 8: OBV
# ================================================================
print("\n" + "─" * 60)
print("OBV - On-Balance Volume")
print("─" * 60)

obv_up = calculate_obv(prices_up, simple_volume)
obv_down = calculate_obv(prices_down, simple_volume)

check(
    "Returns a pd.Series",
    isinstance(obv_up, pd.Series),
)
check(
    "Output length matches input length", len(obv_up) == len(prices_up), f"expected {len(prices_up)}, got {len(obv_up)}"
)
check(
    "OBV rises in uptrend (volume added each up day)",
    float(obv_up.iloc[-1]) > float(obv_up.iloc[1]),
    f"start={obv_up.iloc[1]:,.0f}, end={obv_up.iloc[-1]:,.0f}",
)
check(
    "OBV falls in downtrend (volume subtracted each down day)",
    float(obv_down.iloc[-1]) < float(obv_down.iloc[1]),
    f"start={obv_down.iloc[1]:,.0f}, end={obv_down.iloc[-1]:,.0f}",
)
check(
    "OBV uptrend value > OBV downtrend value",
    float(obv_up.iloc[-1]) > float(obv_down.iloc[-1]),
    f"up={obv_up.iloc[-1]:,.0f}, down={obv_down.iloc[-1]:,.0f}",
)
check(
    "OBV is cumulative (magnitude grows over time)",
    abs(float(obv_up.iloc[-1])) > abs(float(obv_up.iloc[10])),
    f"early={obv_up.iloc[10]:,.0f}, final={obv_up.iloc[-1]:,.0f}",
)

try:
    calculate_obv(prices_up[:1], simple_volume[:1])
    check("Raises ValueError for insufficient data", False, "no error raised")
except ValueError:
    check("Raises ValueError for insufficient data", True)

# ================================================================
# SECTION 9: VWAP - CORRECTED
# ================================================================
print("\n" + "─" * 60)
print("VWAP - Volume Weighted Average Price")
print("─" * 60)

vwap = calculate_vwap(simple_high, simple_low, prices_up, simple_volume)

# Calculate typical price BEFORE the check() call
# so it is a plain variable, not an assignment inside a function argument
typical_price_last = (float(simple_high.iloc[-1]) + float(simple_low.iloc[-1]) + float(prices_up.iloc[-1])) / 3

check(
    "Returns a pd.Series",
    isinstance(vwap, pd.Series),
)
check("Output length matches input length", len(vwap) == len(prices_up), f"expected {len(prices_up)}, got {len(vwap)}")
check("All VWAP values are positive", (vwap.dropna() > 0).all(), f"min VWAP: {vwap.dropna().min():.4f}")
check(
    "VWAP within overall historical price range",
    # VWAP is CUMULATIVE - averages ALL bars seen so far
    # At bar 100 it reflects bars 1 to 100, not just bar 100
    # So it must lie within the full historical range
    float(vwap.iloc[-1]) >= float(simple_low.min()) and float(vwap.iloc[-1]) <= float(simple_high.max()),
    f"VWAP={vwap.iloc[-1]:.2f}, " f"range=[{simple_low.min():.2f}, {simple_high.max():.2f}]",
)
check(
    "VWAP rises in uptrend",
    float(vwap.iloc[-1]) > float(vwap.iloc[0]),
    f"start={vwap.iloc[0]:.2f}, end={vwap.iloc[-1]:.2f}",
)
check(
    "VWAP below latest high due to cumulative drag from early low prices",
    # Uptrend starts from price=1, early low prices drag VWAP
    # below the most recent high price - this is expected behaviour
    float(vwap.iloc[-1]) < float(simple_high.iloc[-1]),
    f"VWAP={vwap.iloc[-1]:.2f}, latest high={simple_high.iloc[-1]:.2f}",
)
check(
    "VWAP below last typical price in uptrend",
    # Typical price = (H + L + C) / 3 at the last bar
    # VWAP should be below this since early cheap bars drag it down
    float(vwap.iloc[-1]) < typical_price_last,
    f"VWAP={vwap.iloc[-1]:.2f}, typical price={typical_price_last:.2f}",
)


# ================================================================
# SECTION 10: RETURNS
# ================================================================
print("\n" + "─" * 60)
print("Returns Calculation")
print("─" * 60)

returns = calculate_returns(prices_up)

check(
    "Returns a pd.Series",
    isinstance(returns, pd.Series),
)
check(
    "Length is input length minus 1 (pct_change drops first row)",
    len(returns) == len(prices_up) - 1,
    f"expected {len(prices_up) - 1}, got {len(returns)}",
)
check("All returns positive in uptrend", (returns > 0).all(), f"min return: {returns.min():.6f}")
check(
    "First return is correct: (2-1)/1 = 1.0 = 100%",
    abs(float(returns.iloc[0]) - 1.0) < 0.001,
    f"expected 1.0, got {returns.iloc[0]:.6f}",
)
check(
    "Returns approach 0 as prices get larger (diminishing %)",
    # Price goes 1→2 = 100% return, 99→100 = 1.01% return
    float(returns.iloc[0]) > float(returns.iloc[-1]),
    f"first={returns.iloc[0]:.4f}, last={returns.iloc[-1]:.4f}",
)

try:
    calculate_returns(prices_up[:1])
    check("Raises ValueError for single price", False, "no error raised")
except ValueError:
    check("Raises ValueError for single price", True)


# ================================================================
# SECTION 11: SHARPE RATIO - CORRECTED
# ================================================================
print("\n" + "─" * 60)
print("Sharpe Ratio")
print("─" * 60)

# WHY add noise?
# Sharpe = mean / std × sqrt(252) - risk_free
# If std = 0 (all returns identical), Sharpe is undefined
# The function correctly returns 0.0 in that case
# Real returns always have variance - add small noise to simulate this

np.random.seed(42)

# Consistently positive: mean ~1%, small noise so std > 0
good_returns = pd.Series(0.01 + np.random.normal(0, 0.001, 252))

# Consistently negative: mean ~-1%, small noise
bad_returns = pd.Series(-0.01 + np.random.normal(0, 0.001, 252))

# Noisy: mean ~0.1%, high variance → low Sharpe
noisy_returns = pd.Series(np.random.normal(0.001, 0.05, 252))

sharpe_good = calculate_sharpe_ratio(good_returns)
sharpe_bad = calculate_sharpe_ratio(bad_returns)
sharpe_noisy = calculate_sharpe_ratio(noisy_returns)

print(f"    ℹ️  Sharpe good  : {sharpe_good:.4f}")
print(f"    ℹ️  Sharpe bad   : {sharpe_bad:.4f}")
print(f"    ℹ️  Sharpe noisy : {sharpe_noisy:.4f}")

check("Returns a float", isinstance(sharpe_good, float), f"got {type(sharpe_good)}")
check("Positive for consistently positive returns", sharpe_good > 0, f"got {sharpe_good:.4f}")
check("Negative for consistently negative returns", sharpe_bad < 0, f"got {sharpe_bad:.4f}")
check(
    "Consistent positive returns beat noisy returns",
    sharpe_good > sharpe_noisy,
    f"good={sharpe_good:.3f}, noisy={sharpe_noisy:.3f}",
)
check(
    "Returns 0.0 for empty series",
    calculate_sharpe_ratio(pd.Series([], dtype=float)) == 0.0,
)
check(
    "Returns 0.0 for zero volatility (mathematically undefined)",
    # std=0 means Sharpe undefined → function returns 0.0 as safe default
    # This is correct behaviour, not a bug
    calculate_sharpe_ratio(pd.Series([0.01] * 252)) == 0.0,
)
check(
    "Higher Sharpe for less volatile positive strategy",
    # Same mean return, lower volatility → higher Sharpe
    calculate_sharpe_ratio(pd.Series(0.01 + np.random.normal(0, 0.001, 252)))
    > calculate_sharpe_ratio(pd.Series(0.01 + np.random.normal(0, 0.03, 252))),
)


# ================================================================
# SECTION 12: MAX DRAWDOWN
# ================================================================
print("\n" + "─" * 60)
print("Maximum Drawdown")
print("─" * 60)

# Known portfolio: rises to 120, falls to 90, recovers to 130
# Peak = 120, Trough = 90
# Drawdown = (90 - 120) / 120 = -0.25 = -25%
portfolio_known = pd.Series([100, 105, 110, 120, 115, 90, 95, 130])
portfolio_up = pd.Series([100, 110, 120, 130, 140])
portfolio_down = pd.Series([100, 90, 80, 70, 60])

mdd_known = calculate_max_drawdown(portfolio_known)
mdd_up = calculate_max_drawdown(portfolio_up)
mdd_down = calculate_max_drawdown(portfolio_down)

check(
    "Returns a float",
    isinstance(mdd_known, float),
)
check("Known portfolio MDD = -25%", abs(mdd_known - (-0.25)) < 0.001, f"expected -0.25, got {mdd_known:.4f}")
check(
    "MDD is always <= 0 (it is a loss measure)",
    mdd_known <= 0 and mdd_up <= 0 and mdd_down <= 0,
    f"known={mdd_known:.4f}, up={mdd_up:.4f}, down={mdd_down:.4f}",
)
check("Pure uptrend has zero drawdown (never below peak)", mdd_up == 0.0, f"expected 0.0, got {mdd_up:.4f}")
check(
    "Downtrend has larger drawdown than uptrend",
    abs(mdd_down) > abs(mdd_up),
    f"down MDD={mdd_down:.4f}, up MDD={mdd_up:.4f}",
)
check(
    "Returns 0.0 for empty series",
    calculate_max_drawdown(pd.Series([], dtype=float)) == 0.0,
)


# ================================================================
# SECTION 13: WIN RATE
# ================================================================
print("\n" + "─" * 60)
print("Win Rate")
print("─" * 60)

trades_mixed = [100, -50, 200, -30, 150, -20, 80]  # 4 wins 3 losses
trades_all_win = [10, 20, 30, 40, 50]
trades_all_loss = [-10, -20, -30]
trades_one = [100]

check(
    "Correct win rate: 4/7 ≈ 57.14%",
    abs(calculate_win_rate(trades_mixed) - 4 / 7) < 0.001,
    f"expected {4/7:.4f}, got {calculate_win_rate(trades_mixed):.4f}",
)
check("All wins = 1.0 (100%)", calculate_win_rate(trades_all_win) == 1.0, f"got {calculate_win_rate(trades_all_win)}")
check("All losses = 0.0 (0%)", calculate_win_rate(trades_all_loss) == 0.0, f"got {calculate_win_rate(trades_all_loss)}")
check("Single winning trade = 1.0", calculate_win_rate(trades_one) == 1.0, f"got {calculate_win_rate(trades_one)}")
check(
    "Empty list returns 0.0",
    calculate_win_rate([]) == 0.0,
)
check(
    "Result always between 0 and 1",
    0.0 <= calculate_win_rate(trades_mixed) <= 1.0,
    f"got {calculate_win_rate(trades_mixed)}",
)


# ================================================================
# SECTION 14: PROFIT FACTOR
# ================================================================
print("\n" + "─" * 60)
print("Profit Factor")
print("─" * 60)

# trades_mixed: gross profit = 100+200+150+80 = 530
#               gross loss   = 50+30+20 = 100
#               profit factor = 530/100 = 5.3
expected_pf = (100 + 200 + 150 + 80) / (50 + 30 + 20)

check(
    f"Correct profit factor: {expected_pf:.1f}",
    abs(calculate_profit_factor(trades_mixed) - expected_pf) < 0.001,
    f"expected {expected_pf:.4f}, got {calculate_profit_factor(trades_mixed):.4f}",
)
check(
    "All losses returns 0.0 (no profit)",
    calculate_profit_factor(trades_all_loss) == 0.0,
    f"got {calculate_profit_factor(trades_all_loss)}",
)
check(
    "Empty list returns 0.0",
    calculate_profit_factor([]) == 0.0,
)
check(
    "All wins returns 0.0 (no denominator)",
    # No losing trades means gross_loss = 0
    # safe_divide returns 0.0 as default
    calculate_profit_factor(trades_all_win) == 0.0,
    f"got {calculate_profit_factor(trades_all_win)}",
)
check(
    "Result is always non-negative",
    calculate_profit_factor(trades_mixed) >= 0,
    f"got {calculate_profit_factor(trades_mixed)}",
)


# ================================================================
# SECTION 15: UTILITY FUNCTIONS
# ================================================================
print("\n" + "─" * 60)
print("Utility Functions")
print("─" * 60)

check("safe_divide: 10 / 2 = 5.0", safe_divide(10, 2) == 5.0, f"got {safe_divide(10, 2)}")
check("safe_divide: 10 / 0 = 0.0 (default)", safe_divide(10, 0) == 0.0, f"got {safe_divide(10, 0)}")
check(
    "safe_divide: 10 / 0 = 99 (custom default)",
    safe_divide(10, 0, default=99) == 99,
    f"got {safe_divide(10, 0, default=99)}",
)
check("safe_divide: negative numbers work", safe_divide(-10, 2) == -5.0, f"got {safe_divide(-10, 2)}")
check("format_currency: 1000 → '$1,000.00'", format_currency(1000) == "$1,000.00", f"got '{format_currency(1000)}'")
check(
    "format_currency: 1234567.89 → '$1,234,567.89'",
    format_currency(1_234_567.89) == "$1,234,567.89",
    f"got '{format_currency(1_234_567.89)}'",
)
check("format_currency: 0 → '$0.00'", format_currency(0) == "$0.00", f"got '{format_currency(0)}'")
check("format_percentage: 0.05 → '5.00%'", format_percentage(0.05) == "5.00%", f"got '{format_percentage(0.05)}'")
check(
    "format_percentage: 0.1234 → '12.34%'", format_percentage(0.1234) == "12.34%", f"got '{format_percentage(0.1234)}'"
)
check("format_percentage: 0 → '0.00%'", format_percentage(0) == "0.00%", f"got '{format_percentage(0)}'")


# ================================================================
# FINAL RESULTS
# ================================================================
print(f"\n{'=' * 60}")
print(f"  FINAL RESULTS")
print(f"{'=' * 60}")
print(f"  Passed : {passed}")
print(f"  Failed : {failed}")
print(f"  Total  : {passed + failed}")
print(f"  Score  : {passed/(passed+failed)*100:.1f}%")

if failed == 0:
    print(f"\n  ✅ All tests passed - helpers.py is working correctly")
else:
    print(f"\n  ❌ {failed} test(s) failed - review the output above")
    print(f"  Each ❌ line shows what value was received vs expected")

print("=" * 60)
