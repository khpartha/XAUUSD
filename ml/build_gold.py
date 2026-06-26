"""
Phase 2/3: Build features and target label from raw OHLC data.

Computes:
- Returns (1d, 5d, 10d)
- EMA20, EMA50 and distance from price
- RSI(14)
- MACD line + signal
- ATR(14)
- Candle shape (body size, wick ratios, range)
- Trader-style features (rolling-window swing high/low, support/resistance
  distance, Fibonacci retracement distance) - all computed using ONLY a
  trailing lookback window, never future data, to mirror how a trader
  actually reads a chart day by day.
- Target: 1 if tomorrow's close > today's close else 0
"""

import pandas as pd
import numpy as np

SWING_WINDOW = 126  # ~6 months of trading days, trailing window for swing high/low


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr


def build_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # --- Returns ---
    df["return_1d"] = c.pct_change(1)
    df["return_5d"] = c.pct_change(5)
    df["return_10d"] = c.pct_change(10)

    # --- Trend (EMA distance) ---
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    df["ema20_dist"] = (c - ema20) / c
    df["ema50_dist"] = (c - ema50) / c

    # --- Momentum ---
    df["rsi_14"] = compute_rsi(c, 14)
    macd_line, signal_line = compute_macd(c)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line

    # --- Volatility ---
    df["atr_14"] = compute_atr(h, l, c, 14)
    df["volatility_10d"] = df["return_1d"].rolling(10).std()

    # --- Candle shape ---
    candle_range = (h - l).replace(0, np.nan)
    df["body_size"] = (c - o).abs() / c
    df["upper_wick_ratio"] = (h - c.where(c > o, o)) / candle_range
    df["lower_wick_ratio"] = (c.where(c < o, o) - l) / candle_range
    df["high_low_range"] = (h - l) / c

    # --- Trader-style features: rolling swing high/low (trailing window only) ---
    swing_high = h.rolling(SWING_WINDOW, min_periods=SWING_WINDOW).max()
    swing_low = l.rolling(SWING_WINDOW, min_periods=SWING_WINDOW).min()
    swing_range = (swing_high - swing_low).replace(0, np.nan)

    df["dist_to_resistance"] = (swing_high - c) / c
    df["dist_to_support"] = (c - swing_low) / c

    fib_0382 = swing_high - 0.382 * swing_range
    fib_0500 = swing_high - 0.500 * swing_range
    fib_0618 = swing_high - 0.618 * swing_range

    df["dist_to_fib_0382"] = (c - fib_0382) / c
    df["dist_to_fib_0500"] = (c - fib_0500) / c
    df["dist_to_fib_0618"] = (c - fib_0618) / c

    # Breakout flag: did today's close make a fresh swing low vs yesterday's window?
    df["broke_support_today"] = (c < swing_low.shift(1)).astype(int)
    df["broke_resistance_today"] = (c > swing_high.shift(1)).astype(int)

    # --- Target: tomorrow up(1) or down(0) ---
    df["target"] = (c.shift(-1) > c).astype(int)

    # Drop rows where rolling windows haven't filled yet, or target is NaN (last row)
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    raw = pd.read_csv("training_data/gold_raw.csv")
    features = build_features(raw)
    features.to_csv("training_data/gold_features.csv", index=False)
    print(f"Built features: {len(features)} usable rows (from {len(raw)} raw rows)")
    print(f"Feature columns: {[c for c in features.columns if c not in ['date','open','high','low','close','target']]}")
    print(f"Target distribution: UP={features['target'].sum()} ({features['target'].mean():.1%}) | "
          f"DOWN={(1-features['target']).sum()} ({1-features['target'].mean():.1%})")