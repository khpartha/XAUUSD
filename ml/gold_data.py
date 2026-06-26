"""
Phase 1: Download raw XAU/USD proxy data (GC=F gold futures via yfinance)
Run this in Google Colab where internet access is unrestricted.
"""

import yfinance as yf
import pandas as pd

# === CONFIG ===
TICKER = "GC=F"          # COMEX Gold Futures - free proxy for XAU/USD
PERIOD = "5y"             # 5 years of history (override below for testing)
INTERVAL = "1d"           # daily candles
OUTPUT_PATH = "training_data/gold_raw.csv"

def download_gold_data(ticker=TICKER, period=PERIOD, interval=INTERVAL):
    print(f"Downloading {ticker} | period={period} | interval={interval} ...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/period/internet connection.")

    # yfinance sometimes returns MultiIndex columns - flatten if so
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume"
    })

    # Keep only what we need - OHLC, drop Volume (unreliable for futures proxy)
    df = df[["date", "open", "high", "low", "close"]]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    print(f"Downloaded {len(df)} rows | {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("training_data", exist_ok=True)
    raw = download_gold_data()
    raw.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    print(raw.head())
    print("...")
    print(raw.tail())