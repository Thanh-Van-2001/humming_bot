"""
Download & cache klines 1m tu Binance cho cac symbol dung de backtest PMM.
Luu vao alpha_factory/data/<SYMBOL>_1m.parquet
"""
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
           "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
           "USDCUSDT", "PAXGUSDT"]

INTERVAL = "1m"
# 3 thang gan nhat: 2 thang in-sample + 1 thang out-of-sample
START_DATE = "2026-02-16"
END_DATE = "2026-05-16"

BINANCE_URL = "https://api.binance.com/api/v3/klines"


def to_ms(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def download_symbol(symbol, interval, start_date, end_date):
    start_ts = to_ms(start_date)
    end_ts = to_ms(end_date)
    all_klines = []
    cur = start_ts
    print(f"  {symbol}: downloading", end="", flush=True)
    while cur < end_ts:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": cur, "endTime": end_ts, "limit": 1000}
        for attempt in range(5):
            try:
                r = requests.get(BINANCE_URL, params=params, timeout=15)
                if r.status_code == 429:
                    time.sleep(5)
                    continue
                data = r.json()
                break
            except Exception as e:
                print(f"[retry {e}]", end="")
                time.sleep(2)
        else:
            raise RuntimeError(f"failed {symbol}")
        if not data:
            break
        all_klines.extend(data)
        cur = data[-1][0] + 1
        print(".", end="", flush=True)
        time.sleep(0.15)
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "tb_base", "tb_quote", "ignore"]
    df = pd.DataFrame(all_klines, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df.set_index("open_time", inplace=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    print(f" {len(df)} bars")
    return df


def main():
    print(f"Downloading {len(SYMBOLS)} symbols {START_DATE} -> {END_DATE} ({INTERVAL})")
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{sym}_{INTERVAL}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            print(f"  {sym}: cached ({len(df)} bars) - skip")
            continue
        df = download_symbol(sym, INTERVAL, START_DATE, END_DATE)
        df.to_parquet(path)
    print("Done.")


if __name__ == "__main__":
    main()
