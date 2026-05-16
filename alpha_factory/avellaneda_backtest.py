"""
Alpha Factory - Avellaneda-Stoikov Market Making
================================================
Random-search backtester cho ho chien luoc HFT MM thu 2 cua Hummingbot:
avellaneda_market_making (mo hinh Avellaneda-Stoikov).

Khac PMM o cho: KHONG dung spread co dinh ma:
  - reservation price = mid - q*gamma*sigma  (lech quote theo ton kho q)
  - half_spread       = base_spread + vol_mult*sigma  (spread co gian theo bien dong)
=> tu can bang ton kho + tu noi rong spread khi thi truong dong.

Random ~8 knob: risk_factor(gamma), base_spread, vol_mult, vol_window,
order_notional, refresh, inventory_target, max_inventory.

Usage: python avellaneda_backtest.py --n-configs 4000
"""
import argparse
import json
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "USDCUSDT", "PAXGUSDT"]

INITIAL_CAPITAL = 10_000.0
MAKER_FEE = 0.0002
OOS_FRACTION = 1.0 / 3.0
BARS_PER_YEAR = 525_600


@njit(cache=True, fastmath=True)
def backtest_avellaneda(high, low, close, gamma, base_spread, vol_mult,
                        vol_window, order_notional, refresh_bars,
                        target_pct, max_inv_mult, maker_fee, initial):
    """Avellaneda-Stoikov MM: quote 1 level, reservation price + spread co gian theo vol."""
    n = len(close)
    base = (initial * target_pct) / close[0]
    cash = initial * (1.0 - target_pct)

    equity = np.empty(n)
    n_fills = 0
    volume = 0.0

    ring = np.zeros(vol_window)
    rsum = 0.0
    rsumsq = 0.0
    filled = 0
    ptr = 0
    prev_close = close[0]

    bid_px = 0.0
    bid_amt = 0.0
    ask_px = 0.0
    ask_amt = 0.0
    has_orders = False
    bars_since = 0

    for i in range(n):
        mid = close[i]
        hi = high[i]
        lo = low[i]

        # --- cap nhat bien dong sigma (rolling std cua return, ring buffer) ---
        r = (mid - prev_close) / prev_close
        prev_close = mid
        if filled < vol_window:
            ring[ptr] = r
            rsum += r
            rsumsq += r * r
            filled += 1
        else:
            old = ring[ptr]
            rsum += r - old
            rsumsq += r * r - old * old
            ring[ptr] = r
        ptr += 1
        if ptr >= vol_window:
            ptr = 0
        if filled > 1:
            var = rsumsq / filled - (rsum / filled) ** 2
            sigma = np.sqrt(var) if var > 0.0 else 0.0
        else:
            sigma = base_spread

        # --- khop lenh dat tu bar truoc ---
        if has_orders:
            if bid_amt > 0.0 and lo <= bid_px:
                cost = bid_px * bid_amt
                if cash >= cost:
                    cash -= cost + cost * maker_fee
                    base += bid_amt
                    n_fills += 1
                    volume += cost
                    bid_amt = 0.0
            if ask_amt > 0.0 and hi >= ask_px:
                if base >= ask_amt:
                    rev = ask_px * ask_amt
                    cash += rev - rev * maker_fee
                    base -= ask_amt
                    n_fills += 1
                    volume += rev
                    ask_amt = 0.0

        equity[i] = cash + base * mid

        # --- refresh quote (Avellaneda) ---
        bars_since += 1
        if (not has_orders) or bars_since >= refresh_bars:
            base_val = base * mid
            total = base_val + cash
            q = (base_val - total * target_pct) / order_notional   # ton kho chuan hoa
            half = base_spread + vol_mult * sigma
            center = mid * (1.0 - q * gamma * sigma)               # reservation price
            bid_px = center * (1.0 - half)
            ask_px = center * (1.0 + half)
            bid_amt = order_notional / bid_px
            ask_amt = order_notional / ask_px
            if q > max_inv_mult:        # ton kho qua nhieu -> ngung mua
                bid_amt = 0.0
            if q < -max_inv_mult:       # ton kho qua it -> ngung ban
                ask_amt = 0.0
            has_orders = True
            bars_since = 0

    return equity, n_fills, volume


def excess_metrics(seg, initial):
    if len(seg) < 10:
        return dict(pnl_pct=0.0, sharpe=0.0, max_dd=0.0)
    pnl = seg[-1] - seg[0]
    pnl_pct = pnl / initial * 100.0
    d = np.diff(seg) / initial
    d = d[np.isfinite(d)]
    if len(d) > 1 and d.std() > 0:
        sharpe = d.mean() / d.std() * np.sqrt(BARS_PER_YEAR)
    else:
        sharpe = 0.0
    roll = np.maximum.accumulate(seg)
    max_dd = ((seg - roll) / initial * 100.0).min()
    return dict(pnl_pct=pnl_pct, sharpe=sharpe, max_dd=max_dd)


def sample_config(rng):
    return dict(
        gamma=float(np.exp(rng.uniform(np.log(0.5), np.log(60.0)))),
        base_spread=float(np.exp(rng.uniform(np.log(0.0002), np.log(0.006)))),
        vol_mult=float(rng.uniform(0.0, 250.0)),
        vol_window=int(rng.choice([30, 60, 120, 240, 480])),
        order_notional=float(rng.choice([50, 75, 100, 150, 200, 300])),
        refresh_bars=int(rng.choice([1, 2, 3, 5, 10])),
        target_pct=float(rng.choice([0.4, 0.5, 0.6])),
        max_inv_mult=float(rng.uniform(2.0, 12.0)),
    )


def generate_configs(n, seed):
    rng = np.random.default_rng(seed)
    return [dict(alpha_id=f"AV{idx:04d}", strategy="avellaneda", **sample_config(rng))
            for idx in range(n)]


_DATA = {}


def _init_worker():
    for sym in SYMBOLS:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{sym}_1m.parquet"))
        h = df["high"].to_numpy(np.float64)
        l = df["low"].to_numpy(np.float64)
        c = df["close"].to_numpy(np.float64)
        split = int(len(c) * (1.0 - OOS_FRACTION))
        _DATA[sym] = (h, l, c, split)


def eval_config(cfg):
    per_sym = []
    for sym in SYMBOLS:
        h, l, c, split = _DATA[sym]
        equity, n_fills, volume = backtest_avellaneda(
            h, l, c, cfg["gamma"], cfg["base_spread"], cfg["vol_mult"],
            cfg["vol_window"], cfg["order_notional"], cfg["refresh_bars"],
            cfg["target_pct"], cfg["max_inv_mult"], MAKER_FEE, INITIAL_CAPITAL)
        base0 = (INITIAL_CAPITAL * cfg["target_pct"]) / c[0]
        cash0 = INITIAL_CAPITAL * (1.0 - cfg["target_pct"])
        excess = equity - (cash0 + base0 * c)
        is_m = excess_metrics(excess[:split], INITIAL_CAPITAL)
        oos_m = excess_metrics(excess[split:], INITIAL_CAPITAL)
        per_sym.append(dict(symbol=sym, n_fills=n_fills,
                            is_pnl_pct=is_m["pnl_pct"], is_sharpe=is_m["sharpe"],
                            is_max_dd=is_m["max_dd"],
                            oos_pnl_pct=oos_m["pnl_pct"], oos_sharpe=oos_m["sharpe"],
                            oos_max_dd=oos_m["max_dd"]))
    df = pd.DataFrame(per_sym)
    res = dict(cfg)
    res["is_sharpe_mean"] = df["is_sharpe"].mean()
    res["is_pnl_pct_mean"] = df["is_pnl_pct"].mean()
    res["is_max_dd_worst"] = df["is_max_dd"].min()
    res["oos_sharpe_mean"] = df["oos_sharpe"].mean()
    res["oos_pnl_pct_mean"] = df["oos_pnl_pct"].mean()
    res["oos_max_dd_worst"] = df["oos_max_dd"].min()
    res["n_sym_profit_oos"] = int((df["oos_pnl_pct"] > 0).sum())
    res["avg_fills"] = df["n_fills"].mean()
    res["per_symbol"] = df.to_dict("records")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-configs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    configs = generate_configs(args.n_configs, args.seed)
    workers = args.workers or max(1, cpu_count() - 1)
    print(f"Avellaneda Factory: {len(configs)} configs x {len(SYMBOLS)} symbols "
          f"= {len(configs) * len(SYMBOLS)} backtests | {workers} workers")

    t0 = time.time()
    with Pool(workers, initializer=_init_worker) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(eval_config, configs, chunksize=8)):
            results.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(configs)} ({time.time() - t0:.0f}s)")
    print(f"Sweep done in {time.time() - t0:.0f}s")

    rows = [{k: v for k, v in r.items() if k != "per_symbol"} for r in results]
    lb = pd.DataFrame(rows)
    lb["score"] = (
        lb["is_sharpe_mean"].clip(-5, 15) * 0.30
        + lb["oos_sharpe_mean"].clip(-5, 15) * 0.45
        + lb["oos_pnl_pct_mean"].clip(-50, 50) * 0.08
        + lb["n_sym_profit_oos"] * 0.8
        + lb["oos_max_dd_worst"].clip(-100, 0) * 0.06
    )
    lb = lb.sort_values("score", ascending=False).reset_index(drop=True)
    robust = lb[
        (lb["is_sharpe_mean"] > 0.5) & (lb["oos_sharpe_mean"] > 0.5)
        & (lb["oos_pnl_pct_mean"] > 0) & (lb["is_pnl_pct_mean"] > 0)
        & (lb["n_sym_profit_oos"] >= 7)
        & (lb["is_max_dd_worst"] > -20) & (lb["oos_max_dd_worst"] > -20)
        & (lb["avg_fills"] > 20)
    ].reset_index(drop=True)
    lb.to_csv(os.path.join(RESULTS_DIR, "avellaneda_full.csv"), index=False)
    robust.to_csv(os.path.join(RESULTS_DIR, "avellaneda_robust.csv"), index=False)

    by_id = {r["alpha_id"]: r for r in results}
    with open(os.path.join(RESULTS_DIR, "avellaneda_top_detail.json"), "w") as f:
        json.dump([by_id[a] for a in robust["alpha_id"].head(12)], f, indent=2, default=str)

    print(f"\n{'=' * 88}")
    print(f"  AVELLANEDA: {len(lb)} alpha | {len(robust)} dat chuan ROBUST")
    print(f"{'=' * 88}")
    cols = ["alpha_id", "gamma", "base_spread", "vol_mult", "vol_window",
            "refresh_bars", "is_sharpe_mean", "oos_sharpe_mean",
            "oos_pnl_pct_mean", "n_sym_profit_oos", "oos_max_dd_worst", "score"]
    show = robust if len(robust) else lb
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(show[cols].head(15).to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))
    print(f"{'=' * 88}")
    print("Saved: results/avellaneda_full.csv | avellaneda_robust.csv")


if __name__ == "__main__":
    main()
