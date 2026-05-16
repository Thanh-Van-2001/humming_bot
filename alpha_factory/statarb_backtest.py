"""
Alpha Factory - Statistical Arbitrage (Pairs Trading)
=====================================================
Random-search backtester cho ho ARBITRAGE cua Hummingbot:
controllers/generic/stat_arb.py (StatArb controller).

Giao dich SPREAD giua 2 tai san tuong quan (vd ETH vs BTC):
  - hedge ratio beta tu hoi quy truot
  - spread = log(A) - beta*log(B);  z-score cua spread
  - |z| > entry  -> mo vi the market-neutral (long ben re / short ben dat)
  - z hoi quy ve 0 -> chot;  |z| qua lon -> cat lo
=> alpha market-neutral that su (long 1 chan / short 1 chan, doc lap gia chung).

Random ~6 knob + chon cap: pair(A,B), lookback, entry, exit, stop_z, hold_limit.

Usage: python statarb_backtest.py --n-configs 4000
"""
import argparse
import itertools
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

# 9 symbol thanh khoan (bo USDCUSDT - stablecoin)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "PAXGUSDT"]
PAIRS = list(itertools.combinations(SYMBOLS, 2))   # 36 cap

TOTAL_QUOTE = 10_000.0
NOTIONAL = 3_000.0          # notional moi chan
TAKER_FEE = 0.0005          # chot bang lenh thi truong -> taker (bao thu)
OOS_FRACTION = 1.0 / 3.0
BARS_PER_YEAR = 525_600


@njit(cache=True, fastmath=True)
def backtest_statarb(logA, logB, priceA, priceB, lookback, entry, exit_thr,
                     stop_z, hold_limit, notional, taker_fee, total):
    """Pairs trading: z-score cua spread, vao/ra market-neutral 2 chan."""
    n = len(logA)
    equity = np.empty(n)
    realized = 0.0
    state = 0           # 0 flat, +1 long-spread (longA/shortB), -1 short-spread
    entryA = entryB = 0.0
    amtA = amtB = 0.0
    open_bar = 0
    n_trades = 0

    sx = sy = sxx = sxy = syy = 0.0
    ring_x = np.zeros(lookback)
    ring_y = np.zeros(lookback)
    filled = 0
    ptr = 0

    for i in range(n):
        x = logB[i]
        y = logA[i]
        if filled < lookback:
            ring_x[ptr] = x
            ring_y[ptr] = y
            sx += x; sy += y; sxx += x * x; sxy += x * y; syy += y * y
            filled += 1
        else:
            ox = ring_x[ptr]; oy = ring_y[ptr]
            sx += x - ox; sy += y - oy
            sxx += x * x - ox * ox
            sxy += x * y - ox * oy
            syy += y * y - oy * oy
            ring_x[ptr] = x; ring_y[ptr] = y
        ptr += 1
        if ptr >= lookback:
            ptr = 0

        if filled < lookback:
            equity[i] = total + realized
            continue

        nn = lookback
        denom = nn * sxx - sx * sx
        beta = (nn * sxy - sx * sy) / denom if denom != 0.0 else 1.0
        mean_sp = sy / nn - beta * sx / nn
        var_sp = (syy / nn - (sy / nn) ** 2) \
            + beta * beta * (sxx / nn - (sx / nn) ** 2) \
            - 2.0 * beta * (sxy / nn - sx * sy / (nn * nn))
        std_sp = np.sqrt(var_sp) if var_sp > 1e-12 else 1e-6
        spread_now = y - beta * x
        z = (spread_now - mean_sp) / std_sp

        pa = priceA[i]
        pb = priceB[i]

        # --- quan ly vi the ---
        if state != 0:
            held = i - open_bar
            do_exit = False
            if state == 1:    # long-spread mo khi z<=-entry, cho z hoi quy len
                if z >= -exit_thr or z <= -stop_z or held >= hold_limit:
                    do_exit = True
            else:             # short-spread mo khi z>=entry, cho z hoi quy xuong
                if z <= exit_thr or z >= stop_z or held >= hold_limit:
                    do_exit = True
            if do_exit:
                if state == 1:
                    pnl = amtA * (pa - entryA) + amtB * (entryB - pb)
                else:
                    pnl = amtA * (entryA - pa) + amtB * (pb - entryB)
                fee = (pa * amtA + pb * amtB) * taker_fee
                realized += pnl - fee
                state = 0
                n_trades += 1

        # --- vao vi the moi ---
        if state == 0:
            if z >= entry:                       # spread cao -> short A / long B
                state = -1
            elif z <= -entry:                    # spread thap -> long A / short B
                state = 1
            if state != 0:
                entryA = pa
                entryB = pb
                amtA = notional / pa
                amtB = notional / pb
                realized -= (pa * amtA + pb * amtB) * taker_fee   # phi vao
                open_bar = i

        # --- equity ---
        unreal = 0.0
        if state == 1:
            unreal = amtA * (pa - entryA) + amtB * (entryB - pb)
        elif state == -1:
            unreal = amtA * (entryA - pa) + amtB * (pb - entryB)
        equity[i] = total + realized + unreal

    return equity, n_trades


def segment_metrics(equity, initial):
    if len(equity) < 10:
        return dict(pnl_pct=0.0, sharpe=0.0, max_dd=0.0)
    pnl = equity[-1] - equity[0]
    pnl_pct = pnl / initial * 100.0
    ret = np.diff(equity) / initial
    ret = ret[np.isfinite(ret)]
    if len(ret) > 1 and ret.std() > 0:
        sharpe = ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR)
    else:
        sharpe = 0.0
    roll = np.maximum.accumulate(equity)
    max_dd = ((equity - roll) / initial * 100.0).min()
    return dict(pnl_pct=pnl_pct, sharpe=sharpe, max_dd=max_dd)


def sample_config(rng):
    pair_idx = int(rng.integers(0, len(PAIRS)))
    symA, symB = PAIRS[pair_idx]
    return dict(
        sym_a=symA, sym_b=symB,
        lookback=int(rng.choice([60, 120, 240, 360, 480])),
        entry=float(rng.uniform(1.5, 3.5)),
        exit_thr=float(rng.uniform(-0.5, 1.2)),
        stop_z=float(rng.uniform(4.0, 7.0)),
        hold_limit=int(rng.choice([120, 360, 720, 1440, 2880])),
    )


def generate_configs(n, seed):
    rng = np.random.default_rng(seed)
    return [dict(alpha_id=f"SA{idx:04d}", strategy="statarb", **sample_config(rng))
            for idx in range(n)]


_DATA = {}


def _init_worker():
    for sym in SYMBOLS:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{sym}_1m.parquet"))
        c = df["close"].to_numpy(np.float64)
        _DATA[sym] = (np.log(c), c)


def eval_config(cfg):
    logA, priceA = _DATA[cfg["sym_a"]]
    logB, priceB = _DATA[cfg["sym_b"]]
    n = min(len(priceA), len(priceB))
    split = int(n * (1.0 - OOS_FRACTION))
    equity, n_trades = backtest_statarb(
        logA[:n], logB[:n], priceA[:n], priceB[:n],
        cfg["lookback"], cfg["entry"], cfg["exit_thr"], cfg["stop_z"],
        cfg["hold_limit"], NOTIONAL, TAKER_FEE, TOTAL_QUOTE)
    is_m = segment_metrics(equity[:split], TOTAL_QUOTE)
    oos_m = segment_metrics(equity[split:], TOTAL_QUOTE)
    res = dict(cfg)
    res["is_sharpe"] = is_m["sharpe"]
    res["is_pnl_pct"] = is_m["pnl_pct"]
    res["is_max_dd"] = is_m["max_dd"]
    res["oos_sharpe"] = oos_m["sharpe"]
    res["oos_pnl_pct"] = oos_m["pnl_pct"]
    res["oos_max_dd"] = oos_m["max_dd"]
    res["n_trades"] = n_trades
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-configs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    configs = generate_configs(args.n_configs, args.seed)
    workers = args.workers or max(1, cpu_count() - 1)
    print(f"StatArb Factory: {len(configs)} configs ({len(PAIRS)} cap kha dung) "
          f"| {workers} workers")

    t0 = time.time()
    with Pool(workers, initializer=_init_worker) as pool:
        results = list(pool.imap_unordered(eval_config, configs, chunksize=16))
    print(f"Sweep done in {time.time() - t0:.0f}s")

    lb = pd.DataFrame(results)
    lb["score"] = (
        lb["is_sharpe"].clip(-5, 15) * 0.30
        + lb["oos_sharpe"].clip(-5, 15) * 0.45
        + lb["oos_pnl_pct"].clip(-50, 50) * 0.10
        + lb["oos_max_dd"].clip(-100, 0) * 0.08
    )
    lb = lb.sort_values("score", ascending=False).reset_index(drop=True)
    robust = lb[
        (lb["is_sharpe"] > 0.7) & (lb["oos_sharpe"] > 0.7)
        & (lb["is_pnl_pct"] > 0) & (lb["oos_pnl_pct"] > 0)
        & (lb["is_max_dd"] > -15) & (lb["oos_max_dd"] > -15)
        & (lb["n_trades"] > 15)
    ].reset_index(drop=True)
    lb.to_csv(os.path.join(RESULTS_DIR, "statarb_full.csv"), index=False)
    robust.to_csv(os.path.join(RESULTS_DIR, "statarb_robust.csv"), index=False)

    print(f"\n{'=' * 92}")
    print(f"  STAT-ARB: {len(lb)} alpha | {len(robust)} dat chuan ROBUST")
    print(f"{'=' * 92}")
    cols = ["alpha_id", "sym_a", "sym_b", "lookback", "entry", "exit_thr",
            "stop_z", "is_sharpe", "oos_sharpe", "oos_pnl_pct", "n_trades", "score"]
    show = robust if len(robust) else lb
    with pd.option_context("display.width", 230, "display.max_columns", 20):
        print(show[cols].head(15).to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))
    print(f"{'=' * 92}")
    print("Saved: results/statarb_full.csv | statarb_robust.csv")


if __name__ == "__main__":
    main()
