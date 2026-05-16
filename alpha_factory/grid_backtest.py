"""
Alpha Factory - Grid Trading (Grid Strike)
==========================================
Random-search backtester cho ho MM thu 3 cua Hummingbot:
controllers/generic/grid_strike.py (GridExecutor).

Khac PMM & Avellaneda: KHONG quote quanh mid theo spread, ma rai 1 LUOI gia
co dinh. Moi muc luoi khop -> ghep ngay 1 lenh chot lai cach 1 buoc luoi
(take_profit = grid_step). Luoi tu re-center khi gia troi xa.

Random ~7 knob: n_grid, grid_step, tp_steps, recenter_bars,
recenter_threshold, leverage, order spacing.

Usage: python grid_backtest.py --n-configs 4000
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
def backtest_grid(high, low, close, n_grid, grid_step, tp_steps,
                  recenter_bars, recenter_threshold, leverage,
                  target_pct, maker_fee, initial):
    """Grid trading: rai luoi 2 phia, moi fill ghep TP cach tp_steps buoc luoi."""
    n = len(close)
    base = (initial * target_pct) / close[0]
    cash = initial * (1.0 - target_pct)

    ns = 2 * n_grid                       # n_grid muc mua + n_grid muc ban
    st = np.zeros(ns, dtype=np.int64)     # 0 empty, 1 resting, 2 inpos
    ord_px = np.zeros(ns)
    amt = np.zeros(ns)
    exit_px = np.zeros(ns)

    notional = initial * leverage * 0.30 / n_grid
    center = close[0]
    bars_since = 0
    placed = False
    equity = np.empty(n)
    n_fills = 0
    volume = 0.0

    for i in range(n):
        mid = close[i]
        hi = high[i]
        lo = low[i]

        # --- A. khop lenh CHOT (exit) ---
        for s in range(ns):
            if st[s] != 2:
                continue
            is_long = s < n_grid
            if is_long and hi >= exit_px[s]:
                rev = exit_px[s] * amt[s]
                cash += rev - rev * maker_fee
                base -= amt[s]
                n_fills += 1
                volume += rev
                st[s] = 0
            elif (not is_long) and lo <= exit_px[s]:
                cost = exit_px[s] * amt[s]
                cash -= cost + cost * maker_fee
                base += amt[s]
                n_fills += 1
                volume += cost
                st[s] = 0

        # --- B. khop lenh VAO (entry) ---
        for s in range(ns):
            if st[s] != 1:
                continue
            is_long = s < n_grid
            if is_long and lo <= ord_px[s]:
                cost = ord_px[s] * amt[s]
                cash -= cost + cost * maker_fee
                base += amt[s]
                n_fills += 1
                volume += cost
                exit_px[s] = ord_px[s] * (1.0 + tp_steps * grid_step)
                st[s] = 2
            elif (not is_long) and hi >= ord_px[s]:
                rev = ord_px[s] * amt[s]
                cash += rev - rev * maker_fee
                base -= amt[s]
                n_fills += 1
                volume += rev
                exit_px[s] = ord_px[s] * (1.0 - tp_steps * grid_step)
                st[s] = 2

        equity[i] = cash + base * mid

        # --- C. re-center luoi & rai lenh vao ---
        bars_since += 1
        drift = abs(mid - center) / center
        if (not placed) or bars_since >= recenter_bars or drift > recenter_threshold:
            center = mid
            for s in range(ns):
                if st[s] == 0 or st[s] == 1:
                    if s < n_grid:
                        k = s
                        px = center * (1.0 - (k + 1) * grid_step)
                    else:
                        k = s - n_grid
                        px = center * (1.0 + (k + 1) * grid_step)
                    ord_px[s] = px
                    amt[s] = notional / px
                    st[s] = 1
            bars_since = 0
            placed = True

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
        n_grid=int(rng.choice([2, 3, 4, 5, 6, 8])),
        grid_step=float(np.exp(rng.uniform(np.log(0.0008), np.log(0.012)))),
        tp_steps=int(rng.choice([1, 1, 2])),
        recenter_bars=int(rng.choice([10, 30, 60, 120, 240, 480])),
        recenter_threshold=float(rng.uniform(0.01, 0.08)),
        leverage=int(rng.choice([1, 2, 3])),
        target_pct=0.5,
    )


def generate_configs(n, seed):
    rng = np.random.default_rng(seed)
    return [dict(alpha_id=f"GR{idx:04d}", strategy="grid", **sample_config(rng))
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
        equity, n_fills, volume = backtest_grid(
            h, l, c, cfg["n_grid"], cfg["grid_step"], cfg["tp_steps"],
            cfg["recenter_bars"], cfg["recenter_threshold"], cfg["leverage"],
            cfg["target_pct"], MAKER_FEE, INITIAL_CAPITAL)
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
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    configs = generate_configs(args.n_configs, args.seed)
    workers = args.workers or max(1, cpu_count() - 1)
    print(f"Grid Factory: {len(configs)} configs x {len(SYMBOLS)} symbols "
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
    lb.to_csv(os.path.join(RESULTS_DIR, "grid_full.csv"), index=False)
    robust.to_csv(os.path.join(RESULTS_DIR, "grid_robust.csv"), index=False)

    by_id = {r["alpha_id"]: r for r in results}
    with open(os.path.join(RESULTS_DIR, "grid_top_detail.json"), "w") as f:
        json.dump([by_id[a] for a in robust["alpha_id"].head(12)], f, indent=2, default=str)

    print(f"\n{'=' * 84}")
    print(f"  GRID: {len(lb)} alpha | {len(robust)} dat chuan ROBUST")
    print(f"{'=' * 84}")
    cols = ["alpha_id", "n_grid", "grid_step", "tp_steps", "recenter_bars",
            "recenter_threshold", "leverage", "is_sharpe_mean", "oos_sharpe_mean",
            "oos_pnl_pct_mean", "n_sym_profit_oos", "oos_max_dd_worst", "score"]
    show = robust if len(robust) else lb
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(show[cols].head(15).to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))
    print(f"{'=' * 84}")
    print("Saved: results/grid_full.csv | grid_robust.csv")


if __name__ == "__main__":
    main()
