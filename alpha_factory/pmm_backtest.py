"""
PMM Alpha Factory v2
====================
Random-search backtester mo phong dung Hummingbot V2 Market Making controller
(pmm_simple / pmm_dynamic): moi lenh maker khop -> 1 PositionExecutor quan ly
bang TRIPLE BARRIER (stop_loss / take_profit / time_limit / trailing_stop),
co cooldown, leverage, multi-level buy/sell spreads & amount distribution.

Sinh ngau nhien N config tren khong gian ~15 tham so (tat ca cac knob that cua
Hummingbot MarketMakingControllerConfigBase), backtest tren 10 symbol, chia
in-sample / out-of-sample, rank theo do on dinh -> xuat leaderboard.

Usage:
    python pmm_backtest.py --n-configs 4000 --seed 42
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

TOTAL_QUOTE = 10_000.0     # total_amount_quote (von)
MAKER_FEE = 0.0002         # 0.02% maker Binance Futures
TAKER_FEE = 0.0005         # 0.05% taker Binance Futures
OOS_FRACTION = 1.0 / 3.0   # 1/3 cuoi du lieu = out-of-sample
BARS_PER_YEAR = 525_600    # 1m bars

# slot state
S_EMPTY, S_RESTING, S_INPOS, S_COOLDOWN = 0, 1, 2, 3

# ----------------------------------------------------------------------
# BACKTEST ENGINE (numba) -- mo phong PMM V2 + triple barrier
# ----------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def backtest_pmm_v2(high, low, close,
                    n_levels, buy_spread_base, sell_spread_base, spread_step,
                    amount_base_w, refresh_bars, cooldown_bars,
                    use_sl, stop_loss, use_tp, take_profit, tp_is_market,
                    time_limit_bars, use_trail, trail_act, trail_delta,
                    leverage, total_quote, maker_fee, taker_fee):
    n = len(close)
    n_slots = 2 * n_levels  # 0..n_levels-1 = long (buy), rest = short (sell)

    state = np.zeros(n_slots, dtype=np.int64)
    ord_px = np.zeros(n_slots)
    ord_amt = np.zeros(n_slots)
    entry_px = np.zeros(n_slots)
    pos_amt = np.zeros(n_slots)
    open_bar = np.zeros(n_slots, dtype=np.int64)
    trail_ext = np.zeros(n_slots)
    cd_until = np.zeros(n_slots, dtype=np.int64)

    # phan bo notional theo level: w[l] = amount_base_w ** l
    side_budget = total_quote * leverage * 0.30
    wsum = 0.0
    for l in range(n_levels):
        wsum += amount_base_w ** l
    notional = np.zeros(n_levels)
    for l in range(n_levels):
        notional[l] = side_budget * (amount_base_w ** l) / wsum

    realized = 0.0
    equity = np.empty(n)
    n_fills = 0
    n_closes = 0
    volume = 0.0
    bars_since = 0
    has_placed = False
    sum_abs_inv = 0.0  # de tinh do lech ton kho trung binh

    for i in range(n):
        mid = close[i]
        hi = high[i]
        lo = low[i]

        # --- A. quan ly position dang mo: TRIPLE BARRIER ---
        for s in range(n_slots):
            if state[s] != S_INPOS:
                continue
            is_long = s < n_levels
            ent = entry_px[s]
            amt = pos_amt[s]
            closed = False
            close_px = 0.0
            exit_taker = True

            if is_long:
                if use_trail and hi > trail_ext[s]:
                    trail_ext[s] = hi
                # SL truoc (bao thu)
                if use_sl and lo <= ent * (1.0 - stop_loss):
                    close_px = ent * (1.0 - stop_loss)
                    closed = True
                elif use_tp and hi >= ent * (1.0 + take_profit):
                    close_px = ent * (1.0 + take_profit)
                    exit_taker = tp_is_market
                    closed = True
                elif use_trail and trail_ext[s] >= ent * (1.0 + trail_act) \
                        and lo <= trail_ext[s] * (1.0 - trail_delta):
                    close_px = trail_ext[s] * (1.0 - trail_delta)
                    closed = True
                elif (i - open_bar[s]) >= time_limit_bars:
                    close_px = mid
                    closed = True
                if closed:
                    gross = amt * (close_px - ent)
                    fee = close_px * amt * (taker_fee if exit_taker else maker_fee)
                    realized += gross - fee
            else:
                if use_trail and (trail_ext[s] == 0.0 or lo < trail_ext[s]):
                    trail_ext[s] = lo
                if use_sl and hi >= ent * (1.0 + stop_loss):
                    close_px = ent * (1.0 + stop_loss)
                    closed = True
                elif use_tp and lo <= ent * (1.0 - take_profit):
                    close_px = ent * (1.0 - take_profit)
                    exit_taker = tp_is_market
                    closed = True
                elif use_trail and trail_ext[s] != 0.0 \
                        and trail_ext[s] <= ent * (1.0 - trail_act) \
                        and hi >= trail_ext[s] * (1.0 + trail_delta):
                    close_px = trail_ext[s] * (1.0 + trail_delta)
                    closed = True
                elif (i - open_bar[s]) >= time_limit_bars:
                    close_px = mid
                    closed = True
                if closed:
                    gross = amt * (ent - close_px)
                    fee = close_px * amt * (taker_fee if exit_taker else maker_fee)
                    realized += gross - fee

            if closed:
                n_closes += 1
                volume += close_px * amt
                if cooldown_bars > 0:
                    state[s] = S_COOLDOWN
                    cd_until[s] = i + cooldown_bars
                else:
                    state[s] = S_EMPTY

        # --- B. khop lenh maker dang RESTING (dat tu bar truoc) ---
        for s in range(n_slots):
            if state[s] != S_RESTING:
                continue
            is_long = s < n_levels
            if is_long and lo <= ord_px[s]:
                entry_px[s] = ord_px[s]
                pos_amt[s] = ord_amt[s]
                open_bar[s] = i
                trail_ext[s] = ord_px[s]
                realized -= ord_px[s] * ord_amt[s] * maker_fee
                volume += ord_px[s] * ord_amt[s]
                state[s] = S_INPOS
                n_fills += 1
            elif (not is_long) and hi >= ord_px[s]:
                entry_px[s] = ord_px[s]
                pos_amt[s] = ord_amt[s]
                open_bar[s] = i
                trail_ext[s] = 0.0
                realized -= ord_px[s] * ord_amt[s] * maker_fee
                volume += ord_px[s] * ord_amt[s]
                state[s] = S_INPOS
                n_fills += 1

        # --- C. het cooldown ---
        for s in range(n_slots):
            if state[s] == S_COOLDOWN and i >= cd_until[s]:
                state[s] = S_EMPTY

        # --- D. refresh: dat/thay lenh maker o slot EMPTY hoac RESTING ---
        bars_since += 1
        if (not has_placed) or bars_since >= refresh_bars:
            for s in range(n_slots):
                if state[s] == S_EMPTY or state[s] == S_RESTING:
                    if s < n_levels:
                        l = s
                        spr = buy_spread_base + spread_step * l
                        px = mid * (1.0 - spr)
                    else:
                        l = s - n_levels
                        spr = sell_spread_base + spread_step * l
                        px = mid * (1.0 + spr)
                    ord_px[s] = px
                    ord_amt[s] = notional[l] / px
                    state[s] = S_RESTING
            bars_since = 0
            has_placed = True

        # --- E. equity = von + realized + unrealized ---
        unreal = 0.0
        inv = 0.0
        for s in range(n_slots):
            if state[s] == S_INPOS:
                if s < n_levels:
                    unreal += pos_amt[s] * (mid - entry_px[s])
                    inv += pos_amt[s] * mid
                else:
                    unreal += pos_amt[s] * (entry_px[s] - mid)
                    inv -= pos_amt[s] * mid
        equity[i] = total_quote + realized + unreal
        sum_abs_inv += abs(inv)

    avg_inv_pct = (sum_abs_inv / n) / total_quote * 100.0
    return equity, n_fills, n_closes, volume, avg_inv_pct


def segment_metrics(equity, initial):
    """Metrics tren 1 doan equity curve."""
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
    roll_max = np.maximum.accumulate(equity)
    dd = (equity - roll_max) / initial * 100.0
    return dict(pnl_pct=pnl_pct, sharpe=sharpe, max_dd=dd.min())


# ----------------------------------------------------------------------
# RANDOM CONFIG GENERATOR -- random ~15 knob that cua Hummingbot PMM V2
# ----------------------------------------------------------------------


def sample_config(rng):
    """Sinh 1 config PMM V2 ngau nhien tren toan bo khong gian tham so."""
    n_levels = int(rng.choice([1, 2, 3, 4, 5]))
    refresh_s = int(rng.choice([30, 60, 120, 180, 300, 600, 900]))
    cooldown_s = int(rng.choice([0, 15, 30, 60, 120, 300]))
    time_limit_s = int(rng.choice([300, 600, 900, 1800, 2700, 3600, 7200]))

    use_sl = bool(rng.random() < 0.7)
    use_tp = bool(rng.random() < 0.7)
    use_trail = bool(rng.random() < 0.45)

    return dict(
        n_levels=n_levels,
        # buy/sell spreads (level-0) + buoc giua cac level  -> buy_spreads/sell_spreads
        buy_spread_base=float(np.exp(rng.uniform(np.log(0.0003), np.log(0.015)))),
        sell_spread_base=float(np.exp(rng.uniform(np.log(0.0003), np.log(0.015)))),
        spread_step=float(np.exp(rng.uniform(np.log(0.0002), np.log(0.004)))),
        # phan bo amount qua cac level  -> buy_amounts_pct/sell_amounts_pct
        amount_base_w=float(rng.uniform(0.5, 2.0)),
        # executor_refresh_time / cooldown_time
        refresh_s=refresh_s,
        cooldown_s=cooldown_s,
        # triple barrier
        use_sl=use_sl,
        stop_loss=float(np.exp(rng.uniform(np.log(0.004), np.log(0.06)))) if use_sl else 0.0,
        use_tp=use_tp,
        take_profit=float(np.exp(rng.uniform(np.log(0.003), np.log(0.04)))) if use_tp else 0.0,
        tp_is_market=bool(rng.random() < 0.3),   # take_profit_order_type LIMIT/MARKET
        time_limit_s=time_limit_s,
        use_trail=use_trail,
        trail_act=float(rng.uniform(0.004, 0.03)) if use_trail else 0.0,
        trail_delta=float(rng.uniform(0.001, 0.012)) if use_trail else 0.0,
        # leverage
        leverage=int(rng.choice([1, 2, 3, 5])),
    )


def generate_configs(n, seed):
    rng = np.random.default_rng(seed)
    return [dict(alpha_id=f"A{idx:04d}", **sample_config(rng)) for idx in range(n)]


# ----------------------------------------------------------------------
# WORKER
# ----------------------------------------------------------------------

_DATA = {}  # symbol -> (high, low, close, split_idx)


def _init_worker():
    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{sym}_1m.parquet")
        df = pd.read_parquet(path)
        h = df["high"].to_numpy(np.float64)
        l = df["low"].to_numpy(np.float64)
        c = df["close"].to_numpy(np.float64)
        split = int(len(c) * (1.0 - OOS_FRACTION))
        _DATA[sym] = (h, l, c, split)


def eval_config(cfg):
    """Backtest 1 config tren toan bo symbol -> aggregate metrics."""
    refresh_bars = max(1, cfg["refresh_s"] // 60)
    cooldown_bars = cfg["cooldown_s"] // 60
    time_limit_bars = max(1, cfg["time_limit_s"] // 60)
    per_sym = []
    for sym in SYMBOLS:
        h, l, c, split = _DATA[sym]
        equity, n_fills, n_closes, volume, avg_inv = backtest_pmm_v2(
            h, l, c,
            cfg["n_levels"], cfg["buy_spread_base"], cfg["sell_spread_base"],
            cfg["spread_step"], cfg["amount_base_w"], refresh_bars, cooldown_bars,
            cfg["use_sl"], cfg["stop_loss"], cfg["use_tp"], cfg["take_profit"],
            cfg["tp_is_market"], time_limit_bars, cfg["use_trail"],
            cfg["trail_act"], cfg["trail_delta"], cfg["leverage"],
            TOTAL_QUOTE, MAKER_FEE, TAKER_FEE)
        is_m = segment_metrics(equity[:split], TOTAL_QUOTE)
        oos_m = segment_metrics(equity[split:], TOTAL_QUOTE)
        per_sym.append(dict(symbol=sym, n_fills=n_fills, n_closes=n_closes,
                            volume=volume, avg_inv_pct=avg_inv,
                            is_pnl_pct=is_m["pnl_pct"], is_sharpe=is_m["sharpe"],
                            is_max_dd=is_m["max_dd"],
                            oos_pnl_pct=oos_m["pnl_pct"], oos_sharpe=oos_m["sharpe"],
                            oos_max_dd=oos_m["max_dd"]))
    df = pd.DataFrame(per_sym)
    res = dict(cfg)
    res["is_sharpe_mean"] = df["is_sharpe"].mean()
    res["is_pnl_pct_mean"] = df["is_pnl_pct"].mean()
    res["is_pnl_pct_worst"] = df["is_pnl_pct"].min()
    res["is_max_dd_worst"] = df["is_max_dd"].min()
    res["oos_sharpe_mean"] = df["oos_sharpe"].mean()
    res["oos_pnl_pct_mean"] = df["oos_pnl_pct"].mean()
    res["oos_pnl_pct_worst"] = df["oos_pnl_pct"].min()
    res["oos_max_dd_worst"] = df["oos_max_dd"].min()
    res["n_sym_profit_is"] = int((df["is_pnl_pct"] > 0).sum())
    res["n_sym_profit_oos"] = int((df["oos_pnl_pct"] > 0).sum())
    res["avg_fills"] = df["n_fills"].mean()
    res["avg_inv_pct"] = df["avg_inv_pct"].mean()
    res["per_symbol"] = df.to_dict("records")
    return res


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-configs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    configs = generate_configs(args.n_configs, args.seed)
    workers = args.workers or max(1, cpu_count() - 1)
    print(f"Alpha Factory v2: {len(configs)} configs x {len(SYMBOLS)} symbols "
          f"= {len(configs) * len(SYMBOLS)} backtests | {workers} workers")

    t0 = time.time()
    with Pool(workers, initializer=_init_worker) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(eval_config, configs, chunksize=8)):
            results.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(configs)} done ({time.time() - t0:.0f}s)")
    print(f"Sweep done in {time.time() - t0:.0f}s")

    rows = [{k: v for k, v in r.items() if k != "per_symbol"} for r in results]
    lb = pd.DataFrame(rows)

    # composite score: thuong on dinh IS+OOS, profit nhieu symbol, phat drawdown
    lb["score"] = (
        lb["is_sharpe_mean"].clip(-5, 15) * 0.25
        + lb["oos_sharpe_mean"].clip(-5, 15) * 0.45
        + lb["oos_pnl_pct_mean"].clip(-50, 50) * 0.08
        + lb["n_sym_profit_oos"] * 0.8
        + lb["oos_max_dd_worst"].clip(-100, 0) * 0.06
    )
    lb = lb.sort_values("score", ascending=False).reset_index(drop=True)

    # robustness filter
    robust = lb[
        (lb["is_sharpe_mean"] > 0.5)
        & (lb["oos_sharpe_mean"] > 0.5)
        & (lb["oos_pnl_pct_mean"] > 0)
        & (lb["is_pnl_pct_mean"] > 0)
        & (lb["n_sym_profit_oos"] >= 7)
        & (lb["is_max_dd_worst"] > -20)
        & (lb["oos_max_dd_worst"] > -20)
        & (lb["avg_fills"] > 20)
    ].reset_index(drop=True)

    lb.to_csv(os.path.join(RESULTS_DIR, "leaderboard_full.csv"), index=False)
    robust.to_csv(os.path.join(RESULTS_DIR, "leaderboard_robust.csv"), index=False)

    by_id = {r["alpha_id"]: r for r in results}
    top_detail = [by_id[aid] for aid in robust["alpha_id"].head(12)]
    with open(os.path.join(RESULTS_DIR, "top_detail.json"), "w") as f:
        json.dump(top_detail, f, indent=2, default=str)

    print(f"\n{'=' * 92}")
    print(f"  KET QUA: {len(lb)} alpha | {len(robust)} dat chuan ROBUST")
    print(f"{'=' * 92}")
    cols = ["alpha_id", "n_levels", "buy_spread_base", "sell_spread_base",
            "refresh_s", "use_sl", "use_tp", "use_trail", "leverage",
            "is_sharpe_mean", "oos_sharpe_mean", "oos_pnl_pct_mean",
            "n_sym_profit_oos", "oos_max_dd_worst", "score"]
    show = robust if len(robust) else lb
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(show[cols].head(15).to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))
    print(f"{'=' * 92}")
    print("Saved: results/leaderboard_full.csv | leaderboard_robust.csv | top_detail.json")


if __name__ == "__main__":
    main()
