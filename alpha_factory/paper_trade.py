"""
PMM Alpha Factory - Live Paper Trader
=====================================
Forward-test song song toan bo portfolio alpha tren GIA BINANCE LIVE.
Moi (alpha x symbol) la 1 PMMSim co trang thai, fill mo phong bang dung
engine triple-barrier da backtest. Log PnL tung alpha lien tuc ra CSV.

Chay nen nhieu ngay:  python paper_trade.py
Trang thai luu o paper_results/state.pkl -> crash van resume duoc.

Doc ket qua:  paper_results/leaderboard.csv  /  STATUS.md
"""
import json
import math
import os
import pickle
import time
import traceback
from datetime import datetime, timezone

import requests

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "results")
PAPER_DIR = os.path.join(BASE, "paper_results")
os.makedirs(PAPER_DIR, exist_ok=True)

STATE_PKL = os.path.join(PAPER_DIR, "state.pkl")
EQUITY_LOG = os.path.join(PAPER_DIR, "equity_log.csv")
LEADERBOARD = os.path.join(PAPER_DIR, "leaderboard.csv")
STATUS_MD = os.path.join(PAPER_DIR, "STATUS.md")

# 8 symbol thanh khoan tot (bo USDCUSDT - stablecoin khong trade, bo PAXG - mong)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
           "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
ALL_SYMBOLS = SYMBOLS + ["PAXGUSDT"]   # +PAXG: cac cap stat-arb co the dung

TOTAL_QUOTE = 10_000.0
STATARB_NOTIONAL = 3_000.0
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
BARS_PER_YEAR = 525_600

S_EMPTY, S_RESTING, S_INPOS, S_COOLDOWN = 0, 1, 2, 3


# ----------------------------------------------------------------------
# STATEFUL PMM V2 ENGINE (port cua backtest_pmm_v2, xu ly tung bar)
# ----------------------------------------------------------------------
class PMMSim:
    def __init__(self, cfg):
        self.n_levels = int(cfg["n_levels"])
        self.buy_spread_base = float(cfg["buy_spread_base"])
        self.sell_spread_base = float(cfg["sell_spread_base"])
        self.spread_step = float(cfg["spread_step"])
        self.amount_base_w = float(cfg["amount_base_w"])
        self.refresh_bars = max(1, int(cfg["refresh_s"]) // 60)
        self.cooldown_bars = int(cfg["cooldown_s"]) // 60
        self.use_sl = bool(cfg["use_sl"])
        self.stop_loss = float(cfg["stop_loss"])
        self.use_tp = bool(cfg["use_tp"])
        self.take_profit = float(cfg["take_profit"])
        self.tp_is_market = bool(cfg["tp_is_market"])
        self.time_limit_bars = max(1, int(cfg["time_limit_s"]) // 60)
        self.use_trail = bool(cfg["use_trail"])
        self.trail_act = float(cfg["trail_act"])
        self.trail_delta = float(cfg["trail_delta"])
        self.leverage = int(cfg["leverage"])

        ns = 2 * self.n_levels
        self.state = [S_EMPTY] * ns
        self.ord_px = [0.0] * ns
        self.ord_amt = [0.0] * ns
        self.entry_px = [0.0] * ns
        self.pos_amt = [0.0] * ns
        self.open_bar = [0] * ns
        self.trail_ext = [0.0] * ns
        self.cd_until = [0] * ns

        side_budget = TOTAL_QUOTE * self.leverage * 0.30
        wsum = sum(self.amount_base_w ** l for l in range(self.n_levels))
        self.notional = [side_budget * (self.amount_base_w ** l) / wsum
                         for l in range(self.n_levels)]

        self.realized = 0.0
        self.bar_idx = 0
        self.bars_since = 0
        self.has_placed = False
        self.n_fills = 0
        self.n_closes = 0
        self.volume = 0.0
        self.equity = TOTAL_QUOTE
        self.last_unreal = 0.0

    def step(self, hi, lo, mid):
        nl = self.n_levels
        ns = 2 * nl
        i = self.bar_idx

        # --- A. triple barrier ---
        for s in range(ns):
            if self.state[s] != S_INPOS:
                continue
            is_long = s < nl
            ent = self.entry_px[s]
            amt = self.pos_amt[s]
            closed = False
            close_px = 0.0
            exit_taker = True
            if is_long:
                if self.use_trail and hi > self.trail_ext[s]:
                    self.trail_ext[s] = hi
                if self.use_sl and lo <= ent * (1.0 - self.stop_loss):
                    close_px = ent * (1.0 - self.stop_loss); closed = True
                elif self.use_tp and hi >= ent * (1.0 + self.take_profit):
                    close_px = ent * (1.0 + self.take_profit)
                    exit_taker = self.tp_is_market; closed = True
                elif self.use_trail and self.trail_ext[s] >= ent * (1.0 + self.trail_act) \
                        and lo <= self.trail_ext[s] * (1.0 - self.trail_delta):
                    close_px = self.trail_ext[s] * (1.0 - self.trail_delta); closed = True
                elif (i - self.open_bar[s]) >= self.time_limit_bars:
                    close_px = mid; closed = True
                if closed:
                    gross = amt * (close_px - ent)
                    fee = close_px * amt * (TAKER_FEE if exit_taker else MAKER_FEE)
                    self.realized += gross - fee
            else:
                if self.use_trail and (self.trail_ext[s] == 0.0 or lo < self.trail_ext[s]):
                    self.trail_ext[s] = lo
                if self.use_sl and hi >= ent * (1.0 + self.stop_loss):
                    close_px = ent * (1.0 + self.stop_loss); closed = True
                elif self.use_tp and lo <= ent * (1.0 - self.take_profit):
                    close_px = ent * (1.0 - self.take_profit)
                    exit_taker = self.tp_is_market; closed = True
                elif self.use_trail and self.trail_ext[s] != 0.0 \
                        and self.trail_ext[s] <= ent * (1.0 - self.trail_act) \
                        and hi >= self.trail_ext[s] * (1.0 + self.trail_delta):
                    close_px = self.trail_ext[s] * (1.0 + self.trail_delta); closed = True
                elif (i - self.open_bar[s]) >= self.time_limit_bars:
                    close_px = mid; closed = True
                if closed:
                    gross = amt * (ent - close_px)
                    fee = close_px * amt * (TAKER_FEE if exit_taker else MAKER_FEE)
                    self.realized += gross - fee
            if closed:
                self.n_closes += 1
                self.volume += close_px * amt
                if self.cooldown_bars > 0:
                    self.state[s] = S_COOLDOWN
                    self.cd_until[s] = i + self.cooldown_bars
                else:
                    self.state[s] = S_EMPTY

        # --- B. khop lenh maker RESTING ---
        for s in range(ns):
            if self.state[s] != S_RESTING:
                continue
            is_long = s < nl
            if is_long and lo <= self.ord_px[s]:
                self.entry_px[s] = self.ord_px[s]
                self.pos_amt[s] = self.ord_amt[s]
                self.open_bar[s] = i
                self.trail_ext[s] = self.ord_px[s]
                self.realized -= self.ord_px[s] * self.ord_amt[s] * MAKER_FEE
                self.volume += self.ord_px[s] * self.ord_amt[s]
                self.state[s] = S_INPOS
                self.n_fills += 1
            elif (not is_long) and hi >= self.ord_px[s]:
                self.entry_px[s] = self.ord_px[s]
                self.pos_amt[s] = self.ord_amt[s]
                self.open_bar[s] = i
                self.trail_ext[s] = 0.0
                self.realized -= self.ord_px[s] * self.ord_amt[s] * MAKER_FEE
                self.volume += self.ord_px[s] * self.ord_amt[s]
                self.state[s] = S_INPOS
                self.n_fills += 1

        # --- C. het cooldown ---
        for s in range(ns):
            if self.state[s] == S_COOLDOWN and i >= self.cd_until[s]:
                self.state[s] = S_EMPTY

        # --- D. refresh dat lenh ---
        self.bars_since += 1
        if (not self.has_placed) or self.bars_since >= self.refresh_bars:
            for s in range(ns):
                if self.state[s] in (S_EMPTY, S_RESTING):
                    if s < nl:
                        l = s
                        px = mid * (1.0 - (self.buy_spread_base + self.spread_step * l))
                    else:
                        l = s - nl
                        px = mid * (1.0 + (self.sell_spread_base + self.spread_step * l))
                    self.ord_px[s] = px
                    self.ord_amt[s] = self.notional[l] / px
                    self.state[s] = S_RESTING
            self.bars_since = 0
            self.has_placed = True

        # --- E. equity ---
        unreal = 0.0
        for s in range(ns):
            if self.state[s] == S_INPOS:
                if s < nl:
                    unreal += self.pos_amt[s] * (mid - self.entry_px[s])
                else:
                    unreal += self.pos_amt[s] * (self.entry_px[s] - mid)
        self.last_unreal = unreal
        self.equity = TOTAL_QUOTE + self.realized + unreal
        self.bar_idx += 1
        return self.equity

    def num_open(self):
        return sum(1 for st in self.state if st == S_INPOS)


# ----------------------------------------------------------------------
# STATEFUL AVELLANEDA-STOIKOV ENGINE (port cua backtest_avellaneda)
# ----------------------------------------------------------------------
class AvellanedaSim:
    def __init__(self, cfg):
        self.gamma = float(cfg["gamma"])
        self.base_spread = float(cfg["base_spread"])
        self.vol_mult = float(cfg["vol_mult"])
        self.vol_window = int(cfg["vol_window"])
        self.order_notional = float(cfg["order_notional"])
        self.refresh_bars = int(cfg["refresh_bars"])
        self.target_pct = float(cfg["target_pct"])
        self.max_inv_mult = float(cfg["max_inv_mult"])

        self.ring = [0.0] * self.vol_window
        self.rsum = 0.0
        self.rsumsq = 0.0
        self.filled = 0
        self.ptr = 0
        self.prev_close = None
        self.base = self.cash = self.base0 = self.cash0 = None
        self.bid_px = self.bid_amt = self.ask_px = self.ask_amt = 0.0
        self.has_orders = False
        self.bars_since = 0
        self.n_fills = 0
        self.n_closes = 0          # Avellaneda quote lien tuc, khong dem close
        self.volume = 0.0
        self.equity = TOTAL_QUOTE

    def num_open(self):
        return int(self.bid_amt > 0) + int(self.ask_amt > 0)

    def step(self, hi, lo, mid):
        if self.prev_close is None:
            self.prev_close = mid
            self.base = (TOTAL_QUOTE * self.target_pct) / mid
            self.cash = TOTAL_QUOTE * (1.0 - self.target_pct)
            self.base0, self.cash0 = self.base, self.cash

        # bien dong sigma (rolling std, ring buffer)
        r = (mid - self.prev_close) / self.prev_close
        self.prev_close = mid
        vw = self.vol_window
        if self.filled < vw:
            self.ring[self.ptr] = r
            self.rsum += r
            self.rsumsq += r * r
            self.filled += 1
        else:
            old = self.ring[self.ptr]
            self.rsum += r - old
            self.rsumsq += r * r - old * old
            self.ring[self.ptr] = r
        self.ptr = (self.ptr + 1) % vw
        if self.filled > 1:
            var = self.rsumsq / self.filled - (self.rsum / self.filled) ** 2
            sigma = var ** 0.5 if var > 0 else 0.0
        else:
            sigma = self.base_spread

        # khop lenh dat tu bar truoc
        if self.has_orders:
            if self.bid_amt > 0 and lo <= self.bid_px:
                cost = self.bid_px * self.bid_amt
                if self.cash >= cost:
                    self.cash -= cost + cost * MAKER_FEE
                    self.base += self.bid_amt
                    self.n_fills += 1
                    self.volume += cost
                    self.bid_amt = 0.0
            if self.ask_amt > 0 and hi >= self.ask_px:
                if self.base >= self.ask_amt:
                    rev = self.ask_px * self.ask_amt
                    self.cash += rev - rev * MAKER_FEE
                    self.base -= self.ask_amt
                    self.n_fills += 1
                    self.volume += rev
                    self.ask_amt = 0.0

        # refresh quote Avellaneda
        self.bars_since += 1
        if (not self.has_orders) or self.bars_since >= self.refresh_bars:
            base_val = self.base * mid
            total = base_val + self.cash
            q = (base_val - total * self.target_pct) / self.order_notional
            half = self.base_spread + self.vol_mult * sigma
            center = mid * (1.0 - q * self.gamma * sigma)
            self.bid_px = center * (1.0 - half)
            self.ask_px = center * (1.0 + half)
            self.bid_amt = self.order_notional / self.bid_px
            self.ask_amt = self.order_notional / self.ask_px
            if q > self.max_inv_mult:
                self.bid_amt = 0.0
            if q < -self.max_inv_mult:
                self.ask_amt = 0.0
            self.has_orders = True
            self.bars_since = 0

        # equity market-neutral (excess vs buy&hold) de so sanh cong bang voi PMM
        raw = self.cash + self.base * mid
        hold = self.cash0 + self.base0 * mid
        self.equity = TOTAL_QUOTE + raw - hold
        return self.equity


# ----------------------------------------------------------------------
# STATEFUL GRID TRADING ENGINE (port cua backtest_grid)
# ----------------------------------------------------------------------
class GridSim:
    def __init__(self, cfg):
        self.n_grid = int(cfg["n_grid"])
        self.grid_step = float(cfg["grid_step"])
        self.tp_steps = int(cfg["tp_steps"])
        self.recenter_bars = int(cfg["recenter_bars"])
        self.recenter_threshold = float(cfg["recenter_threshold"])
        self.leverage = int(cfg["leverage"])
        self.target_pct = float(cfg["target_pct"])

        ns = 2 * self.n_grid
        self.st = [0] * ns          # 0 empty, 1 resting, 2 inpos
        self.ord_px = [0.0] * ns
        self.amt = [0.0] * ns
        self.exit_px = [0.0] * ns
        self.notional = TOTAL_QUOTE * self.leverage * 0.30 / self.n_grid
        self.center = None
        self.bars_since = 0
        self.placed = False
        self.base = self.cash = self.base0 = self.cash0 = None
        self.n_fills = 0
        self.n_closes = 0
        self.volume = 0.0
        self.equity = TOTAL_QUOTE

    def num_open(self):
        return sum(1 for s in self.st if s == 2)

    def step(self, hi, lo, mid):
        if self.center is None:
            self.center = mid
            self.base = (TOTAL_QUOTE * self.target_pct) / mid
            self.cash = TOTAL_QUOTE * (1.0 - self.target_pct)
            self.base0, self.cash0 = self.base, self.cash
        ng = self.n_grid
        ns = 2 * ng

        # A. khop lenh chot (exit)
        for s in range(ns):
            if self.st[s] != 2:
                continue
            if s < ng and hi >= self.exit_px[s]:
                rev = self.exit_px[s] * self.amt[s]
                self.cash += rev - rev * MAKER_FEE
                self.base -= self.amt[s]
                self.n_fills += 1
                self.n_closes += 1
                self.volume += rev
                self.st[s] = 0
            elif s >= ng and lo <= self.exit_px[s]:
                cost = self.exit_px[s] * self.amt[s]
                self.cash -= cost + cost * MAKER_FEE
                self.base += self.amt[s]
                self.n_fills += 1
                self.n_closes += 1
                self.volume += cost
                self.st[s] = 0

        # B. khop lenh vao (entry)
        for s in range(ns):
            if self.st[s] != 1:
                continue
            if s < ng and lo <= self.ord_px[s]:
                cost = self.ord_px[s] * self.amt[s]
                self.cash -= cost + cost * MAKER_FEE
                self.base += self.amt[s]
                self.n_fills += 1
                self.volume += cost
                self.exit_px[s] = self.ord_px[s] * (1.0 + self.tp_steps * self.grid_step)
                self.st[s] = 2
            elif s >= ng and hi >= self.ord_px[s]:
                rev = self.ord_px[s] * self.amt[s]
                self.cash += rev - rev * MAKER_FEE
                self.base -= self.amt[s]
                self.n_fills += 1
                self.volume += rev
                self.exit_px[s] = self.ord_px[s] * (1.0 - self.tp_steps * self.grid_step)
                self.st[s] = 2

        # C. re-center & rai lenh vao
        self.bars_since += 1
        drift = abs(mid - self.center) / self.center
        if (not self.placed) or self.bars_since >= self.recenter_bars \
                or drift > self.recenter_threshold:
            self.center = mid
            for s in range(ns):
                if self.st[s] in (0, 1):
                    if s < ng:
                        px = self.center * (1.0 - (s + 1) * self.grid_step)
                    else:
                        px = self.center * (1.0 + (s - ng + 1) * self.grid_step)
                    self.ord_px[s] = px
                    self.amt[s] = self.notional / px
                    self.st[s] = 1
            self.bars_since = 0
            self.placed = True

        raw = self.cash + self.base * mid
        hold = self.cash0 + self.base0 * mid
        self.equity = TOTAL_QUOTE + raw - hold
        return self.equity


# ----------------------------------------------------------------------
# STATEFUL STATISTICAL ARBITRAGE ENGINE (port cua backtest_statarb)
# ----------------------------------------------------------------------
class StatArbSim:
    """Pairs trading: nhan close cua 2 symbol moi step (clA, clB)."""
    def __init__(self, cfg):
        self.sym_a = cfg["sym_a"]
        self.sym_b = cfg["sym_b"]
        self.lookback = int(cfg["lookback"])
        self.entry = float(cfg["entry"])
        self.exit_thr = float(cfg["exit_thr"])
        self.stop_z = float(cfg["stop_z"])
        self.hold_limit = int(cfg["hold_limit"])
        self.notional = STATARB_NOTIONAL

        self.rx = [0.0] * self.lookback
        self.ry = [0.0] * self.lookback
        self.sx = self.sy = self.sxx = self.sxy = self.syy = 0.0
        self.filled = 0
        self.ptr = 0
        self.state = 0          # 0 flat, +1 long-spread, -1 short-spread
        self.entryA = self.entryB = 0.0
        self.amtA = self.amtB = 0.0
        self.open_bar = 0
        self.bar = 0
        self.realized = 0.0
        self.n_fills = 0
        self.n_closes = 0
        self.volume = 0.0
        self.equity = TOTAL_QUOTE

    def num_open(self):
        return 1 if self.state != 0 else 0

    def step(self, cl_a, cl_b):
        x = math.log(cl_b)
        y = math.log(cl_a)
        lb = self.lookback
        if self.filled < lb:
            self.rx[self.ptr] = x
            self.ry[self.ptr] = y
            self.sx += x; self.sy += y
            self.sxx += x * x; self.sxy += x * y; self.syy += y * y
            self.filled += 1
        else:
            ox = self.rx[self.ptr]; oy = self.ry[self.ptr]
            self.sx += x - ox; self.sy += y - oy
            self.sxx += x * x - ox * ox
            self.sxy += x * y - ox * oy
            self.syy += y * y - oy * oy
            self.rx[self.ptr] = x; self.ry[self.ptr] = y
        self.ptr = (self.ptr + 1) % lb
        self.bar += 1

        if self.filled < lb:
            self.equity = TOTAL_QUOTE + self.realized
            return self.equity

        nn = lb
        denom = nn * self.sxx - self.sx * self.sx
        beta = (nn * self.sxy - self.sx * self.sy) / denom if denom != 0 else 1.0
        mean_sp = self.sy / nn - beta * self.sx / nn
        var_sp = (self.syy / nn - (self.sy / nn) ** 2) \
            + beta * beta * (self.sxx / nn - (self.sx / nn) ** 2) \
            - 2.0 * beta * (self.sxy / nn - self.sx * self.sy / (nn * nn))
        std_sp = var_sp ** 0.5 if var_sp > 1e-12 else 1e-6
        z = ((y - beta * x) - mean_sp) / std_sp
        pa, pb = cl_a, cl_b

        if self.state != 0:
            held = self.bar - self.open_bar
            do_exit = False
            if self.state == 1:
                if z >= -self.exit_thr or z <= -self.stop_z or held >= self.hold_limit:
                    do_exit = True
            else:
                if z <= self.exit_thr or z >= self.stop_z or held >= self.hold_limit:
                    do_exit = True
            if do_exit:
                if self.state == 1:
                    pnl = self.amtA * (pa - self.entryA) + self.amtB * (self.entryB - pb)
                else:
                    pnl = self.amtA * (self.entryA - pa) + self.amtB * (pb - self.entryB)
                self.realized += pnl - (pa * self.amtA + pb * self.amtB) * TAKER_FEE
                self.state = 0
                self.n_fills += 1
                self.n_closes += 1

        if self.state == 0:
            if z >= self.entry:
                self.state = -1
            elif z <= -self.entry:
                self.state = 1
            if self.state != 0:
                self.entryA, self.entryB = pa, pb
                self.amtA = self.notional / pa
                self.amtB = self.notional / pb
                self.realized -= (pa * self.amtA + pb * self.amtB) * TAKER_FEE
                self.open_bar = self.bar
                self.n_fills += 1

        unreal = 0.0
        if self.state == 1:
            unreal = self.amtA * (pa - self.entryA) + self.amtB * (self.entryB - pb)
        elif self.state == -1:
            unreal = self.amtA * (self.entryA - pa) + self.amtB * (pb - self.entryB)
        self.equity = TOTAL_QUOTE + self.realized + unreal
        return self.equity


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def fetch_klines(symbol, limit=5):
    """Lay cac bar 1m gan nhat. Tra ve list bar DA DONG (bo bar dang chay)."""
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": symbol, "interval": "1m", "limit": limit},
                     timeout=15)
    data = r.json()
    bars = []
    for k in data[:-1]:  # bar cuoi = dang chay -> bo
        bars.append((int(k[0]), float(k[2]), float(k[3]), float(k[4])))  # ts,hi,lo,close
    return bars


# ----------------------------------------------------------------------
# RUNNER
# ----------------------------------------------------------------------
class PaperTrader:
    def __init__(self):
        self.alphas = json.load(open(os.path.join(RESULTS_DIR, "portfolio_combined.json")))
        self.sims = {}            # key -> Sim  (key = (aid,sym) cho MM, aid cho stat-arb)
        self.alpha_keys = {}      # aid -> [sim keys]
        self.alpha_base = {}      # aid -> von trien khai
        self.equity_hist = {}     # aid -> [agg_equity ...]
        self.last_global_ts = 0
        self.start_time = None
        self.bars = 0

    def _build(self):
        SIM_CLS = {"avellaneda": AvellanedaSim, "grid": GridSim, "pmm": PMMSim}
        for a in self.alphas:
            aid = a["alpha_id"]
            strat = a.get("strategy", "pmm")
            if strat == "statarb":
                self.sims[aid] = StatArbSim(a)
                self.alpha_keys[aid] = [aid]
            else:
                cls = SIM_CLS.get(strat, PMMSim)
                keys = []
                for sym in SYMBOLS:
                    self.sims[(aid, sym)] = cls(a)
                    keys.append((aid, sym))
                self.alpha_keys[aid] = keys
            self.alpha_base[aid] = TOTAL_QUOTE * len(self.alpha_keys[aid])
            self.equity_hist[aid] = []

    def init_fresh(self):
        self._build()
        self.last_global_ts = 0
        self.start_time = datetime.now(timezone.utc).isoformat()
        with open(EQUITY_LOG, "w") as f:
            f.write("ts,alpha_id,strategy,agg_equity,agg_pnl,agg_pnl_pct,"
                    "n_fills,n_closes,n_open\n")

    def save(self):
        with open(STATE_PKL, "wb") as f:
            pickle.dump({"sims": self.sims, "alpha_keys": self.alpha_keys,
                         "alpha_base": self.alpha_base, "equity_hist": self.equity_hist,
                         "last_global_ts": self.last_global_ts,
                         "start_time": self.start_time, "bars": self.bars}, f)

    def load(self):
        d = pickle.load(open(STATE_PKL, "rb"))
        self.sims = d["sims"]; self.alpha_keys = d["alpha_keys"]
        self.alpha_base = d["alpha_base"]; self.equity_hist = d["equity_hist"]
        self.last_global_ts = d["last_global_ts"]
        self.start_time = d["start_time"]; self.bars = d["bars"]

    def step_once(self):
        """Lay bar moi DONG BO cho moi symbol, day vao tat ca sim, log."""
        bars = {}
        for sym in ALL_SYMBOLS:
            try:
                kl = fetch_klines(sym)
            except Exception as e:
                print(f"  [warn] fetch {sym}: {e} -> bo qua chu ky de giu dong bo")
                return False
            bars[sym] = {b[0]: (b[1], b[2], b[3]) for b in kl}

        common = set(bars[ALL_SYMBOLS[0]])
        for sym in ALL_SYMBOLS[1:]:
            common &= set(bars[sym])
        new_ts = sorted(t for t in common if t > self.last_global_ts)
        if not new_ts:
            return False

        for ts in new_ts:
            for a in self.alphas:
                aid = a["alpha_id"]
                if a.get("strategy") == "statarb":
                    ca = bars[a["sym_a"]][ts][2]
                    cb = bars[a["sym_b"]][ts][2]
                    self.sims[aid].step(ca, cb)
                else:
                    for sym in SYMBOLS:
                        hi, lo, cl = bars[sym][ts]
                        self.sims[(aid, sym)].step(hi, lo, cl)
            self.last_global_ts = ts
            self.bars += 1

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(EQUITY_LOG, "a") as f:
            for a in self.alphas:
                aid = a["alpha_id"]
                keys = self.alpha_keys[aid]
                eq = sum(self.sims[k].equity for k in keys)
                fills = sum(self.sims[k].n_fills for k in keys)
                closes = sum(self.sims[k].n_closes for k in keys)
                nopen = sum(self.sims[k].num_open() for k in keys)
                base = self.alpha_base[aid]
                self.equity_hist[aid].append(eq)
                f.write(f"{now},{aid},{a.get('strategy', 'pmm')},{eq:.2f},"
                        f"{eq - base:.2f},{(eq - base) / base * 100:.4f},"
                        f"{fills},{closes},{nopen}\n")
        self.write_leaderboard(now)
        return True

    def write_leaderboard(self, now):
        rows = []
        for a in self.alphas:
            aid = a["alpha_id"]
            hist = self.equity_hist[aid]
            base = self.alpha_base[aid]
            pnl_pct = (hist[-1] - base) / base * 100.0
            sharpe = 0.0
            if len(hist) > 30:
                rets = [(hist[k] - hist[k - 1]) / base for k in range(1, len(hist))]
                m = sum(rets) / len(rets)
                sd = (sum((x - m) ** 2 for x in rets) / len(rets)) ** 0.5
                if sd > 0:
                    sharpe = m / sd * (BARS_PER_YEAR ** 0.5)
            fills = sum(self.sims[k].n_fills for k in self.alpha_keys[aid])
            bt_sharpe = float(a.get("oos_sharpe_mean", a.get("oos_sharpe", 0.0)))
            rows.append((aid, a.get("strategy", "pmm"), pnl_pct, sharpe,
                         int(fills), bt_sharpe))
        rows.sort(key=lambda r: r[2], reverse=True)
        with open(LEADERBOARD, "w") as f:
            f.write("rank,alpha_id,strategy,paper_pnl_pct,paper_sharpe,"
                    "n_fills,backtest_oos_sharpe\n")
            for rank, r in enumerate(rows, 1):
                f.write(f"{rank},{r[0]},{r[1]},{r[2]:.4f},{r[3]:.2f},{r[4]},{r[5]:.3f}\n")
        with open(STATUS_MD, "w") as f:
            f.write("# Paper Trade Status\n\n")
            f.write(f"- Cap nhat: {now} UTC | Bat dau: {self.start_time}\n")
            f.write(f"- So phut da chay: ~{self.bars}\n")
            f.write(f"- Alpha: {len(self.alphas)} (PMM/Avellaneda/Grid/StatArb)\n\n")
            f.write("| # | Alpha | Strategy | Paper PnL% | Paper Sharpe | Fills | BT OOS Sharpe |\n")
            f.write("|---|-------|----------|-----------|--------------|-------|---------------|\n")
            for rank, r in enumerate(rows, 1):
                f.write(f"| {rank} | {r[0]} | {r[1]} | {r[2]:+.3f}% | {r[3]:.2f} "
                        f"| {r[4]} | {r[5]:.2f} |\n")

    def run(self):
        if os.path.exists(STATE_PKL):
            try:
                self.load()
                print(f"Resume: ~{self.bars} phut da chay, {len(self.alphas)} alpha")
            except Exception as e:
                print(f"Load state fail ({e}) -> khoi tao moi")
                self.sims.clear()
                self.init_fresh()
        else:
            self.init_fresh()
            print(f"Khoi tao moi: {len(self.alphas)} alpha, {len(self.sims)} sim")

        print("Bat dau forward-test (gia Binance live, poll 60s)... Ctrl+C de dung.")
        while True:
            try:
                if self.step_once():
                    self.save()
                    top = open(LEADERBOARD).readlines()[1:4]
                    tag = " | ".join(t.split(",")[1] + "/" + t.split(",")[2]
                                     + " " + t.split(",")[3] + "%" for t in top)
                    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] "
                          f"min~{self.bars} | TOP: {tag}")
            except Exception:
                print("ERROR loop:\n" + traceback.format_exc())
            time.sleep(60)


if __name__ == "__main__":
    PaperTrader().run()
