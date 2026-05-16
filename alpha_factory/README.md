# Alpha Factory — đào alpha số lượng lớn cho Hummingbot

Pipeline tự **random-search số lượng lớn** các chiến lược market making &
arbitrage của Hummingbot, sàng lọc bằng backtest in-sample/out-of-sample,
rồi **paper trade trên giá Binance live** để chọn con chuyển sang live trade.

## Pipeline

```
download_data.py        ->  tải klines 1m, 10 symbol, 3 tháng         (data/)
pmm_backtest.py         ->  random-search 4000 config PMM V2 (triple barrier)
avellaneda_backtest.py  ->  random-search 4000 config Avellaneda-Stoikov
grid_backtest.py        ->  random-search 4000 config Grid Trading
statarb_backtest.py     ->  random-search 4000 config Statistical Arbitrage
   => leaderboard_*.csv + portfolio_combined.json                     (results/)
paper_trade.py          ->  forward-test portfolio trên giá Binance LIVE (paper_results/)
```

## 4 họ chiến lược được random

| Họ | Engine | Knob random | Khớp Hummingbot |
|----|--------|-------------|-----------------|
| **PMM V2** | `pmm_backtest.py` | n_levels, buy/sell spreads, spread_step, amount dist, refresh, cooldown, stop_loss, take_profit, tp_order_type, time_limit, trailing_stop, leverage (~15) | `controllers/market_making/pmm_simple.py` |
| **Avellaneda-Stoikov** | `avellaneda_backtest.py` | risk_factor γ, base_spread, vol_mult, vol_window, order_notional, refresh, inventory_target, max_inventory (~8) | `hummingbot/strategy/avellaneda_market_making/` |
| **Grid Trading** | `grid_backtest.py` | n_grid, grid_step, tp_steps, recenter_bars, recenter_threshold, leverage (~7) | `controllers/generic/grid_strike.py` |
| **Statistical Arbitrage** | `statarb_backtest.py` | pair(A,B), lookback, entry_z, exit_z, stop_z, hold_limit (~6) | `controllers/generic/stat_arb.py` |

Không tích hợp: XEMM / arbitrage_controller / cross_exchange / amm_arb (cần 2
sàn hoặc DEX — không backtest bằng data 1 sàn), DMan Maker (≈ PMM + DCA),
PMM Dynamic (≈ Avellaneda — spread theo NATR), Liquidity Mining (≈ PMM đa cặp),
funding_rate_arb (cần dữ liệu funding rate).

## Phương pháp

- **In-sample / out-of-sample**: 2 tháng IS + 1 tháng OOS, chỉ giữ alpha tốt cả 2.
- **Market-neutral**: tách beta tồn kho (excess vs buy&hold) — đo alpha thuần.
- **Fee thực tế**: maker 0.02% / taker 0.05% (Binance Futures).
- **Không look-ahead**: lệnh đặt ở bar i chỉ khớp từ bar i+1.
- Engine numba: ~40.000 backtest / ~70 giây mỗi họ.
- **Hạn chế đã biết**: mô hình fill "touch = fill" → lạc quan hơn thực tế (chưa
  mô phỏng hàng đợi maker). Vì vậy paper trade live là bước kiểm chứng bắt buộc.

## Kết quả sweep (data 2026-02-16 → 2026-05-16)

| Họ | Robust / 4000 | Nhận xét |
|----|--------------|----------|
| Grid Trading | 1741 (43%) | Tốt nhất — crypto kỳ này ranging mạnh (chú ý touch=fill) |
| Avellaneda-Stoikov | 234 (5.8%) | Ổn định, Sharpe OOS 2–4, drawdown rất nhỏ |
| PMM V2 (triple barrier) | ~17 (0.4%) | Khó — đa số config lỗ vì phí + adverse selection |
| Statistical Arbitrage | 0 | Không ra alpha bền — pairs crypto trend/tách nhau in-sample |

Portfolio paper trade: 15 PMM + 15 Avellaneda + 15 Grid + 8 StatArb = **53 alpha**
(StatArb đưa vào dạng đầu cơ để live tự kiểm chứng).

## Chạy lại

```bash
python download_data.py
python pmm_backtest.py        --n-configs 4000
python avellaneda_backtest.py --n-configs 4000
python grid_backtest.py       --n-configs 4000
python statarb_backtest.py    --n-configs 4000
python paper_trade.py         # chạy nền nhiều ngày, poll giá Binance live 60s
```

## Đọc kết quả paper trade

- `paper_results/STATUS.md`      — bảng xếp hạng dễ đọc
- `paper_results/leaderboard.csv` — xếp hạng đầy đủ (PnL%, Sharpe live, cột `strategy`)
- `paper_results/equity_log.csv`  — lịch sử equity từng phút
- `paper_results/state.pkl`       — trạng thái (crash vẫn resume được)

**Tiêu chí chuyển live**: alpha có paper Sharpe > 0 và PnL > 0 **ổn định nhiều
ngày**, khớp kỳ vọng backtest OOS. Không live ngay theo số backtest.

## Thư mục

```
*.py                   code (commit)
results/*_robust.csv   alpha đạt chuẩn từng họ (commit)
results/portfolio_combined.json  portfolio paper trade (commit)
data/                  klines cache — không commit, chạy download_data.py
paper_results/         output runtime — không commit
```
