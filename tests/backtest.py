"""
BACKTEST — Walk-Forward Predictive Capacity Test
==================================================
Tests the model's REAL predictive power with realistic trading simulation.

Uses RANDOM SECTOR SUBSETS (~20 tickers) for speed.
Runs 3 independent trials with different sectors to get robust statistics.

Strategies (long/short):
  1. Mean Reversion:  buy z<-1.5, sell z>+1.5 → hold 10 days
  2. Composite:       z-score + fundamental + δ mispricing
  3. Random:          random picks (null hypothesis)

Walk-forward: retrain every 40 days, trade on fresh signals only.
Cost: 10bps round-trip.

python backtest.py
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import timedelta

from db.database_manager import DatabaseManager
from core.graph_builder import GraphBuilder
from core.fundamental_filter import FundamentalFilter
from core.heat_engine import HeatEngine
import config

logging.basicConfig(level=logging.WARNING)

# ── Config ──
REFIT_DAYS = 20
HOLD_MR = 5
HOLD_COMP = 7
K_TOP = 5
Z_ENTRY_BASE = 0.8        # P1.1: base threshold, adjusted by VIX
TX_BPS_LARGE = 5           # P1.5: large-cap US (SPY, AAPL, etc.)
TX_BPS_MID = 15            # P1.5: mid-cap / intl developed
TX_BPS_SMALL = 25          # P1.5: EM / small-cap / exotic
MAX_POS_PCT = 0.05         # P1.3: max 5% of equity per position
STOP_ATR_MULT = 2.0        # P1.4: trailing stop at 2× ATR
STOP_HARD = -0.10          # P1.4: hard stop at -10%
INITIAL = 100_000
N_TRIALS = 3
TICKERS_PER_TRIAL = 20
START = "2025-01-01"


def get_tx_cost(ticker):
    """P1.5: realistic cost by ticker type."""
    import config as cfg
    country = cfg.TICKER_COUNTRY.get(ticker, "US")
    zone = cfg.COUNTRY_TO_ZONE.get(country, "USD")
    if zone == "USD" and ticker not in cfg.ETF_TICKERS:
        return TX_BPS_LARGE / 10000   # 5bps for large US
    elif zone in ("EUR", "ASIA"):
        return TX_BPS_MID / 10000     # 15bps for intl developed
    else:
        return TX_BPS_SMALL / 10000   # 25bps for EM / exotic


def adaptive_z_entry(engine):
    """P1.1: Z_ENTRY adapts to regime. Lower in calm (more trades), higher in stress."""
    vix = 15.0  # default
    try:
        if hasattr(engine.gb, 'dimensions') and 'vix' in engine.gb.dimensions:
            v = engine.gb.dimensions['vix']
            if hasattr(v, 'iloc') and len(v) > 0:
                vix = float(v.iloc[-1])
        elif hasattr(engine.gb, 'vix_level'):
            vix = float(engine.gb.vix_level)
    except Exception:
        pass
    # Z = base + 0.03 per VIX point above 15 (more selective in stress)
    z = Z_ENTRY_BASE + max(0, (vix - 15)) * 0.03
    return np.clip(z, 0.5, 2.5)


def kelly_size(win_rate, avg_win, avg_loss, equity, max_pct=MAX_POS_PCT):
    """P1.3: Half-Kelly position sizing. Returns exposure multiplier ~1.0."""
    if avg_loss == 0 or win_rate <= 0:
        return 1.0  # equal weight fallback
    b = avg_win / abs(avg_loss)
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b
    half_kelly = kelly / 2
    # Clamp to [0.5, 1.5] — don't deviate too far from equal weight
    return np.clip(half_kelly + 0.5, 0.5, 1.5)


def optimize_composite_weights(engine, returns_win, tickers, N):
    """P1.2: Walk-forward weight optimization. Tests grid of weights and picks best Sharpe."""
    if engine.z_scores is None or len(engine.z_scores) == 0:
        return (-0.4, 0.3, -0.3)

    ff_scores = {tk: engine.ff.scores.get(tk, 0) for tk in tickers}
    mp = getattr(engine, 'mispricing', None)

    best_w = (-0.4, 0.3, -0.3)
    best_sh = -999

    # Small grid search over weight combinations
    for wz in [-0.6, -0.4, -0.2]:
        for wf in [0.0, 0.15, 0.3]:
            for wd in [-0.4, -0.2, 0.0]:
                # Simulate composite signal over last 60 days
                daily_ret = []
                for t in range(max(0, len(engine.z_scores) - 60), len(engine.z_scores)):
                    z = engine.z_scores[t]
                    d = mp[t] if mp is not None and t < len(mp) else np.zeros(N)
                    scores = np.array([
                        wz * float(z[i]) + wf * ff_scores.get(tickers[i], 0) * 5 + wd * float(d[i])
                        for i in range(min(N, len(z)))
                    ])
                    order = np.argsort(scores)
                    # Top K long, bottom K short
                    r = returns_win[t] if t < len(returns_win) else np.zeros(N)
                    pnl = 0
                    for idx in order[-K_TOP:]:
                        if idx < N: pnl += r[idx] / K_TOP
                    for idx in order[:K_TOP]:
                        if idx < N: pnl -= r[idx] / K_TOP
                    daily_ret.append(pnl)

                if len(daily_ret) > 5:
                    dr = np.array(daily_ret)
                    sh = np.mean(dr) / (np.std(dr) + 1e-10) * np.sqrt(252)
                    if sh > best_sh:
                        best_sh = sh
                        best_w = (wz, wf, wd)

    return best_w


def pick_sector_subset(seed: int) -> list:
    """Pick a random subset of sectors and their tickers."""
    rng = np.random.default_rng(seed)
    all_sectors = list(config.TICKERS.keys())
    # Always include SPY for benchmark
    picked = ["SPY"]
    # Pick 3-4 random sectors
    sectors = rng.choice(
        [s for s in all_sectors if s not in ("FACTORS", "SECTORS", "COMMODITIES", "BONDS_GOVT", "BONDS_CORP")],
        size=min(4, len(all_sectors) - 5),
        replace=False,
    )
    for s in sectors:
        picked.extend(config.TICKERS[s])
    # Dedupe and limit
    seen = set()
    result = []
    for t in picked:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result[:TICKERS_PER_TRIAL]


def run_trial(db, trial_id: int) -> dict:
    """Run one trial with a random sector subset."""
    tickers = pick_sector_subset(seed=trial_id * 42 + 7)
    N = len(tickers)

    print(f"\n  Trial {trial_id + 1}: {N} tickers — {tickers[:8]}...")

    # Load prices for these tickers only
    prices_dict = {}
    for tk in tickers:
        try:
            df = db.get_prices(tk, start_date=START)
            if df is not None and not df.empty and "close" in df.columns:
                prices_dict[tk] = df["close"]
        except Exception:
            pass

    if len(prices_dict) < 10:
        print(f"    ⚠ Only {len(prices_dict)} tickers with data, skipping")
        return None

    prices = pd.DataFrame(prices_dict).sort_index().dropna(how="all")
    prices = prices.ffill().bfill()
    tickers = list(prices.columns)
    N = len(tickers)
    T = len(prices)
    returns = np.log(prices / prices.shift(1)).fillna(0).values
    dates = prices.index

    spy_idx = tickers.index("SPY") if "SPY" in tickers else 0
    print(f"    {N} tickers × {T} days ({dates[0].date()} → {dates[-1].date()})")

    # Fundamentals
    ff = FundamentalFilter(db)
    ff.compute_all()

    # Strategy state — positions now: (idx, direction, remaining, entry_price, peak_price, size, tx_cost)
    equity_mr = [INITIAL]
    equity_comp = [INITIAL]
    equity_rand = [INITIAL]
    equity_spy = [INITIAL]
    pos_mr = []
    pos_comp = []
    pos_rand = []
    n_trades_mr = 0
    n_trades_comp = 0
    n_trades_rand = 0
    trade_pnls_mr = []
    trade_pnls_comp = []

    current_z = None
    current_comp = None
    n_refits = 0
    warmup = 120  # min days before first trade
    comp_weights = (-0.4, 0.3, -0.3)  # P1.2: optimized per refit
    current_z_entry = Z_ENTRY_BASE

    # Pre-compute ATR for stop losses
    atr = pd.DataFrame(np.zeros_like(prices.values), index=prices.index, columns=tickers)
    for col in tickers:
        high_low = prices[col].rolling(14).max() - prices[col].rolling(14).min()
        atr[col] = high_low.rolling(14).mean() / prices[col]  # as % of price
    atr = atr.fillna(0.02)  # default 2% ATR

    for t in range(warmup, T):
        day_in_bt = t - warmup

        # ── Refit ──
        if day_in_bt % REFIT_DAYS == 0:
            try:
                win_start = max(0, t - 252)
                p_win = prices.iloc[win_start:t+1]
                r_win = np.log(p_win / p_win.shift(1)).fillna(0)

                # Build graph properly with __init__
                gb = GraphBuilder(db)
                gb.prices = p_win
                gb.tickers = tickers
                gb.N = N
                gb.returns = r_win
                gb.u = r_win.values.astype(np.float64)
                gb.T = len(gb.u)
                gb.volume = pd.DataFrame(np.ones_like(gb.u), index=p_win.index, columns=tickers)

                # Set attrs that load_data normally sets from macro
                gb.inflation_daily = pd.Series(0.0, index=p_win.index)
                gb.macro_velocity = pd.DataFrame(0.0, index=p_win.index, columns=tickers)

                gb.build()

                engine = HeatEngine(gb, ff)
                engine.solve(calibrate=True)

                current_z = engine.z_scores[-1] if len(engine.z_scores) > 0 else None
                mp = getattr(engine, 'mispricing', None)
                delta = mp[-1] if mp is not None and len(mp) > 0 else np.zeros(N)

                # P1.1: Adaptive Z_ENTRY
                current_z_entry = adaptive_z_entry(engine)

                # P1.2: Optimize composite weights on training window
                comp_weights = optimize_composite_weights(
                    engine, r_win.values, tickers, N
                )

                current_comp = np.zeros(N)
                if current_z is not None:
                    wz, wf, wd = comp_weights
                    for i, tk in enumerate(tickers):
                        zi = float(current_z[i]) if i < len(current_z) else 0
                        F = ff.scores.get(tk, 0)
                        di = float(delta[i]) if i < len(delta) else 0
                        current_comp[i] = wz * zi + wf * F * 5 + wd * di

                n_refits += 1
                if n_refits <= 3 or n_refits % 3 == 0:
                    print(f"    [{dates[t].date()}] Refit #{n_refits}: "
                          f"edges={int(np.sum(gb.W!=0))} s={gb.s:.3f}")
            except Exception as e:
                import traceback
                logging.warning(f"Refit error: {e}\n{traceback.format_exc()}")

        # ── Daily P&L with stops ──
        ret = returns[t] if t < len(returns) else np.zeros(N)
        cur_prices = prices.iloc[t].values if t < len(prices) else np.ones(N)
        cur_atr = atr.iloc[t].values if t < len(atr) else np.full(N, 0.02)

        def step_positions_with_stops(positions, equity_list):
            pnl = 0.0
            alive = []
            closed_pnls = []
            for pos in positions:
                idx, direction, remaining, entry_p, peak_p, size, tx_cost = pos
                if idx >= N:
                    continue
                pos_ret = direction * ret[idx] * size
                pnl += pos_ret

                # Update peak for trailing stop
                cur_p = cur_prices[idx] if idx < len(cur_prices) else entry_p
                new_peak = max(peak_p, cur_p) if direction > 0 else min(peak_p, cur_p)

                # P1.4: Check stops (hard stop only — trailing too aggressive for short holds)
                pnl_since_entry = direction * (cur_p - entry_p) / (entry_p + 1e-10)
                stopped = False
                if pnl_since_entry < STOP_HARD:  # Hard stop at -10%
                    stopped = True

                remaining -= 1
                if remaining <= 0 or stopped:
                    pnl -= tx_cost  # exit cost
                    closed_pnls.append(pos_ret)
                else:
                    alive.append((idx, direction, remaining, entry_p, new_peak, size, tx_cost))

            equity_list.append(equity_list[-1] * (1 + pnl))
            return alive, pnl, closed_pnls

        pos_mr, pnl_mr, closed_mr = step_positions_with_stops(pos_mr, equity_mr)
        pos_comp, pnl_comp, closed_comp = step_positions_with_stops(pos_comp, equity_comp)
        pos_rand, _, _ = step_positions_with_stops(pos_rand, equity_rand)
        equity_spy.append(equity_spy[-1] * (1 + ret[spy_idx]))
        trade_pnls_mr.extend(closed_mr)
        trade_pnls_comp.extend(closed_comp)

        # ── Open trades on refit ──
        if day_in_bt % REFIT_DAYS == 0 and current_z is not None:
            z = np.nan_to_num(current_z[:N], nan=0)
            order = np.argsort(z)

            # P1.3: Kelly sizing from historical trade PnLs
            if len(trade_pnls_mr) >= 10:
                wins = [p for p in trade_pnls_mr if p > 0]
                losses = [p for p in trade_pnls_mr if p <= 0]
                wr = len(wins) / len(trade_pnls_mr) if trade_pnls_mr else 0.5
                aw = np.mean(wins) if wins else 0.01
                al = np.mean(losses) if losses else -0.01
                pos_size = kelly_size(wr, aw, al, equity_mr[-1])
            else:
                pos_size = 1.0  # equal weight until enough data

            # MR trades with adaptive Z and Kelly sizing
            for idx in order[:K_TOP]:
                if z[idx] < -current_z_entry:
                    tx = get_tx_cost(tickers[idx])
                    entry_p = cur_prices[idx] if idx < len(cur_prices) else 1.0
                    pos_mr.append((idx, +1, HOLD_MR, entry_p, entry_p, pos_size, tx))
                    n_trades_mr += 1
            for idx in order[-K_TOP:]:
                if z[idx] > current_z_entry:
                    tx = get_tx_cost(tickers[idx])
                    entry_p = cur_prices[idx] if idx < len(cur_prices) else 1.0
                    pos_mr.append((idx, -1, HOLD_MR, entry_p, entry_p, pos_size, tx))
                    n_trades_mr += 1

            # Composite with optimized weights
            c = np.nan_to_num(current_comp[:N], nan=0)
            c_order = np.argsort(c)
            for idx in c_order[-K_TOP:]:
                tx = get_tx_cost(tickers[idx])
                entry_p = cur_prices[idx] if idx < len(cur_prices) else 1.0
                pos_comp.append((idx, +1, HOLD_COMP, entry_p, entry_p, 1.0, tx))
                n_trades_comp += 1
            for idx in c_order[:K_TOP]:
                tx = get_tx_cost(tickers[idx])
                entry_p = cur_prices[idx] if idx < len(cur_prices) else 1.0
                pos_comp.append((idx, -1, HOLD_COMP, entry_p, entry_p, 1.0, tx))
                n_trades_comp += 1

            # Random
            rng = np.random.default_rng(seed=t)
            for idx in rng.choice(N, size=min(K_TOP, N), replace=False):
                tx = get_tx_cost(tickers[idx])
                entry_p = cur_prices[idx] if idx < len(cur_prices) else 1.0
                pos_rand.append((idx, +1 if rng.random() > 0.5 else -1, 10,
                                entry_p, entry_p, 1.0, tx))
                n_trades_rand += 1

    # ── Metrics ──
    def metrics(eq, n_trades):
        eq = np.array(eq)
        ret = (eq[-1] / eq[0] - 1) * 100
        dr = np.diff(eq) / eq[:-1]
        dr = dr[np.isfinite(dr)]
        sh = np.mean(dr) / (np.std(dr) + 1e-10) * np.sqrt(252)
        pk = np.maximum.accumulate(eq)
        dd = np.min((eq - pk) / pk) * 100
        wp = np.mean(dr > 0) * 100 if len(dr) > 0 else 0
        return {"return": ret, "sharpe": sh, "max_dd": dd, "n_trades": n_trades, "win_pct": wp}

    return {
        "MR": metrics(equity_mr, n_trades_mr),
        "Composite": metrics(equity_comp, n_trades_comp),
        "Random": metrics(equity_rand, n_trades_rand),
        "SPY": metrics(equity_spy, 0),
        "tickers": tickers,
        "n_refits": n_refits,
    }


def main():
    db = DatabaseManager()

    print("=" * 80)
    print("  BACKTEST — Walk-Forward Predictive Capacity")
    print(f"  {N_TRIALS} trials × {TICKERS_PER_TRIAL} tickers | TX={TX_BPS_LARGE}/{TX_BPS_MID}/{TX_BPS_SMALL}bps")
    print("=" * 80)

    all_results = []

    for trial in range(N_TRIALS):
        r = run_trial(db, trial)
        if r:
            all_results.append(r)

    if not all_results:
        print("\n  ✗ No trials completed")
        return

    # ── Aggregate ──
    print(f"\n{'=' * 90}")
    print(f"  RESULTADOS AGREGADOS — {len(all_results)} trials")
    print(f"{'=' * 90}")
    print(f"  {'Estrategia':<20} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7} {'Win%':>6}")
    print(f"{'─' * 90}")

    for strat in ["MR", "Composite", "Random", "SPY"]:
        rets = [r[strat]["return"] for r in all_results]
        sharpes = [r[strat]["sharpe"] for r in all_results]
        dds = [r[strat]["max_dd"] for r in all_results]
        trades = [r[strat]["n_trades"] for r in all_results]
        wins = [r[strat]["win_pct"] for r in all_results]

        name = {"MR": "MR (z-score)", "SPY": "SPY B&H"}.get(strat, strat)
        print(f"  {name:<20} {np.mean(rets):>+7.1f}% {np.mean(sharpes):>6.2f} "
              f"{np.mean(dds):>7.1f}% {int(np.mean(trades)):>6} {np.mean(wins):>5.1f}%")

        # Per-trial detail
        for i, r in enumerate(all_results):
            m = r[strat]
            print(f"    Trial {i+1}: ret={m['return']:+.1f}% sh={m['sharpe']:.2f} dd={m['max_dd']:.1f}%")

    print(f"{'─' * 90}")

    # ── Alpha ──
    print(f"\n  ALPHA vs SPY / Random:")
    spy_avg = np.mean([r["SPY"]["return"] for r in all_results])
    rand_avg = np.mean([r["Random"]["return"] for r in all_results])

    for strat in ["MR", "Composite"]:
        avg_ret = np.mean([r[strat]["return"] for r in all_results])
        avg_sh = np.mean([r[strat]["sharpe"] for r in all_results])
        alpha_spy = avg_ret - spy_avg
        alpha_rand = avg_ret - rand_avg

        ic = "✅" if alpha_spy > 0 and avg_sh > 0.5 else "⚠" if avg_sh > 0 else "❌"
        name = {"MR": "MR (z-score)"}.get(strat, strat)
        print(f"  {ic} {name:<20} α_SPY={alpha_spy:+.1f}% α_Rand={alpha_rand:+.1f}% Sharpe={avg_sh:.2f}")

    # ── Verdict ──
    print(f"\n{'─' * 90}")
    best = max(["MR", "Composite"], key=lambda s: np.mean([r[s]["sharpe"] for r in all_results]))
    bs = np.mean([r[best]["sharpe"] for r in all_results])
    best_name = {"MR": "MR (z-score)"}.get(best, best)

    if bs > 1.0:
        print(f"  🏆 VEREDICTO: Capacidad predictiva FUERTE (Sharpe={bs:.2f}, best={best_name})")
    elif bs > 0.5:
        print(f"  ✅ VEREDICTO: Capacidad predictiva REAL (Sharpe={bs:.2f}, best={best_name})")
    elif bs > 0:
        print(f"  ⚠ VEREDICTO: Capacidad predictiva DÉBIL (Sharpe={bs:.2f}, best={best_name})")
    else:
        print(f"  ❌ VEREDICTO: Sin capacidad predictiva (Sharpe={bs:.2f})")
    print(f"{'=' * 90}")

    # Save
    save = {}
    for strat in ["MR", "Composite", "Random", "SPY"]:
        save[strat] = {
            "mean_return": float(np.mean([r[strat]["return"] for r in all_results])),
            "mean_sharpe": float(np.mean([r[strat]["sharpe"] for r in all_results])),
            "mean_max_dd": float(np.mean([r[strat]["max_dd"] for r in all_results])),
            "trials": [r[strat] for r in all_results],
        }
    with open("backtest_results.json", "w") as f:
        json.dump(save, f, indent=2, default=float)
    print(f"\n  Guardado: backtest_results.json")


if __name__ == "__main__":
    main()
