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
REFIT_DAYS = 40
HOLD_MR = 10
HOLD_COMP = 15
K_TOP = 5
Z_ENTRY = 1.5
TX_BPS = 10
INITIAL = 100_000
N_TRIALS = 3
TICKERS_PER_TRIAL = 20
START = "2025-01-01"


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

    # Strategy state
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

    tx = TX_BPS / 10000
    current_z = None
    current_comp = None
    n_refits = 0
    warmup = 120  # min days before first trade

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

                current_comp = np.zeros(N)
                if current_z is not None:
                    for i, tk in enumerate(tickers):
                        zi = float(current_z[i]) if i < len(current_z) else 0
                        F = ff.scores.get(tk, 0)
                        di = float(delta[i]) if i < len(delta) else 0
                        current_comp[i] = -0.4 * zi + 0.3 * F * 5 - 0.3 * di

                n_refits += 1
                if n_refits <= 3 or n_refits % 3 == 0:
                    print(f"    [{dates[t].date()}] Refit #{n_refits}: "
                          f"edges={int(np.sum(gb.W!=0))} s={gb.s:.3f}")
            except Exception as e:
                import traceback
                logging.warning(f"Refit error: {e}\n{traceback.format_exc()}")

        # ── Daily P&L ──
        ret = returns[t] if t < len(returns) else np.zeros(N)

        def step_positions(positions, equity_list):
            pnl = 0.0
            alive = []
            for idx, direction, remaining in positions:
                if idx < N:
                    pnl += direction * ret[idx]
                remaining -= 1
                if remaining > 0:
                    alive.append((idx, direction, remaining))
                else:
                    pnl -= tx
            equity_list.append(equity_list[-1] * (1 + pnl))
            return alive, pnl

        pos_mr, pnl_mr = step_positions(pos_mr, equity_mr)
        pos_comp, pnl_comp = step_positions(pos_comp, equity_comp)
        pos_rand, _ = step_positions(pos_rand, equity_rand)
        equity_spy.append(equity_spy[-1] * (1 + ret[spy_idx]))

        # ── Open trades on refit ──
        if day_in_bt % REFIT_DAYS == 0 and current_z is not None:
            z = np.nan_to_num(current_z[:N], nan=0)
            order = np.argsort(z)

            # MR
            for idx in order[:K_TOP]:
                if z[idx] < -Z_ENTRY:
                    pos_mr.append((idx, +1, HOLD_MR))
                    n_trades_mr += 1
            for idx in order[-K_TOP:]:
                if z[idx] > Z_ENTRY:
                    pos_mr.append((idx, -1, HOLD_MR))
                    n_trades_mr += 1

            # Composite
            c = np.nan_to_num(current_comp[:N], nan=0)
            c_order = np.argsort(c)
            for idx in c_order[-K_TOP:]:
                pos_comp.append((idx, +1, HOLD_COMP))
                n_trades_comp += 1
            for idx in c_order[:K_TOP]:
                pos_comp.append((idx, -1, HOLD_COMP))
                n_trades_comp += 1

            # Random
            rng = np.random.default_rng(seed=t)
            for idx in rng.choice(N, size=min(K_TOP, N), replace=False):
                pos_rand.append((idx, +1 if rng.random() > 0.5 else -1, 10))
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
    print(f"  {N_TRIALS} trials × {TICKERS_PER_TRIAL} tickers | TX={TX_BPS}bps")
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
