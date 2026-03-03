"""
CRISIS BACKTEST — P4.1
========================
Walk-forward backtest focused on 3 crisis periods:
  1. Volmageddon (2017-09 → 2018-06) — VIX spike regime detection
  2. COVID crash (2019-09 → 2020-09) — drawdown protection, credit delta lead
  3. Fed Rate Hikes (2021-09 → 2023-03) — sustained bear, defensive signals

For each crisis:
  - Full pipeline: graph → O-U → signals → walk-forward trading
  - UKF s(t) trajectory analysis: does s drop BEFORE VIX spikes?
  - Credit spread delta analysis: does it widen BEFORE drawdown?
  - Equity curves: MR strategy vs SPY B&H vs Random

Usage:
    python tests/crisis_backtest.py
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime

from db.database_manager import DatabaseManager
from core.graph_builder import GraphBuilder
from core.fundamental_filter import FundamentalFilter
from core.heat_engine import HeatEngine
import config

logging.basicConfig(level=logging.WARNING)

# ── Refuge tickers (bonds/gold) ──
REFUGE_TICKERS = {"TLT", "GLD", "SHY", "TIP", "IEF"}

# ── Crisis definitions ──
CRISIS_PERIODS = [
    {
        "name": "Volmageddon",
        "start": "2017-09-01",
        "end": "2018-06-30",
        "description": "VIX spike Feb 5 2018, XIV liquidation",
        "vix_peak_date": "2018-02-06",
        "crisis_start": "2018-02-01",
        "expected_regime": "stress",
    },
    {
        "name": "COVID Crash",
        "start": "2019-09-01",
        "end": "2020-09-30",
        "description": "Pandemic crash Mar 2020, fastest bear market ever",
        "vix_peak_date": "2020-03-16",
        "crisis_start": "2020-02-20",
        "expected_regime": "crisis",
    },
    {
        "name": "Fed Rate Hikes",
        "start": "2021-09-01",
        "end": "2023-03-31",
        "description": "Fed tightening cycle, sustained bear 2022",
        "vix_peak_date": "2022-06-13",
        "crisis_start": "2022-01-03",
        "expected_regime": "stress",
    },
    # ── Non-crisis periods for comparison ──
    {
        "name": "Bull 2019",
        "start": "2019-01-01",
        "end": "2019-09-30",
        "description": "Pre-COVID bull market, low volatility",
        "vix_peak_date": "2019-08-05",
        "crisis_start": "2019-08-01",
        "expected_regime": "normal",
    },
    {
        "name": "AI Rally 2023-24",
        "start": "2023-04-01",
        "end": "2024-06-30",
        "description": "Post-bear recovery, AI-driven rally",
        "vix_peak_date": "2023-10-20",
        "crisis_start": "2023-10-01",
        "expected_regime": "normal",
    },
]

# ── Trading params (same as backtest.py) ──
REFIT_DAYS = 20
HOLD_MR = 5
K_TOP = 5
Z_ENTRY_BASE = 0.8
TX_BPS = 10
INITIAL = 100_000
STOP_HARD = -0.10


def get_tx_cost(ticker):
    """Realistic cost by ticker type."""
    country = config.TICKER_COUNTRY.get(ticker, "US")
    zone = config.COUNTRY_TO_ZONE.get(country, "USD")
    if zone == "USD" and ticker not in config.ETF_TICKERS:
        return 5 / 10000
    elif zone in ("EUR", "ASIA"):
        return 15 / 10000
    else:
        return 25 / 10000


def run_crisis_backtest(db, crisis: dict) -> dict:
    """Run walk-forward backtest on a single crisis period."""
    name = crisis["name"]
    start = crisis["start"]
    end = crisis["end"]
    crisis_start = crisis["crisis_start"]
    vix_peak_date = crisis["vix_peak_date"]

    print(f"\n{'═' * 80}")
    print(f"  CRISIS: {name}")
    print(f"  {crisis['description']}")
    print(f"  Period: {start} → {end}")
    print(f"{'═' * 80}")

    # ── Load data for the full period ──
    gb = GraphBuilder(db)
    gb.load_data(start_date=start, end_date=end)

    if len(gb.prices) < 60:
        print(f"  ⚠ Insufficient data ({len(gb.prices)} rows)")
        return None

    gb.build()
    ff = FundamentalFilter(db)
    ff.compute_all()
    engine = HeatEngine(gb, ff)
    engine.solve(calibrate=True)

    tickers = gb.tickers
    N = len(tickers)
    T = len(gb.returns)
    returns = gb.returns.apply(pd.to_numeric, errors='coerce').fillna(0).values
    dates = gb.returns.index
    z = engine.z_scores

    print(f"  N={N} tickers, T={T} days, edges={int(np.sum(gb.W != 0))}")
    print(f"  s={gb.s:.3f}, α={engine.alpha:.4f}")

    # ── SPY index for benchmark ──
    spy_idx = tickers.index("SPY") if "SPY" in tickers else 0

    # ── Walk-forward trading simulation ──
    equity_mr = [INITIAL]
    equity_spy = [INITIAL]
    equity_rand = [INITIAL]
    equity_gate = [INITIAL]  # P5: MR + Regime Gate
    equity_pairs = [INITIAL] # Graph Pairs (dollar-neutral)
    equity_combo = [INITIAL] # Pairs + Gate combined
    pos_mr = []
    pos_rand = []
    pos_gate = []            # P5
    pos_pairs = []           # Pairs: list of (long_idx, short_idx, remaining, size)
    pos_combo = []           # Combined: pairs or refuge depending on regime
    n_trades_mr = 0
    n_trades_rand = 0
    n_trades_gate = 0        # P5
    n_trades_pairs = 0
    n_trades_combo = 0
    warmup = 80
    s_timeline = []  # Track s values over time for diagnostic

    # P5: Fundamental scores for quality filter
    fund_scores = ff.scores if hasattr(ff, 'scores') else {}

    for t in range(warmup, T):
        day_idx = t - warmup
        ret = returns[t] if t < len(returns) else np.zeros(N)

        # ── Step MR positions ──
        pnl_mr = 0.0
        alive = []
        for pos in pos_mr:
            idx, direction, remaining, entry_p, size, tx = pos
            if idx >= N:
                continue
            pnl_mr += direction * ret[idx] * size
            remaining -= 1
            # Hard stop
            cum_ret = direction * np.sum(returns[max(0, t - HOLD_MR):t + 1, idx])
            if cum_ret < STOP_HARD:
                pnl_mr -= tx
            elif remaining <= 0:
                pnl_mr -= tx
            else:
                alive.append((idx, direction, remaining, entry_p, size, tx))
        pos_mr = alive
        equity_mr.append(equity_mr[-1] * (1 + pnl_mr))

        # ── Step Random positions ──
        pnl_rand = 0.0
        alive_r = []
        for pos in pos_rand:
            idx, direction, remaining, entry_p, size, tx = pos
            if idx >= N:
                continue
            pnl_rand += direction * ret[idx] * size
            remaining -= 1
            if remaining <= 0:
                pnl_rand -= tx
            else:
                alive_r.append((idx, direction, remaining, entry_p, size, tx))
        pos_rand = alive_r
        equity_rand.append(equity_rand[-1] * (1 + pnl_rand))

        # SPY benchmark
        equity_spy.append(equity_spy[-1] * (1 + ret[spy_idx]))

        # ── Step Regime-Gate positions (P5) ──
        pnl_gate = 0.0
        alive_g = []
        for pos in pos_gate:
            idx, direction, remaining, entry_p, size, tx = pos
            if idx >= N:
                continue
            pnl_gate += direction * ret[idx] * size
            remaining -= 1
            cum_ret = direction * np.sum(returns[max(0, t - HOLD_MR):t + 1, idx])
            if cum_ret < STOP_HARD:
                pnl_gate -= tx
            elif remaining <= 0:
                pnl_gate -= tx
            else:
                alive_g.append((idx, direction, remaining, entry_p, size, tx))
        pos_gate = alive_g
        equity_gate.append(equity_gate[-1] * (1 + pnl_gate))

        # ── Step Pairs positions ──
        pnl_pairs = 0.0
        alive_p = []
        for pos in pos_pairs:
            long_idx, short_idx, remaining, size = pos
            if long_idx >= N or short_idx >= N:
                continue
            # P&L = long return - short return (market neutral)
            pair_ret = (ret[long_idx] - ret[short_idx]) * size
            pnl_pairs += pair_ret
            remaining -= 1
            if remaining <= 0:
                pass  # let it expire
            else:
                alive_p.append((long_idx, short_idx, remaining, size))
        pos_pairs = alive_p
        equity_pairs.append(equity_pairs[-1] * (1 + pnl_pairs))

        # ── Step Combo positions (pairs format) ──
        pnl_combo = 0.0
        alive_cb = []
        for pos in pos_combo:
            if len(pos) == 4:
                # Pair position: (long_idx, short_idx, remaining, size)
                long_idx, short_idx, remaining, size = pos
                if long_idx >= N or short_idx >= N:
                    continue
                pair_ret = (ret[long_idx] - ret[short_idx]) * size
                pnl_combo += pair_ret
                remaining -= 1
                if remaining > 0:
                    alive_cb.append((long_idx, short_idx, remaining, size))
            else:
                # Refuge position: (idx, direction, remaining, entry_p, size, tx)
                idx, direction, remaining, entry_p, size, tx = pos
                if idx >= N:
                    continue
                pnl_combo += direction * ret[idx] * size
                remaining -= 1
                if remaining <= 0:
                    pnl_combo -= tx
                else:
                    alive_cb.append((idx, direction, remaining, entry_p, size, tx))
        pos_combo = alive_cb
        equity_combo.append(equity_combo[-1] * (1 + pnl_combo))

        # ── Open trades on refit days ──
        if day_idx % REFIT_DAYS == 0 and z is not None and t < len(z):
            # Get current VIX for regime gate (no s recalibration needed)
            ref_date = dates[t] if t < len(dates) else dates[-1]
            vix_now = float(gb.vix.asof(ref_date)) if len(gb.vix) > 0 else 15.0
            if np.isnan(vix_now):
                vix_now = 15.0

            # Determine VIX-based regime
            if vix_now > 35:
                vix_mode = "REFUGE"
            elif vix_now > 25:
                vix_mode = "DEFENSIVE"
            else:
                vix_mode = "ALPHA"

            # Log for diagnostic
            s_timeline.append({
                "date": str(ref_date.date()) if hasattr(ref_date, 'date') else str(ref_date),
                "vix": round(vix_now, 1),
                "mode": vix_mode,
            })

            zt = np.nan_to_num(z[t][:N], nan=0)
            order = np.argsort(zt)

            # MR: buy cold (z < -threshold), sell hot (z > threshold)
            for idx in order[:K_TOP]:
                if zt[idx] < -Z_ENTRY_BASE:
                    tx = get_tx_cost(tickers[idx])
                    pos_mr.append((idx, +1, HOLD_MR, 1.0, 1.0, tx))
                    n_trades_mr += 1
            for idx in order[-K_TOP:]:
                if zt[idx] > Z_ENTRY_BASE:
                    tx = get_tx_cost(tickers[idx])
                    pos_mr.append((idx, -1, HOLD_MR, 1.0, 1.0, tx))
                    n_trades_mr += 1

            # Random baseline
            rng = np.random.default_rng(seed=t)
            for idx in rng.choice(N, size=min(K_TOP, N), replace=False):
                tx = get_tx_cost(tickers[idx])
                pos_rand.append((idx, +1 if rng.random() > 0.5 else -1,
                                 10, 1.0, 1.0, tx))
                n_trades_rand += 1

            # Graph Pairs: dollar-neutral long-short
            # Long the K most undervalued (z << 0), short the K most overvalued (z >> 0)
            n_pairs = min(K_TOP, N // 2)
            long_candidates = order[:n_pairs]   # most negative z
            short_candidates = order[-n_pairs:]  # most positive z
            for k in range(n_pairs):
                li = long_candidates[k]
                si = short_candidates[n_pairs - 1 - k]  # pair best long with best short
                if zt[li] < -0.5 and zt[si] > 0.5:  # both must be dislocated
                    pos_pairs.append((li, si, HOLD_MR, 0.5))  # half size per pair
                    n_trades_pairs += 1

            # P6: VIX-gated MR
            if vix_mode == "REFUGE":
                # REFUGE: close all equity, long refuge
                pos_gate = []
                for idx in range(N):
                    if tickers[idx] in REFUGE_TICKERS:
                        tx = get_tx_cost(tickers[idx])
                        pos_gate.append((idx, +1, HOLD_MR, 1.0, 0.2, tx))
                        n_trades_gate += 1
            elif vix_mode == "DEFENSIVE":
                # DEFENSIVE: only quality longs, 50% sizing, no shorts
                for idx in order[:K_TOP]:
                    if zt[idx] < -Z_ENTRY_BASE:
                        tk = tickers[idx]
                        F = fund_scores.get(tk, 0.0)
                        if F >= 0:
                            tx = get_tx_cost(tk)
                            pos_gate.append((idx, +1, HOLD_MR, 1.0, 0.5, tx))
                            n_trades_gate += 1
            else:
                # ALPHA: normal MR long/short
                for idx in order[:K_TOP]:
                    if zt[idx] < -Z_ENTRY_BASE:
                        tx = get_tx_cost(tickers[idx])
                        pos_gate.append((idx, +1, HOLD_MR, 1.0, 1.0, tx))
                        n_trades_gate += 1
                for idx in order[-K_TOP:]:
                    if zt[idx] > Z_ENTRY_BASE:
                        tx = get_tx_cost(tickers[idx])
                        pos_gate.append((idx, -1, HOLD_MR, 1.0, 1.0, tx))
                        n_trades_gate += 1

            # P6: Combined — VIX gate + Pairs trading
            if vix_mode == "REFUGE":
                # REFUGE: close ALL pairs, long refuge ETFs only
                pos_combo = []  # wipe all pairs
                for idx in range(N):
                    if tickers[idx] in REFUGE_TICKERS:
                        tx = get_tx_cost(tickers[idx])
                        pos_combo.append((idx, +1, HOLD_MR, 1.0, 0.2, tx))
                        n_trades_combo += 1
            elif vix_mode == "DEFENSIVE":
                # DEFENSIVE: pairs at 25% size (reduced risk, still market-neutral)
                for k in range(n_pairs):
                    li = long_candidates[k]
                    si = short_candidates[n_pairs - 1 - k]
                    if zt[li] < -0.5 and zt[si] > 0.5:
                        pos_combo.append((li, si, HOLD_MR, 0.25))
                        n_trades_combo += 1
            else:
                # ALPHA: full pairs trading
                for k in range(n_pairs):
                    li = long_candidates[k]
                    si = short_candidates[n_pairs - 1 - k]
                    if zt[li] < -0.5 and zt[si] > 0.5:
                        pos_combo.append((li, si, HOLD_MR, 0.5))
                        n_trades_combo += 1

    # ── Compute metrics ──
    def metrics(eq, label):
        eq = np.array(eq)
        total_ret = (eq[-1] / eq[0] - 1) * 100
        dr = np.diff(eq) / eq[:-1]
        dr = dr[np.isfinite(dr)]
        sh = np.mean(dr) / (np.std(dr) + 1e-10) * np.sqrt(252) if len(dr) > 0 else 0
        pk = np.maximum.accumulate(eq)
        dd = np.min((eq - pk) / pk) * 100
        return {"label": label, "return": total_ret, "sharpe": sh, "max_dd": dd,
                "final_equity": eq[-1]}

    m_mr = metrics(equity_mr, "MR (z-score)")
    m_spy = metrics(equity_spy, "SPY B&H")
    m_rand = metrics(equity_rand, "Random")
    m_gate = metrics(equity_gate, "MR + Regime Gate")
    m_pairs = metrics(equity_pairs, "Graph Pairs")
    m_combo = metrics(equity_combo, "Pairs + Gate")

    print(f"\n  {'Strategy':<22} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7}")
    print(f"  {'─' * 58}")
    for m, nt in [(m_mr, n_trades_mr), (m_pairs, n_trades_pairs),
                  (m_combo, n_trades_combo), (m_gate, n_trades_gate),
                  (m_spy, '-'), (m_rand, n_trades_rand)]:
        print(f"  {m['label']:<22} {m['return']:>+7.1f}% {m['sharpe']:>6.2f} {m['max_dd']:>7.1f}% {str(nt):>7}")

    # ── VIX Gate Diagnostic ──
    if s_timeline:
        vix_vals = [e["vix"] for e in s_timeline]
        modes = [e["mode"] for e in s_timeline]
        n_alpha = modes.count("ALPHA")
        n_def = modes.count("DEFENSIVE")
        n_ref = modes.count("REFUGE")
        print(f"\n  ── VIX Gate Timeline ──")
        print(f"  VIX range: [{min(vix_vals):.1f}, {max(vix_vals):.1f}] | mean: {np.mean(vix_vals):.1f}")
        print(f"  Modes: ALPHA={n_alpha} | DEFENSIVE={n_def} | REFUGE={n_ref} (of {len(s_timeline)} refits)")
        for e in s_timeline[:3] + [{"date": "...", "vix": 0, "mode": "..."}] + s_timeline[-3:]:
            if e["date"] == "...":
                print(f"  ...")
            else:
                print(f"  {e['date']}  VIX={e['vix']:>5.1f}  → {e['mode']}")

    # ── UKF s(t) anticipation analysis ──
    s_anticipation = _analyze_s_anticipation(gb, dates, crisis)

    # ── Credit spread delta anticipation ──
    credit_anticipation = _analyze_credit_anticipation(gb, dates, crisis)

    # ── Hit rate analysis ──
    hits = 0
    total = 0
    for t in range(warmup, T - 6):
        if z is None or t >= len(z):
            continue
        for i in range(N):
            zt = z[t, i]
            if abs(zt) < 1.5 or np.isnan(zt):
                continue
            ret_5d = np.nansum(returns[t + 1:t + 6, i])
            if np.isnan(ret_5d):
                continue
            if (zt > 0 and ret_5d < 0) or (zt < 0 and ret_5d > 0):
                hits += 1
            total += 1
    hit_rate = hits / max(total, 1)

    result = {
        "crisis": name,
        "start": start,
        "end": end,
        "N": N,
        "T": T,
        "s": float(gb.s),
        "alpha": float(engine.alpha),
        "mr": m_mr,
        "pairs": m_pairs,
        "combo": m_combo,
        "gate": m_gate,
        "spy": m_spy,
        "random": m_rand,
        "alpha_vs_spy": m_mr["return"] - m_spy["return"],
        "pairs_vs_spy": m_pairs["return"] - m_spy["return"],
        "combo_vs_spy": m_combo["return"] - m_spy["return"],
        "gate_vs_spy": m_gate["return"] - m_spy["return"],
        "n_trades": n_trades_mr,
        "n_trades_pairs": n_trades_pairs,
        "n_trades_combo": n_trades_combo,
        "hit_rate_5d": hit_rate,
        "s_anticipation": s_anticipation,
        "credit_anticipation": credit_anticipation,
    }

    # ── Print anticipation results ──
    print(f"\n  ── Anticipation Analysis ──")
    if s_anticipation:
        lead = s_anticipation.get("s_lead_days", "N/A")
        s_at_crisis = s_anticipation.get("s_at_crisis_start", "N/A")
        print(f"  UKF s anticipation: s dropped {lead} days before crisis")
        print(f"  s at crisis start: {s_at_crisis}")
    if credit_anticipation:
        lead = credit_anticipation.get("credit_lead_days", "N/A")
        print(f"  Credit delta lead: widened {lead} days before crisis")

    print(f"  Hit rate (5d): {hit_rate:.1%}")
    print(f"  MR α vs SPY: {result['alpha_vs_spy']:+.1f}%")
    print(f"  Pairs α vs SPY: {result['pairs_vs_spy']:+.1f}%")
    print(f"  Gate α vs SPY: {result['gate_vs_spy']:+.1f}%")

    return result


def _analyze_s_anticipation(gb, dates, crisis: dict) -> dict:
    """
    Analyze whether UKF s(t) drops BEFORE VIX spikes.
    Returns: lead time in days, s value at crisis start.
    """
    crisis_start = pd.Timestamp(crisis["crisis_start"])
    vix_peak = pd.Timestamp(crisis["vix_peak_date"])

    # s(t) is a scalar for the whole period — we need to track it over time
    # Use the heuristic s values at different points
    s_threshold = 0.65  # s below this = model detected stress
    s_vals = []
    vix_vals = []

    ret_idx = gb.returns.index
    for t in range(20, len(ret_idx) - 1, 5):
        d = ret_idx[t]
        if d > vix_peak + pd.Timedelta(days=5):
            break
        gb._calibrate_s(str(d.date()))
        s_val = gb.s

        v = gb.vix.asof(d) if len(gb.vix) > 0 else np.nan
        s_vals.append({"date": d, "s": s_val, "vix": float(v) if pd.notna(v) else None})
        if pd.notna(v):
            vix_vals.append(float(v))

    if not s_vals:
        return {}

    # Find first date s dropped below threshold before crisis
    first_drop = None
    for sv in s_vals:
        if sv["date"] < crisis_start and sv["s"] < s_threshold:
            first_drop = sv["date"]
            break

    # s at crisis start
    s_at_crisis = None
    for sv in s_vals:
        if sv["date"] >= crisis_start:
            s_at_crisis = sv["s"]
            break

    lead_days = (crisis_start - first_drop).days if first_drop else 0

    return {
        "s_lead_days": lead_days,
        "s_at_crisis_start": round(s_at_crisis, 4) if s_at_crisis else None,
        "first_drop_date": str(first_drop.date()) if first_drop else None,
        "s_timeline": [
            {"date": str(sv["date"].date()), "s": round(sv["s"], 4), "vix": sv["vix"]}
            for sv in s_vals[-20:]  # last 20 samples
        ],
    }


def _analyze_credit_anticipation(gb, dates, crisis: dict) -> dict:
    """
    Analyze whether credit spread delta widened BEFORE the drawdown.
    """
    crisis_start = pd.Timestamp(crisis["crisis_start"])

    if len(gb.credit_spread_delta) < 10:
        return {"credit_lead_days": "N/A", "reason": "insufficient credit data"}

    # Find first date credit delta > 1 std above mean before crisis
    cd = gb.credit_spread_delta
    cd_mean = cd.mean()
    cd_std = cd.std()
    threshold = cd_mean + cd_std

    first_warning = None
    for d, v in cd.items():
        if d >= crisis_start:
            break
        if v > threshold:
            if first_warning is None:
                first_warning = d

    if first_warning:
        lead_days = (crisis_start - first_warning).days
    else:
        lead_days = 0

    return {
        "credit_lead_days": lead_days,
        "first_warning_date": str(first_warning.date()) if first_warning else None,
        "credit_delta_mean": round(float(cd_mean), 6),
        "credit_delta_std": round(float(cd_std), 6),
    }


def main():
    db = DatabaseManager()

    print("=" * 80)
    print("  P4.1 — CRISIS BACKTEST")
    print("  Three crisis periods: Volmageddon, COVID, Fed Rate Hikes")
    print("=" * 80)

    all_results = []

    for crisis in CRISIS_PERIODS:
        try:
            result = run_crisis_backtest(db, crisis)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  ❌ {crisis['name']}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\n  ✗ No crises completed")
        return

    # ── Summary ──
    print(f"\n{'═' * 90}")
    print(f"  RESUMEN — CRISIS BACKTEST")
    print(f"{'═' * 90}")
    print(f"  {'Crisis':<18} {'MR':>8} {'Pairs':>8} {'Combo':>8} {'Gate':>8} {'SPY':>8} {'CombDD':>8}")
    print(f"  {'─' * 80}")

    for r in all_results:
        print(f"  {r['crisis']:<18} {r['mr']['return']:>+7.1f}% {r['pairs']['return']:>+7.1f}% "
              f"{r['combo']['return']:>+7.1f}% {r['gate']['return']:>+7.1f}% {r['spy']['return']:>+7.1f}% "
              f"{r['combo']['max_dd']:>7.1f}%")

    print(f"  {'─' * 80}")

    # ── Verdict ──
    avg_alpha = np.mean([r["alpha_vs_spy"] for r in all_results])
    avg_pairs_alpha = np.mean([r["pairs_vs_spy"] for r in all_results])
    avg_combo_alpha = np.mean([r["combo_vs_spy"] for r in all_results])
    avg_gate_alpha = np.mean([r["gate_vs_spy"] for r in all_results])
    avg_dd_mr = np.mean([r["mr"]["max_dd"] for r in all_results])
    avg_dd_pairs = np.mean([r["pairs"]["max_dd"] for r in all_results])
    avg_dd_combo = np.mean([r["combo"]["max_dd"] for r in all_results])
    avg_dd_gate = np.mean([r["gate"]["max_dd"] for r in all_results])

    print(f"\n  Average MR α vs SPY:     {avg_alpha:+.1f}%")
    print(f"  Average Pairs α vs SPY:  {avg_pairs_alpha:+.1f}%")
    print(f"  Average Combo α vs SPY:  {avg_combo_alpha:+.1f}%")
    print(f"  Average Gate α vs SPY:   {avg_gate_alpha:+.1f}%")
    print(f"  Average MR MaxDD:        {avg_dd_mr:.1f}%")
    print(f"  Average Pairs MaxDD:     {avg_dd_pairs:.1f}%")
    print(f"  Average Combo MaxDD:     {avg_dd_combo:.1f}%")
    print(f"  Average Gate MaxDD:      {avg_dd_gate:.1f}%")

    # Find the best strategy
    best_name = "Combo"
    best_alpha = avg_combo_alpha
    if avg_pairs_alpha > best_alpha:
        best_name, best_alpha = "Pairs", avg_pairs_alpha
    if avg_gate_alpha > best_alpha:
        best_name, best_alpha = "Gate", avg_gate_alpha
    print(f"\n  🏆 MEJOR ESTRATEGIA: {best_name} (α vs SPY: {best_alpha:+.1f}%, MaxDD: {avg_dd_combo:.1f}%)")

    # ── UKF Anticipation Summary ──
    print(f"\n  ── UKF s(t) Anticipation ──")
    for r in all_results:
        s = r.get("s_anticipation", {})
        lead = s.get("s_lead_days", 0)
        icon = "✅" if lead > 5 else ("🟡" if lead > 0 else "❌")
        print(f"  {icon} {r['crisis']:<20} s lead: {lead} days | "
              f"s at crisis: {s.get('s_at_crisis_start', 'N/A')}")

    # ── Credit Delta Summary ──
    print(f"\n  ── Credit Spread Delta Anticipation ──")
    for r in all_results:
        c = r.get("credit_anticipation", {})
        lead = c.get("credit_lead_days", "N/A")
        icon = "✅" if isinstance(lead, int) and lead > 5 else "🟡"
        print(f"  {icon} {r['crisis']:<20} credit lead: {lead} days")

    print(f"{'═' * 90}")

    # ── Save results ──
    save_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "s_anticipation" or k != "credit_anticipation"}
        save_results.append(sr)

    with open("crisis_backtest_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Guardado: crisis_backtest_results.json")


if __name__ == "__main__":
    main()
