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
    pos_mr = []
    pos_rand = []
    pos_gate = []            # P5
    n_trades_mr = 0
    n_trades_rand = 0
    n_trades_gate = 0        # P5
    warmup = 80

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

        # ── Open trades on refit days ──
        if day_idx % REFIT_DAYS == 0 and z is not None and t < len(z):
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

            # P5: Regime-gated MR (s + ds/dt)
            s_val = gb.s
            ds_dt = getattr(gb, 'ds_dt', 0.0)
            refuge_sig = getattr(engine, 'refuge_signal', 0.0)

            if s_val < 0.40 or refuge_sig > 0.5 or ds_dt < -0.05:
                # REFUGE: close all equity, long refuge
                pos_gate = []
                for idx in range(N):
                    if tickers[idx] in REFUGE_TICKERS:
                        tx = get_tx_cost(tickers[idx])
                        pos_gate.append((idx, +1, HOLD_MR, 1.0, 0.2, tx))
                        n_trades_gate += 1
            elif s_val < 0.70 or ds_dt < -0.02:
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

    print(f"\n  {'Strategy':<20} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"  {'─' * 45}")
    for m in [m_mr, m_gate, m_spy, m_rand]:
        print(f"  {m['label']:<20} {m['return']:>+7.1f}% {m['sharpe']:>6.2f} {m['max_dd']:>7.1f}%")

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
        "gate": m_gate,
        "spy": m_spy,
        "random": m_rand,
        "alpha_vs_spy": m_mr["return"] - m_spy["return"],
        "gate_vs_spy": m_gate["return"] - m_spy["return"],
        "gate_vs_mr": m_gate["return"] - m_mr["return"],
        "n_trades": n_trades_mr,
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
    print(f"  Gate α vs SPY: {result['gate_vs_spy']:+.1f}%")
    print(f"  Gate improvement over MR: {result['gate_vs_mr']:+.1f}%")

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
    print(f"  {'Crisis':<20} {'MR Ret':>8} {'Gate Ret':>9} {'SPY Ret':>8} {'Gate α':>8} {'MR DD':>8} {'Gate DD':>8} {'Hit5d':>6} {'s_lead':>7}")
    print(f"  {'─' * 100}")

    for r in all_results:
        s_lead = r["s_anticipation"].get("s_lead_days", "N/A") if r["s_anticipation"] else "N/A"
        print(f"  {r['crisis']:<20} {r['mr']['return']:>+7.1f}% {r['gate']['return']:>+8.1f}% {r['spy']['return']:>+7.1f}% "
              f"{r['gate_vs_spy']:>+7.1f}% {r['mr']['max_dd']:>7.1f}% {r['gate']['max_dd']:>7.1f}% "
              f"{r['hit_rate_5d']:>5.1%} {s_lead:>7}")

    print(f"  {'─' * 100}")

    # ── Verdict ──
    avg_alpha = np.mean([r["alpha_vs_spy"] for r in all_results])
    avg_gate_alpha = np.mean([r["gate_vs_spy"] for r in all_results])
    avg_gate_improve = np.mean([r["gate_vs_mr"] for r in all_results])
    avg_hit = np.mean([r["hit_rate_5d"] for r in all_results])
    avg_dd_mr = np.mean([r["mr"]["max_dd"] for r in all_results])
    avg_dd_gate = np.mean([r["gate"]["max_dd"] for r in all_results])

    print(f"\n  Average MR α vs SPY:      {avg_alpha:+.1f}%")
    print(f"  Average Gate α vs SPY:    {avg_gate_alpha:+.1f}%")
    print(f"  Average Gate improvement: {avg_gate_improve:+.1f}% over MR")
    print(f"  Average hit rate 5d:      {avg_hit:.1%}")
    print(f"  Average MR MaxDD:         {avg_dd_mr:.1f}%")
    print(f"  Average Gate MaxDD:       {avg_dd_gate:.1f}%")

    if avg_gate_alpha > 0:
        print(f"  🏆 VEREDICTO: Regime Gate genera α positivo vs SPY ({avg_gate_alpha:+.1f}%)")
    elif avg_gate_alpha > avg_alpha:
        print(f"  ✅ VEREDICTO: Regime Gate mejora sobre MR puro ({avg_gate_improve:+.1f}%)")
    elif avg_dd_gate > avg_dd_mr:  # less negative = better
        print(f"  ⚠ VEREDICTO: Regime Gate reduce drawdown pero no genera α")
    else:
        print(f"  ❌ VEREDICTO: Regime Gate no mejora significativamente")

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
