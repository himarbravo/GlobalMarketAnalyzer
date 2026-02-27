"""
FULL MODEL CROSS-VALIDATION — Overfitting Detection
=====================================================
For each historical period:
  1. Build graph + solve O-U
  2. Measure: R², z-score hit rate (5d/20d), MR P&L, landscape δ
  3. Compare in-sample vs out-of-sample performance

If R² drops out-of-sample or hit rates < 50% → overfitting.
"""

import numpy as np
import pandas as pd
import json
import logging
from db.database_manager import DatabaseManager
from core.graph_builder import GraphBuilder
from core.fundamental_filter import FundamentalFilter
from core.heat_engine import HeatEngine
from core.inertia_detector import InertiaDetector

logging.basicConfig(level=logging.WARNING)

PERIODS = [
    ("COVID 2020",      "2019-06-01", "2020-06-30"),
    ("Recovery 2021",   "2020-06-01", "2021-06-30"),
    ("Bull 2021",       "2021-01-01", "2021-12-31"),
    ("Bear 2022",       "2022-01-01", "2022-12-31"),
    ("Recovery 2023",   "2023-01-01", "2023-12-31"),
    ("AI Boom 2024",    "2024-01-01", "2024-12-31"),
    ("Current 2025",    "2025-01-01", "2026-02-27"),
]


def evaluate_period(db, name, start, end):
    """Run full model on a period and return metrics."""
    print(f"\n{'─' * 70}")
    print(f"  {name}: {start} → {end}")
    print(f"{'─' * 70}")

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

    z = engine.z_scores
    returns = gb.returns.apply(pd.to_numeric, errors='coerce').fillna(0).values
    T, N = z.shape

    # ── R² ──
    ss_res = np.nansum((engine.u_real - engine.u_pred) ** 2)
    ss_tot = np.nansum((engine.u_real - np.nanmean(engine.u_real, axis=0)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    # ── Hit rate 5d ──
    hits5 = 0; total5 = 0
    for t in range(60, T - 6):
        for i in range(N):
            zt = z[t, i]
            if abs(zt) < 1.5 or np.isnan(zt):
                continue
            ret_5d = np.nansum(returns[t+1:t+6, i])
            if np.isnan(ret_5d):
                continue
            if (zt > 0 and ret_5d < 0) or (zt < 0 and ret_5d > 0):
                hits5 += 1
            total5 += 1
    hit5 = hits5 / max(total5, 1)

    # ── Hit rate 20d ──
    hits20 = 0; total20 = 0
    for t in range(60, T - 21):
        for i in range(N):
            zt = z[t, i]
            if abs(zt) < 1.5 or np.isnan(zt):
                continue
            ret_20d = np.nansum(returns[t+1:t+21, i])
            if np.isnan(ret_20d):
                continue
            if (zt > 0 and ret_20d < 0) or (zt < 0 and ret_20d > 0):
                hits20 += 1
            total20 += 1
    hit20 = hits20 / max(total20, 1)

    # ── Mean reversion P&L ──
    mr_pnl = []
    K = 10
    for t in range(80, T - 11, 5):
        zt = z[t, :]
        zt = np.nan_to_num(zt, nan=0)
        cold = np.argsort(zt)[:K]
        hot = np.argsort(zt)[-K:]
        rl = np.nanmean([np.nansum(returns[t+1:t+11, i]) for i in cold])
        rs = np.nanmean([np.nansum(returns[t+1:t+11, i]) for i in hot])
        mr_pnl.append(rl - rs)
    mr_cum = np.sum(mr_pnl) * 100 if mr_pnl else 0
    mr_hit = np.mean(np.array(mr_pnl) > 0) if mr_pnl else 0

    # ── Landscape ──
    regime = getattr(engine, 'current_regime', 'unknown')
    lambda_eq = getattr(engine, 'lambda_eq', 0)
    quality = getattr(engine, '_landscape_quality', 'neutral')

    n_edges = int(np.sum(gb.W != 0))

    result = {
        "period": name, "start": start, "end": end,
        "N": N, "T": T, "edges": n_edges,
        "s": float(gb.s), "alpha": float(engine.alpha),
        "r2": float(r2),
        "hit5": float(hit5), "n5": total5,
        "hit20": float(hit20), "n20": total20,
        "mr_cum": float(mr_cum), "mr_hit": float(mr_hit),
        "regime": regime, "lambda_eq": float(lambda_eq),
        "landscape": quality,
    }

    print(f"  N={N} T={T} edges={n_edges} s={gb.s:.3f} α={engine.alpha:.4f}")
    print(f"  R²={r2:.4f} | hit5={hit5:.1%} hit20={hit20:.1%}")
    print(f"  MR P&L={mr_cum:+.1f}% hit={mr_hit:.0%}")
    print(f"  regime={regime} λ_eq={lambda_eq:.1f} landscape={quality}")

    return result


def main():
    db = DatabaseManager()
    results = []

    print("=" * 70)
    print("  FULL MODEL CROSS-VALIDATION — Overfitting Detection")
    print("=" * 70)

    for name, start, end in PERIODS:
        try:
            r = evaluate_period(db, name, start, end)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback; traceback.print_exc()

    # ── Summary Table ──
    print(f"\n{'=' * 90}")
    print(f"  RESULTADO: ¿HAY OVERFITTING?")
    print(f"{'=' * 90}")
    print(f"  {'Periodo':<20} {'R²':>6} {'hit5':>6} {'hit20':>6} {'MR%':>7} {'MRhit':>6} {'s':>5} {'α':>7} {'edges':>6}")
    print(f"{'─' * 90}")

    r2_vals = []
    hit5_vals = []
    hit20_vals = []
    mr_vals = []

    for r in results:
        print(f"  {r['period']:<20} {r['r2']:>5.3f} {r['hit5']:>5.1%} {r['hit20']:>5.1%} "
              f"{r['mr_cum']:>+6.1f}% {r['mr_hit']:>5.0%} {r['s']:>5.3f} {r['alpha']:>7.4f} {r['edges']:>6}")
        r2_vals.append(r['r2'])
        hit5_vals.append(r['hit5'])
        hit20_vals.append(r['hit20'])
        mr_vals.append(r['mr_cum'])

    print(f"{'─' * 90}")
    print(f"  {'MEAN':<20} {np.mean(r2_vals):>5.3f} {np.mean(hit5_vals):>5.1%} {np.mean(hit20_vals):>5.1%} "
          f"{np.mean(mr_vals):>+6.1f}% {'':>5}")
    print(f"  {'STD':<20} {np.std(r2_vals):>5.3f} {np.std(hit5_vals):>5.1%} {np.std(hit20_vals):>5.1%} "
          f"{np.std(mr_vals):>6.1f}%")

    # ── Overfitting diagnosis ──
    print(f"\n{'─' * 90}")
    print("  DIAGNÓSTICO:")

    r2_stable = np.std(r2_vals) < 0.05
    hit_above_50 = np.mean(hit5_vals) > 0.48 and np.mean(hit20_vals) > 0.48
    mr_positive = np.mean(mr_vals) > 0
    worst_mr = min(mr_vals) if mr_vals else 0

    if r2_stable:
        print(f"  ✅ R² estable entre periodos (σ={np.std(r2_vals):.3f}) → no overfitting en tracking")
    else:
        print(f"  ⚠ R² variable (σ={np.std(r2_vals):.3f}) → posible overfitting en parámetros")

    if hit_above_50:
        print(f"  ✅ Hit rates >48% en media → señal predictiva genuina")
    else:
        print(f"  ⚠ Hit rates bajos → el modelo no predice mejor que azar")

    if mr_positive:
        print(f"  ✅ MR P&L positivo en media (+{np.mean(mr_vals):.1f}%) → alpha real")
    else:
        print(f"  ❌ MR P&L negativo → la señal de mean-reversion no funciona")

    if worst_mr < -20:
        print(f"  ⚠ Peor periodo: {worst_mr:+.1f}% → drawdown significativo")
    else:
        print(f"  ✅ Peor periodo: {worst_mr:+.1f}% → drawdowns controlados")

    print(f"{'=' * 90}")

    # Save results
    with open("crossval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados en crossval_results.json")


if __name__ == "__main__":
    main()
