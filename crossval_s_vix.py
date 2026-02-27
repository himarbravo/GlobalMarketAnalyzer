"""
CROSS-VALIDATION: s(t) vs VIX por periodo histórico
=====================================================
Descarga datos de N periodos (crisis, stress, calm, hype),
construye el grafo en cada uno, y mide Corr(s, VIX).

Hipótesis: VIX↑ → s↓ (difusión global en crisis)
"""

import numpy as np
import pandas as pd
import logging
from database_manager import DatabaseManager
from graph_builder import GraphBuilder
from config import get_all_tickers

logging.basicConfig(level=logging.WARNING)

PERIODS = [
    # (nombre, start, end, VIX_esperado, descripción)
    ("COVID crash",     "2020-01-01", "2020-06-30", "high",  "VIX 80+ → s debe bajar"),
    ("COVID recovery",  "2020-06-01", "2021-06-30", "mid",   "VIX bajando, s subiendo"),
    ("2021 bull run",   "2021-01-01", "2021-12-31", "low",   "VIX 15-20, s alto"),
    ("2022 bear mkt",   "2022-01-01", "2022-12-31", "high",  "VIX 25-35, s debe bajar"),
    ("2023 recovery",   "2023-01-01", "2023-12-31", "mid",   "VIX bajando, s subiendo"),
    ("2024 AI boom",    "2024-01-01", "2024-12-31", "low",   "VIX 12-18, s alto"),
    ("2025 current",    "2025-01-01", "2026-02-27", "low",   "VIX actual"),
]

def run_crossval():
    db = DatabaseManager()
    tickers = get_all_tickers()

    all_s = []
    all_vix = []
    period_results = []

    print("=" * 75)
    print("  CROSS-VALIDATION: s(t) vs VIX")
    print("=" * 75)

    for name, start, end, expected_vix, desc in PERIODS:
        print(f"\n{'─' * 70}")
        print(f"  📊 {name}: {start} → {end}")
        print(f"     {desc}")
        print(f"{'─' * 70}")

        try:
            gb = GraphBuilder(db)
            gb.load_data(start_date=start, end_date=end)

            if len(gb.prices) < 30:
                print(f"  ⚠ Insuficientes datos ({len(gb.prices)} filas)")
                continue

            gb.build()

            n_edges = np.sum(gb.W != 0)
            print(f"  N={gb.N}, T={len(gb.returns)}, edges={n_edges}")

            # Compute s(t) at multiple points within the period
            s_vals = []
            vix_vals = []
            dates_used = []

            ret_idx = gb.returns.index
            step = max(5, len(ret_idx) // 30)  # ~30 samples per period

            for t in range(20, len(ret_idx) - 1, step):
                d = str(ret_idx[t].date())
                gb._calibrate_s(d)
                s_val = gb.s

                # Get VIX at this date
                v = gb.vix.asof(ret_idx[t]) if len(gb.vix) > 0 else np.nan
                if pd.notna(v) and v > 0:
                    s_vals.append(s_val)
                    vix_vals.append(float(v))
                    dates_used.append(d)

            if len(s_vals) < 5:
                print(f"  ⚠ Insuficientes pares s-VIX ({len(s_vals)})")
                continue

            s_arr = np.array(s_vals)
            v_arr = np.array(vix_vals)

            corr = np.corrcoef(s_arr, v_arr)[0, 1]
            vix_mean = np.mean(v_arr)
            vix_std = np.std(v_arr)
            s_mean = np.mean(s_arr)
            s_std = np.std(s_arr)

            # Collect for global correlation
            all_s.extend(s_vals)
            all_vix.extend(vix_vals)

            result = {
                "period": name,
                "start": start,
                "end": end,
                "n_samples": len(s_vals),
                "vix_mean": vix_mean,
                "vix_std": vix_std,
                "s_mean": s_mean,
                "s_std": s_std,
                "corr": corr,
                "expected_vix": expected_vix,
            }
            period_results.append(result)

            status = "✅" if corr < -0.2 else ("🟡" if corr < 0 else "❌")
            print(f"  VIX: {vix_mean:.1f} ± {vix_std:.1f}")
            print(f"  s:   {s_mean:.3f} ± {s_std:.3f}")
            print(f"  {status} Corr(s, VIX) = {corr:.3f}  (n={len(s_vals)})")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback; traceback.print_exc()

    # ── Global correlation across ALL periods ──
    print(f"\n{'=' * 75}")
    print(f"  RESULTADO GLOBAL")
    print(f"{'=' * 75}")

    if len(all_s) > 10:
        global_corr = np.corrcoef(all_s, all_vix)[0, 1]
        print(f"\n  Total samples: {len(all_s)}")
        print(f"  VIX range: [{min(all_vix):.1f}, {max(all_vix):.1f}]")
        print(f"  s range:   [{min(all_s):.3f}, {max(all_s):.3f}]")
        print(f"\n  {'✅' if global_corr < -0.3 else '❌'} Corr(s, VIX) GLOBAL = {global_corr:.4f}")
    else:
        print("  Insuficientes datos para correlación global")

    # ── Summary table ──
    print(f"\n{'─' * 75}")
    print(f"  {'Periodo':<20} {'VIX':>6} {'s':>6} {'Corr':>7} {'n':>4} {'Status':>6}")
    print(f"{'─' * 75}")
    for r in period_results:
        status = "✅" if r["corr"] < -0.2 else ("🟡" if r["corr"] < 0 else "❌")
        print(f"  {r['period']:<20} {r['vix_mean']:>5.1f} {r['s_mean']:>5.3f} {r['corr']:>+7.3f} {r['n_samples']:>4} {status:>6}")
    print(f"{'─' * 75}")

    if len(all_s) > 10:
        global_corr = np.corrcoef(all_s, all_vix)[0, 1]
        print(f"  {'GLOBAL':<20} {'':>6} {'':>6} {global_corr:>+7.3f} {len(all_s):>4}")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    run_crossval()
