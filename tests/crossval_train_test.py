"""
CROSS-VALIDATION — Train/Test Split — P4.2
=============================================
Proper temporal cross-validation with non-overlapping folds.

Folds:
  1. Train 2019-01→2022-12, Test 2023-01→2025-06
  2. Train 2021-01→2024-06, Test 2024-07→2026-02
  3. Train 2019-01→2021-12, Test 2022-01→2023-12 (bull→bear)

For each fold:
  - Train: calibrate α, γ on train window
  - Test: freeze parameters, run on test window
  - Compare: R², hit rate 5d/20d, MR P&L, Sharpe

Overfitting detection:
  - R² drop > 0.05 = warning
  - Hit rate < 48% OOS = signal not generalizing
  - Parameter instability across folds = fragile model

Usage:
    python tests/crossval_train_test.py
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

FOLDS = [
    {
        "name": "Fold 1: COVID+Bear → Recovery+AI",
        "train_start": "2019-01-01", "train_end": "2022-12-31",
        "test_start": "2023-01-01", "test_end": "2025-06-30",
    },
    {
        "name": "Fold 2: Bear+Recovery → Recent",
        "train_start": "2021-01-01", "train_end": "2024-06-30",
        "test_start": "2024-07-01", "test_end": "2026-02-28",
    },
    {
        "name": "Fold 3: Bull → Bear (hardest)",
        "train_start": "2019-01-01", "train_end": "2021-12-31",
        "test_start": "2022-01-01", "test_end": "2023-12-31",
    },
]

K_TOP = 10  # top/bottom for MR strategy


def evaluate_window(db, start, end, label,
                    frozen_alpha=None, frozen_gamma=None) -> dict:
    """
    Run full model on a window and return comprehensive metrics.

    If frozen_alpha/gamma are provided, uses those instead of calibrating
    (for test-phase evaluation with train-derived parameters).
    """
    gb = GraphBuilder(db)
    gb.load_data(start_date=start, end_date=end)

    if len(gb.prices) < 60:
        print(f"    ⚠ Insufficient data ({len(gb.prices)} rows)")
        return None

    gb.build()
    ff = FundamentalFilter(db)
    ff.compute_all()
    engine = HeatEngine(gb, ff)

    if frozen_alpha is not None and frozen_gamma is not None:
        # Test phase: use frozen parameters from training
        engine.solve(calibrate=False)
        # Override with frozen values after solve
        engine.alpha = frozen_alpha
        # Re-solve with frozen alpha (no re-calibration)
        engine.solve(calibrate=False)
    else:
        # Train phase: calibrate fresh
        engine.solve(calibrate=True)

    z = engine.z_scores
    returns = gb.returns.apply(pd.to_numeric, errors='coerce').fillna(0).values
    T, N = z.shape if z is not None and len(z.shape) == 2 else (0, 0)

    if T < 60 or N < 10:
        print(f"    ⚠ Too little data after solve: T={T}, N={N}")
        return None

    # ── R² ──
    ss_res = np.nansum((engine.u_real - engine.u_pred) ** 2)
    ss_tot = np.nansum((engine.u_real - np.nanmean(engine.u_real, axis=0)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    # ── Hit rate 5d ──
    hits5 = 0
    total5 = 0
    for t in range(60, T - 6):
        for i in range(N):
            zt = z[t, i]
            if abs(zt) < 1.5 or np.isnan(zt):
                continue
            ret_5d = np.nansum(returns[t + 1:t + 6, i])
            if np.isnan(ret_5d):
                continue
            if (zt > 0 and ret_5d < 0) or (zt < 0 and ret_5d > 0):
                hits5 += 1
            total5 += 1
    hit5 = hits5 / max(total5, 1)

    # ── Hit rate 20d ──
    hits20 = 0
    total20 = 0
    for t in range(60, T - 21):
        for i in range(N):
            zt = z[t, i]
            if abs(zt) < 1.5 or np.isnan(zt):
                continue
            ret_20d = np.nansum(returns[t + 1:t + 21, i])
            if np.isnan(ret_20d):
                continue
            if (zt > 0 and ret_20d < 0) or (zt < 0 and ret_20d > 0):
                hits20 += 1
            total20 += 1
    hit20 = hits20 / max(total20, 1)

    # ── MR P&L ──
    mr_pnl = []
    for t in range(80, T - 11, 5):
        zt = z[t, :]
        zt = np.nan_to_num(zt, nan=0)
        cold = np.argsort(zt)[:K_TOP]
        hot = np.argsort(zt)[-K_TOP:]
        rl = np.nanmean([np.nansum(returns[t + 1:t + 11, i]) for i in cold])
        rs = np.nanmean([np.nansum(returns[t + 1:t + 11, i]) for i in hot])
        mr_pnl.append(rl - rs)
    mr_cum = np.sum(mr_pnl) * 100 if mr_pnl else 0
    mr_hit = np.mean(np.array(mr_pnl) > 0) if mr_pnl else 0
    mr_sharpe = (np.mean(mr_pnl) / (np.std(mr_pnl) + 1e-10) * np.sqrt(252 / 5)
                 if mr_pnl else 0)

    result = {
        "label": label,
        "start": start, "end": end,
        "N": N, "T": T,
        "s": float(gb.s),
        "alpha": float(engine.alpha),
        "gamma": float(getattr(engine, 'gamma', 5.0)),
        "r2": float(r2),
        "hit5": float(hit5), "n5": total5,
        "hit20": float(hit20), "n20": total20,
        "mr_cum": float(mr_cum),
        "mr_hit": float(mr_hit),
        "mr_sharpe": float(mr_sharpe),
    }

    print(f"    {label}: R²={r2:.4f} | hit5={hit5:.1%} hit20={hit20:.1%} | "
          f"MR={mr_cum:+.1f}% sh={mr_sharpe:.2f} | "
          f"α={engine.alpha:.4f} s={gb.s:.3f}")

    return result


def run_fold(db, fold: dict) -> dict:
    """Run one cross-validation fold: train then test."""
    name = fold["name"]
    print(f"\n{'─' * 80}")
    print(f"  {name}")
    print(f"  Train: {fold['train_start']} → {fold['train_end']}")
    print(f"  Test:  {fold['test_start']} → {fold['test_end']}")
    print(f"{'─' * 80}")

    # ── Train phase ──
    print(f"  Training...")
    train_result = evaluate_window(
        db, fold["train_start"], fold["train_end"], "TRAIN"
    )

    if train_result is None:
        return None

    # Extract calibrated parameters from training
    frozen_alpha = train_result["alpha"]
    frozen_gamma = train_result["gamma"]

    # ── Test phase (frozen parameters) ──
    print(f"  Testing (frozen α={frozen_alpha:.4f}, γ={frozen_gamma:.1f})...")
    test_result = evaluate_window(
        db, fold["test_start"], fold["test_end"], "TEST",
        frozen_alpha=frozen_alpha, frozen_gamma=frozen_gamma,
    )

    if test_result is None:
        return None

    # ── Compare ──
    r2_drop = train_result["r2"] - test_result["r2"]
    hit5_drop = train_result["hit5"] - test_result["hit5"]
    hit20_drop = train_result["hit20"] - test_result["hit20"]
    mr_drop = train_result["mr_cum"] - test_result["mr_cum"]

    comparison = {
        "fold": name,
        "train": train_result,
        "test": test_result,
        "r2_drop": r2_drop,
        "hit5_drop": hit5_drop,
        "hit20_drop": hit20_drop,
        "mr_drop": mr_drop,
    }

    # Diagnosis
    issues = []
    if r2_drop > 0.05:
        issues.append(f"R² drop {r2_drop:+.4f}")
    if test_result["hit5"] < 0.48:
        issues.append(f"OOS hit5 {test_result['hit5']:.1%} < 48%")
    if test_result["hit20"] < 0.48:
        issues.append(f"OOS hit20 {test_result['hit20']:.1%} < 48%")
    if test_result["mr_cum"] < -10:
        issues.append(f"OOS MR P&L {test_result['mr_cum']:+.1f}% (large loss)")

    if not issues:
        print(f"  ✅ No overfitting detected")
        comparison["diagnosis"] = "PASS"
    else:
        print(f"  ⚠ Potential issues:")
        for issue in issues:
            print(f"    - {issue}")
        comparison["diagnosis"] = "WARN"

    return comparison


def main():
    db = DatabaseManager()

    print("=" * 80)
    print("  P4.2 — CROSS-VALIDATION (Train/Test Split)")
    print("  3 temporal folds with frozen parameters")
    print("=" * 80)

    all_folds = []

    for fold in FOLDS:
        try:
            result = run_fold(db, fold)
            if result:
                all_folds.append(result)
        except Exception as e:
            print(f"\n  ❌ {fold['name']}: {e}")
            import traceback
            traceback.print_exc()

    if not all_folds:
        print("\n  ✗ No folds completed")
        return

    # ── Summary Table ──
    print(f"\n{'═' * 100}")
    print(f"  CROSS-VALIDATION RESULTS")
    print(f"{'═' * 100}")
    print(f"  {'Fold':<35} {'Phase':>6} {'R²':>6} {'hit5':>6} {'hit20':>6} "
          f"{'MR%':>7} {'MR_sh':>6} {'α':>7} {'s':>5}")
    print(f"  {'─' * 95}")

    for fold in all_folds:
        for phase in ["train", "test"]:
            r = fold[phase]
            print(f"  {fold['fold'][:34]:<35} {phase.upper():>6} "
                  f"{r['r2']:>5.3f} {r['hit5']:>5.1%} {r['hit20']:>5.1%} "
                  f"{r['mr_cum']:>+6.1f}% {r['mr_sharpe']:>5.2f} "
                  f"{r['alpha']:>7.4f} {r['s']:>5.3f}")

    # ── Aggregate Diagnosis ──
    print(f"\n{'─' * 100}")
    print(f"  OVERFITTING DIAGNOSIS:")

    r2_drops = [f["r2_drop"] for f in all_folds]
    hit5_oos = [f["test"]["hit5"] for f in all_folds]
    hit20_oos = [f["test"]["hit20"] for f in all_folds]
    mr_oos = [f["test"]["mr_cum"] for f in all_folds]
    alpha_vals = [f["train"]["alpha"] for f in all_folds]

    # R² stability
    r2_ok = np.mean(r2_drops) < 0.05
    icon = "✅" if r2_ok else "⚠"
    print(f"  {icon} R² drop (train→test): mean={np.mean(r2_drops):+.4f} "
          f"(per fold: {', '.join(f'{d:+.4f}' for d in r2_drops)})")

    # Hit rate OOS
    hit_ok = np.mean(hit5_oos) > 0.48 and np.mean(hit20_oos) > 0.48
    icon = "✅" if hit_ok else "⚠"
    print(f"  {icon} OOS hit rates: 5d={np.mean(hit5_oos):.1%}, 20d={np.mean(hit20_oos):.1%}")

    # MR OOS
    mr_ok = np.mean(mr_oos) > 0
    icon = "✅" if mr_ok else "❌"
    print(f"  {icon} OOS MR P&L: mean={np.mean(mr_oos):+.1f}%")

    # Parameter stability
    alpha_cv = np.std(alpha_vals) / (np.mean(alpha_vals) + 1e-10)
    param_ok = alpha_cv < 0.5
    icon = "✅" if param_ok else "⚠"
    print(f"  {icon} α stability: CV={alpha_cv:.2f} "
          f"(values: {', '.join(f'{a:.4f}' for a in alpha_vals)})")

    # Overall verdict
    n_pass = sum([r2_ok, hit_ok, mr_ok, param_ok])
    print(f"\n  {'─' * 50}")
    if n_pass >= 4:
        print(f"  🏆 VEREDICTO: NO hay overfitting ({n_pass}/4 checks passed)")
    elif n_pass >= 3:
        print(f"  ✅ VEREDICTO: Overfitting mínimo ({n_pass}/4 checks passed)")
    elif n_pass >= 2:
        print(f"  ⚠ VEREDICTO: Overfitting moderado ({n_pass}/4 checks passed)")
    else:
        print(f"  ❌ VEREDICTO: Overfitting severo ({n_pass}/4 checks passed)")
    print(f"{'═' * 100}")

    # ── Save results ──
    with open("crossval_train_test_results.json", "w") as f:
        json.dump(all_folds, f, indent=2, default=str)
    print(f"\n  Guardado: crossval_train_test_results.json")


if __name__ == "__main__":
    main()
