"""
TEST SUITE — GlobalMarketAnalyzer (Water-Landscape Model)
============================================================
3 categorías:
  A. Correcta implementación matemática (12 + 3 landscape tests)
  B. Correcta importación de datos
  C. Capacidad y mejora del modelo

python historical_tests.py
"""

import numpy as np
import pandas as pd
import logging
from scipy.linalg import eigh

from db.database_manager import DatabaseManager
from core.graph_builder import GraphBuilder
from core.fundamental_filter import FundamentalFilter
from core.heat_engine import HeatEngine
from core.inertia_detector import InertiaDetector

logging.basicConfig(level=logging.WARNING)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "🟡 WARN"
results = []

def record(category, name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append({"cat": category, "name": name, "status": status, "detail": detail})
    print(f"  {status}  {name}" + (f"  — {detail}" if detail else ""))


def setup():
    print("⏳ Pipeline...")
    db = DatabaseManager()
    ff = FundamentalFilter(db)
    ff.compute_all()
    gb = GraphBuilder(db)
    gb.load_data()
    gb.build()
    engine = HeatEngine(gb, ff)
    engine.solve(calibrate=True)
    detector = InertiaDetector(engine)
    detector.analyze()
    print(f"   OK: N={gb.N}, T={len(gb.returns)}, s={gb.s:.3f}, α={engine.alpha:.4f}")
    print(f"   Aristas: {np.sum(gb.W > 0)} pos + {np.sum(gb.W < 0)} neg\n")
    return db, ff, gb, engine, detector


# ═══════════════════════════════════════════════════════════════════════════
#  A. CORRECTA IMPLEMENTACIÓN MATEMÁTICA
# ═══════════════════════════════════════════════════════════════════════════

def test_math(gb, engine, ff):
    CAT = "A.MATH"
    print("=" * 70)
    print("  A. CORRECTA IMPLEMENTACIÓN MATEMÁTICA")
    print("=" * 70)

    # A1. Laplaciano: L = D - W, simétrico, semidefinido positivo
    W = gb.W
    D = np.diag(np.sum(np.abs(W), axis=1))
    L = D - W
    L_sym = np.allclose(L, L.T, atol=1e-10)
    record(CAT, "A1. Laplaciano simétrico (L = Lᵀ)", L_sym,
           f"max|L-Lᵀ|={np.max(np.abs(L - L.T)):.2e}")

    # A2. Eigenvalues ≥ 0 (semidefinido positivo)
    evals = gb.eigenvalues
    all_nonneg = np.all(evals >= -1e-10)
    record(CAT, "A2. Eigenvalues λ ≥ 0", all_nonneg,
           f"min={evals[0]:.6f}, max={evals[-1]:.6f}")

    # A3. λ₁ ≈ 0 (modo constante = conservación de dinero)
    lambda1_zero = abs(evals[0]) < 1e-6
    record(CAT, "A3. λ₁ ≈ 0 (conservación de dinero)", lambda1_zero,
           f"λ₁={evals[0]:.2e}")

    # A4. Eigenvectors ortonormales: ΦᵀΦ = I
    Phi = gb.eigenvectors
    PhiTPhi = Phi.T @ Phi
    ortho = np.allclose(PhiTPhi, np.eye(gb.N), atol=1e-8)
    record(CAT, "A4. Eigenvectors ortonormales (ΦᵀΦ=I)", ortho,
           f"max|ΦᵀΦ-I|={np.max(np.abs(PhiTPhi - np.eye(gb.N))):.2e}")

    # A5. L^s = Φ · diag(λˢ) · Φᵀ
    lam_s = np.power(evals, gb.s)
    lam_s[0] = 0.0
    L_s_check = Phi @ np.diag(lam_s) @ Phi.T
    L_s_match = np.allclose(gb.fractional_laplacian, L_s_check, atol=1e-8)
    record(CAT, "A5. L^s = Φ·diag(λˢ)·Φᵀ", L_s_match,
           f"max|diff|={np.max(np.abs(gb.fractional_laplacian - L_s_check)):.2e}")

    # A6. O-U one-step: m_k(t+1) = m_k(t)·e^{-μ} + f_k/μ·(1-e^{-μ})
    m = engine.gb.u.astype(np.float64)
    m = np.nan_to_num(m, nan=0.0)
    f_vec = np.nan_to_num(ff.get_source_vector(gb.tickers))
    f_k = Phi.T @ f_vec
    m_k = m @ Phi
    alpha = engine.alpha
    mu = alpha * lam_s
    decay = np.where(mu > 1e-10, np.exp(-mu), 1.0)
    eq = np.where(mu > 1e-10, f_k / np.maximum(mu, 1e-10) * (1 - decay), f_k)
    t = 50
    pred_51 = m_k[t] * decay + eq
    real_51 = m_k[t + 1]
    err = np.max(np.abs(pred_51 - real_51))
    record(CAT, "A6. O-U one-step: m(t+1)=m(t)e^{-μ}+f/μ(1-e^{-μ})", err < 0.1,
           f"max|ε|={err:.4f} at t={t}")

    # A7. W²/W³ sign propagation
    W_sq = W @ W
    np.fill_diagonal(W_sq, 0)
    n_neg = np.sum(W_sq < 0)
    record(CAT, "A7. W² tiene aristas negativas (anti-corr)", n_neg > 0,
           f"n_neg={n_neg}")

    # A8. s(t) ∈ [s_min, s_max]
    s_ok = 0.15 <= gb.s <= 1.0
    record(CAT, "A8. s(t) ∈ [0.15, 1.0]", s_ok, f"s={gb.s:.4f}")

    # A9. α calibrado en rango
    a_ok = 0.001 <= engine.alpha <= 0.5
    record(CAT, "A9. α ∈ [0.001, 0.5]", a_ok, f"α={engine.alpha:.4f}")

    # A10. z-scores: media ≈ 0, std ≈ 1
    z = engine.z_scores
    z_flat = z[~np.isnan(z)]
    z_mean = np.mean(z_flat)
    z_std = np.std(z_flat)
    z_ok = abs(z_mean) < 0.5 and 0.3 < z_std < 3.0
    record(CAT, "A10. z-scores: |μ|<0.5, σ∈[0.3,3]", z_ok,
           f"μ={z_mean:.3f}, σ={z_std:.3f}")

    # A11. Residuos no todos cero
    res_nonzero = np.mean(np.abs(engine.residuals)) > 1e-6
    record(CAT, "A11. Residuos no degenerados (|ε|>0)", res_nonzero,
           f"mean|ε|={np.mean(np.abs(engine.residuals)):.6f}")

    # A12. Source vector: value_creators > speculative
    vc_scores = [ff.scores[t] for t in ff.scores if ff.classifications.get(t) == "value_creator" and not np.isnan(ff.scores[t])]
    sp_scores = [ff.scores[t] for t in ff.scores if ff.classifications.get(t) == "speculative" and not np.isnan(ff.scores[t])]
    f_ok = (np.mean(vc_scores) > 0 if vc_scores else False) and (np.mean(sp_scores) <= np.mean(vc_scores) if sp_scores else True)
    record(CAT, "A12. F(value_creator) > F(speculative)", f_ok,
           f"vc_mean={np.mean(vc_scores):.4f}, sp_mean={np.mean(sp_scores):.4f}" if sp_scores else "no speculative")

    # A13. LANDSCAPE: λ = m/K terrain exists
    has_landscape = hasattr(engine, 'lambda_field') and engine.lambda_field is not None
    landscape_quality = getattr(engine, '_landscape_quality', 'neutral')
    record(CAT, "A13. Landscape K(t) terrain built", has_landscape and landscape_quality == "real",
           f"quality={landscape_quality}")

    # A14. LANDSCAPE: L_K capital-weighted Laplacian is valid
    if hasattr(engine, 'L_K') and isinstance(engine.L_K, np.ndarray):
        lk_finite = np.all(np.isfinite(engine.L_K))
        lk_shape = engine.L_K.shape == (gb.N, gb.N)
        record(CAT, "A14. L_K finite and correct shape", lk_finite and lk_shape,
               f"shape={engine.L_K.shape}, finite={lk_finite}")
    else:
        record(CAT, "A14. L_K exists", False, "not found")

    # A15. LANDSCAPE: regime classification works
    regime = getattr(engine, 'current_regime', None)
    lambda_eq = getattr(engine, 'lambda_eq', None)
    record(CAT, "A15. Regime classified with λ_eq", regime is not None and lambda_eq is not None,
           f"regime={regime}, λ_eq={lambda_eq}")

    # A16. Grafo tiene aristas (difusión activa)
    n_pos = np.sum(W > 0)
    n_neg_w = np.sum(W < 0)
    record(CAT, "A16. Grafo tiene >100 aristas", (n_pos + n_neg_w) > 100,
           f"{n_pos} pos + {n_neg_w} neg = {n_pos + n_neg_w}")


# ═══════════════════════════════════════════════════════════════════════════
#  B. CORRECTA IMPORTACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

def test_data(db, gb, ff):
    CAT = "B.DATA"
    print("\n" + "=" * 70)
    print("  B. CORRECTA IMPORTACIÓN DE DATOS")
    print("=" * 70)

    # B1. Precios: some NaN ok (not all tickers have 2y history)
    prices = gb.prices
    null_pct = prices.isnull().mean().mean() * 100
    record(CAT, "B1. Precios close: <40% NaN", null_pct < 40,
           f"{null_pct:.1f}% NaN")

    # B2. vol_20d
    vol_pct = gb.vol_20d.isnull().mean().mean() * 100 if not gb.vol_20d.empty else 100
    record(CAT, "B2. vol_20d poblado (<40% NaN)", vol_pct < 40,
           f"{vol_pct:.1f}% NaN")

    # B3. Volume
    vol_v = gb.volume.isnull().mean().mean() * 100 if not gb.volume.empty else 100
    record(CAT, "B3. Volume poblado (<40% NaN)", vol_v < 40,
           f"{vol_v:.1f}% NaN")

    # B4. Macro VIX
    record(CAT, "B4. VIX: >100 puntos", len(gb.vix) > 100,
           f"{len(gb.vix)} puntos")

    # B5. DXY
    record(CAT, "B5. DXY cargado (>100 puntos)", len(gb.dxy) > 100,
           f"{len(gb.dxy)} puntos" + (f", last={gb.dxy.iloc[-1]:.1f}" if len(gb.dxy) > 0 else ""))

    # B6. Copper
    record(CAT, "B6. Copper cargado (>100 puntos)", len(gb.copper) > 100,
           f"{len(gb.copper)} puntos")

    # B7. Oil
    record(CAT, "B7. Oil cargado (>100 puntos)", len(gb.oil) > 100,
           f"{len(gb.oil)} puntos")

    # B8. Yield spread
    record(CAT, "B8. Yield spread poblado (>100 pts)", len(gb.yield_spread) > 100,
           f"{len(gb.yield_spread)} puntos")

    # B9. Sectores dinámicos
    n_sectors = len(gb.sectors)
    n_mapped = len(gb.sector_map)
    record(CAT, "B9. Sectores dinámicos (>10 sectores)", n_sectors > 10,
           f"{n_sectors} sectores, {n_mapped} tickers mapeados")

    # B10. Fundamentals
    n_scored = sum(1 for v in ff.scores.values() if not np.isnan(v))
    record(CAT, "B10. Fundamentals: >50 tickers scored", n_scored > 50,
           f"{n_scored} scored")

    # B11. Consistencia temporal
    p_dates = len(prices)
    r_dates = len(gb.returns)
    record(CAT, "B11. Prices y Returns misma longitud", abs(p_dates - r_dates) < 5,
           f"prices={p_dates}, returns={r_dates}")

    # B12. Indicadores spot-check
    r = db.client.table("prices").select("rsi_14,bb_width,atr_14").eq("ticker", "AAPL").order("date", desc=True).limit(1).execute()
    if r.data:
        row = r.data[0]
        has_rsi = row.get("rsi_14") is not None
        has_bb = row.get("bb_width") is not None
        has_atr = row.get("atr_14") is not None
        record(CAT, "B12. AAPL: RSI/BB/ATR no NULL", has_rsi and has_bb and has_atr,
               f"rsi={row.get('rsi_14')}, bb={row.get('bb_width')}, atr={row.get('atr_14')}")
    else:
        record(CAT, "B12. AAPL: RSI/BB/ATR no NULL", False, "no data")


# ═══════════════════════════════════════════════════════════════════════════
#  C. CAPACIDAD Y MEJORA DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════

def test_capability(gb, engine, ff):
    CAT = "C.CAP"
    print("\n" + "=" * 70)
    print("  C. CAPACIDAD Y MEJORA DEL MODELO")
    print("=" * 70)

    z = engine.z_scores
    returns = gb.returns.apply(pd.to_numeric, errors='coerce').fillna(0).values
    T, N = z.shape
    tickers = gb.tickers

    # C1. R² global: model fit
    u_real = engine.u_real
    u_pred = engine.u_pred
    ss_res = np.nansum((u_real - u_pred) ** 2)
    ss_tot = np.nansum((u_real - np.nanmean(u_real, axis=0)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    record(CAT, "C1. R² global O-U", r2 > 0.5,
           f"R²={r2:.4f}")

    # C2. z-score predice dirección 5d
    hits = 0; total = 0
    for t in range(60, T - 6):
        for i in range(N):
            zt = z[t, i]
            if abs(zt) < 1.5 or np.isnan(zt):
                continue
            ret_5d = np.nansum(returns[t+1:t+6, i])
            if np.isnan(ret_5d):
                continue
            if (zt > 0 and ret_5d < 0) or (zt < 0 and ret_5d > 0):
                hits += 1
            total += 1
    hit5 = hits / max(total, 1)
    record(CAT, "C2. z-score predice dirección 5d (hit>48%)", hit5 > 0.48,
           f"hit={hit5:.1%} ({hits}/{total})")

    # C3. z-score predice dirección 20d
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
    record(CAT, "C3. z-score predice dirección 20d (hit>48%)", hit20 > 0.48,
           f"hit={hit20:.1%} ({hits20}/{total20})")

    # C4. Sector rotation
    sectors = gb.sectors
    sector_indices = {}
    for sec, sec_tickers in sectors.items():
        idx = [tickers.index(t) for t in sec_tickers if t in tickers]
        if idx:
            sector_indices[sec] = idx

    rot_pnl = []
    for t in range(80, T - 22, 20):
        sec_z = {}
        for sec, idx in sector_indices.items():
            zv = z[t, idx]
            zv = zv[~np.isnan(zv)]
            if len(zv) > 0:
                sec_z[sec] = np.mean(zv)
        if len(sec_z) < 4:
            continue
        sorted_s = sorted(sec_z.items(), key=lambda x: x[1])
        cold = sector_indices[sorted_s[0][0]]
        hot = sector_indices[sorted_s[-1][0]]
        r_cold = np.nanmean(np.nansum(returns[t+1:t+21, cold], axis=0))
        r_hot = np.nanmean(np.nansum(returns[t+1:t+21, hot], axis=0))
        rot_pnl.append(r_cold - r_hot)

    rot_cum = np.sum(rot_pnl) * 100 if rot_pnl else 0
    rot_hit = np.mean(np.array(rot_pnl) > 0) if rot_pnl else 0
    record(CAT, "C4. Sector rotation alpha > 0", rot_cum > 0,
           f"cum={rot_cum:+.1f}%, hit={rot_hit:.0%}")

    # C5. s(t) responds to VIX (guard: skip if VIX empty)
    if len(gb.vix) > 10:
        s_vals = []; vix_vals = []
        for t in range(30, T, 20):
            d = str(gb.returns.index[t].date())
            gb._calibrate_s(d)
            v = gb.vix.asof(gb.returns.index[t])
            if pd.notna(v):
                s_vals.append(gb.s); vix_vals.append(v)
        corr_sv = np.corrcoef(s_vals, vix_vals)[0, 1] if len(s_vals) > 5 else 0
        record(CAT, "C5. Corr(s, VIX) < -0.3", corr_sv < -0.3,
               f"corr={corr_sv:.3f}")
    else:
        record(CAT, "C5. Corr(s, VIX) < -0.3", False, "VIX data empty")

    # C6. Mean reversion portfolio 10d P&L
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
    mr_cum = np.sum(mr_pnl) * 100
    mr_hit = np.mean(np.array(mr_pnl) > 0)
    record(CAT, "C6. Mean reversion 10d P&L > 0", mr_cum > 0,
           f"cum={mr_cum:+.1f}%, hit={mr_hit:.0%}")

    # C7. α calibrado < α default
    from scipy.optimize import minimize_scalar
    lam_s = np.power(gb.eigenvalues, gb.s)
    lam_s[0] = 0.0
    f_k = gb.eigenvectors.T @ np.nan_to_num(ff.get_source_vector(tickers))
    u_k = np.nan_to_num(engine.gb.u.astype(np.float64)) @ gb.eigenvectors

    def err_alpha(a):
        mu = a * lam_s
        dk = np.where(mu > 1e-10, np.exp(-mu), 1.0)
        eq = np.where(mu > 1e-10, f_k / np.maximum(mu, 1e-10) * (1 - dk), f_k)
        pred = u_k[:-1] * dk[np.newaxis, :] + eq[np.newaxis, :]
        return float(np.sum((u_k[1:] - pred) ** 2))

    err_calibrated = err_alpha(engine.alpha)
    err_default = err_alpha(0.05)
    record(CAT, "C7. α calibrado < α default error", err_calibrated <= err_default,
           f"calib={err_calibrated:.2f}, default={err_default:.2f}")

    # C8. Volume data loaded
    record(CAT, "C8. Volume data loaded", not gb.volume.empty,
           f"shape={gb.volume.shape}")

    # C9. F scores diversificados
    scores = [v for v in ff.scores.values() if not np.isnan(v)]
    f_std = np.std(scores) if scores else 0
    record(CAT, "C9. F scores diversificados (std>0.05)", f_std > 0.05,
           f"std={f_std:.4f}, n={len(scores)}")

    # C10. Green kernel diagonal
    K_green = gb.green_kernel(engine.alpha, dt=5.0)
    diag = np.diag(K_green)
    record(CAT, "C10. Green kernel diagonal ∈ (0,1]", np.all(diag > 0) and np.all(diag <= 1.001),
           f"min={np.min(diag):.4f}, max={np.max(diag):.4f}")


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  TEST SUITE — Water-Landscape Model                              ║")
    print("╚" + "═" * 68 + "╝\n")

    db, ff, gb, engine, detector = setup()

    test_math(gb, engine, ff)
    test_data(db, gb, ff)
    test_capability(gb, engine, ff)

    # ── RESUMEN ──
    print("\n" + "=" * 70)
    print("  📊 RESUMEN")
    print("=" * 70)

    df = pd.DataFrame(results)
    for cat in ["A.MATH", "B.DATA", "C.CAP"]:
        sub = df[df["cat"] == cat]
        passed = sum(PASS in s for s in sub["status"])
        total = len(sub)
        label = {"A.MATH": "Matemáticas", "B.DATA": "Datos", "C.CAP": "Capacidad"}[cat]
        pct = passed / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {label:<15} {bar} {passed}/{total} ({pct:.0f}%)")

    failed = df[df["status"] == FAIL]
    if len(failed) > 0:
        print(f"\n  FALLOS ({len(failed)}):")
        for _, row in failed.iterrows():
            print(f"    {row['name']:55} {row['detail']}")

    print("=" * 70)


if __name__ == "__main__":
    main()
