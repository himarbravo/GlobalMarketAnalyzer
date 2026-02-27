"""
MODEL DIAGNOSTIC — GlobalMarketAnalyzer
==========================================
Evaluación cuantitativa del modelo O-U en grafo fraccional.

5 tests:
  1. Tracking por sector: ¿dónde sigue bien la tendencia?
  2. Capacidad predictiva: ¿z-scores predicen retornos futuros?
  3. Leading indicators: ¿el modelo "chiva" antes de cambios reales?
  4. Correlaciones no obvias: cobre→tech, bonds→equities, etc.
  5. Régimen macro: comportamiento bajo inflación/deflación

Uso:
  python model_diagnostic.py
"""

import logging
import time
import numpy as np
import pandas as pd
from scipy import stats

from db.database_manager import DatabaseManager
from core.fundamental_filter import FundamentalFilter
from core.graph_builder import GraphBuilder
from core.heat_engine import HeatEngine
from core.inertia_detector import InertiaDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Clasificación por sector ──
SECTORS = {
    "Tech":       ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "ADBE", "CRM",
                   "AMD", "AVGO", "ASML", "QCOM", "INTC", "WDAY", "SAP"],
    "Finanzas":   ["JPM", "BAC", "GS", "MS", "WFC", "V", "MA"],
    "Salud":      ["JNJ", "UNH", "PFE", "ABBV", "AMGN", "VRTX", "NVO", "LLY"],
    "Energía":    ["XOM", "CVX", "COP", "ENPH"],
    "Industrial": ["CAT", "DE", "BA", "HON", "RTX", "GE", "LMT"],
    "Consumo":    ["WMT", "COST", "PG", "KO", "PEP", "MCD", "SBUX", "NKE"],
    "Bonds/RF":   ["TLT", "IEF", "SHY", "LQD", "HYG"],
    "Commodities":["GLD", "SLV", "USO", "DBA"],
    "Crypto":     ["BTC-USD", "ETH-USD"],
    "ETF_Sector": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE"],
    "Emergentes": ["EWZ", "EWG", "EWU", "FXI", "BABA"],
}

# ── Correlaciones no obvias que deberían existir ──
EXPECTED_CROSS_CORRELATIONS = [
    # (activo_líder, activo_seguidor, dirección_esperada, lag_días, explicación)
    ("GLD",     "XLK",   "negative", 5,  "Oro sube → tech baja (flight to safety)"),
    ("USO",     "XLE",   "positive", 1,  "Petróleo sube → energéticas suben (directo)"),
    ("DBA",     "XLP",   "positive", 3,  "Agrícolas suben → consumo sube (costes)"),

    # Bonds vs Equities
    ("TLT",     "XLK",   "negative", 3,  "Bonos suben → tech baja (risk-off)"),
    ("IEF",     "XLF",   "negative", 5,  "Bonos suben → financieras bajan (márgenes)"),
    ("SHY",     "XLY",   "negative", 1,  "Cash sube → consumo discrecional baja"),

    # Deuda y tipos
    ("HYG",     "XLF",   "positive", 3,  "High yield sube → financieras sube (optimismo)"),
    ("LQD",     "XLU",   "positive", 5,  "Corp bonds → utilities (ambos yield)"),

    # Metales → Tech (cobre como proxy de demanda industrial)
    # No tenemos cobre directamente, usaremos FCX (Freeport-McMoRan) como proxy
    ("FCX",     "NVDA",  "positive", 10, "Cobre (proxy) sube → tech industrial sube (demanda)"),
    ("FCX",     "CAT",   "positive", 5,  "Cobre → industrial (construcción)"),

    # Inflación proxies
    ("GLD",     "TLT",   "negative", 10, "Oro sube → bonos bajan (inflación esperada)"),
    ("USO",     "SHY",   "negative", 5,  "Petróleo → cash pierde valor (inflación)"),

    # Crypto vs traditional
    ("BTC-USD", "GLD",   "positive", 5,  "Bitcoin → oro (correlación digital/real)"),
    ("BTC-USD", "XLK",   "positive", 3,  "Bitcoin → tech (sentimiento risk-on)"),
]


def run_diagnostics():
    """Ejecuta los 5 tests de diagnóstico."""
    t0 = time.time()

    # ── Setup ──
    logger.info("="*70)
    logger.info("  MODEL DIAGNOSTIC — GlobalMarketAnalyzer")
    logger.info("="*70)

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

    logger.info(f"  Datos: {gb.returns.shape[0]} días × {gb.N} activos")
    logger.info(f"  α={engine.alpha:.4f}, s={gb.s:.3f}")

    results = {}

    # ══════════════════════════════════════════════════════════════
    # TEST 1: TRACKING POR SECTOR
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("  TEST 1: TRACKING POR SECTOR")
    logger.info("="*70)

    # Métricas: MAE, correlation, R² entre u_pred y u_real por sector
    u_real = engine.u_real  # (T, N)
    u_pred = engine.u_pred  # (T, N)

    sector_results = {}
    for sector, tickers in SECTORS.items():
        indices = [gb.tickers.index(t) for t in tickers if t in gb.tickers]
        if not indices:
            continue

        r = u_real[:, indices]
        p = u_pred[:, indices]

        # Tracking error (MAE)
        mae = np.nanmean(np.abs(r - p))

        # Correlation between real and predicted (last 100 days)
        r_last = r[-100:].flatten()
        p_last = p[-100:].flatten()
        mask = ~(np.isnan(r_last) | np.isnan(p_last))
        if mask.sum() > 10:
            corr_val = np.corrcoef(r_last[mask], p_last[mask])[0, 1]
        else:
            corr_val = np.nan

        # R² = 1 - SSres/SStot
        ss_res = np.nansum((r - p) ** 2)
        ss_tot = np.nansum((r - np.nanmean(r)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        sector_results[sector] = {
            "n_tickers": len(indices),
            "MAE": round(mae, 4),
            "R²": round(r_squared, 4),
            "corr": round(corr_val, 4) if not np.isnan(corr_val) else None,
        }

    # Print sorted by R²
    print(f"\n  {'Sector':<14} {'N':>3} {'MAE':>8} {'R²':>8} {'Corr':>8}  Evaluación")
    print(f"  {'-'*60}")
    for sector, m in sorted(sector_results.items(), key=lambda x: x[1].get("R²", 0), reverse=True):
        r2 = m["R²"]
        if r2 > 0.8:
            grade = "🟢 Excelente tracking"
        elif r2 > 0.5:
            grade = "🟡 Buen tracking"
        elif r2 > 0.2:
            grade = "🟠 Tracking parcial"
        else:
            grade = "🔴 Tracking pobre"
        corr_str = f"{m['corr']:.3f}" if m['corr'] is not None else "  N/A"
        print(f"  {sector:<14} {m['n_tickers']:>3} {m['MAE']:>8.4f} {r2:>8.4f} {corr_str:>8}  {grade}")

    results["sector_tracking"] = sector_results

    # ══════════════════════════════════════════════════════════════
    # TEST 2: CAPACIDAD PREDICTIVA
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("  TEST 2: CAPACIDAD PREDICTIVA (z-scores → retornos futuros)")
    logger.info("="*70)

    # ¿Un z-score extremo hoy predice el retorno futuro?
    # Para cada horizonte h: corr(z[t], return[t+h])
    # Reversión → correlación negativa (z alto → retorno negativo)
    returns_df = gb.returns.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float64)
    z_scores = engine.z_scores

    horizons = [1, 3, 5, 10, 20]
    predictive_results = {}

    print(f"\n  Horizonte | Corr(z, r_fwd) | p-value  | ¿Predictivo?")
    print(f"  {'-'*55}")

    for h in horizons:
        # Forward return
        fwd_returns = returns_df.shift(-h)  # return over next h days
        # Actually we want the cumulative return
        fwd_cum = returns_df.rolling(h).sum().shift(-h)

        all_z = []
        all_r = []
        for i in range(gb.N):
            z_col = z_scores[:-h, i] if h < len(z_scores) else []
            r_col = fwd_cum.iloc[:-h, i].values if h < len(fwd_cum) else []

            min_len = min(len(z_col), len(r_col))
            if min_len > 20:
                z_valid = z_col[:min_len]
                r_valid = r_col[:min_len]
                mask = ~(np.isnan(z_valid) | np.isnan(r_valid))
                all_z.extend(z_valid[mask])
                all_r.extend(r_valid[mask])

        if len(all_z) > 100:
            corr_val, p_val = stats.spearmanr(all_z, all_r)
            is_predictive = "✅ SÍ" if p_val < 0.05 and corr_val < 0 else ("⚠️ parcial" if p_val < 0.10 else "❌ NO")
            print(f"  {h:>3}d       | {corr_val:>+.4f}         | {p_val:.2e} | {is_predictive}")
            predictive_results[f"{h}d"] = {
                "corr": round(corr_val, 4),
                "p_value": round(p_val, 6),
                "predictive": p_val < 0.05 and corr_val < 0,
            }

    # Per-sector predictiveness (5-day horizon)
    print(f"\n  Predictividad por sector (horizonte 5d):")
    print(f"  {'Sector':<14} {'corr(z,r)':>10} {'p-value':>10} {'Predictivo':>12}")
    print(f"  {'-'*50}")

    fwd_5d = returns_df.rolling(5).sum().shift(-5)
    for sector, tickers in SECTORS.items():
        indices = [gb.tickers.index(t) for t in tickers if t in gb.tickers]
        if not indices:
            continue

        all_z, all_r = [], []
        for i in indices:
            z_col = z_scores[:-5, i]
            r_col = fwd_5d.iloc[:-5, i].values
            min_len = min(len(z_col), len(r_col))
            if min_len > 20:
                mask = ~(np.isnan(z_col[:min_len]) | np.isnan(r_col[:min_len]))
                all_z.extend(z_col[:min_len][mask])
                all_r.extend(r_col[:min_len][mask])

        if len(all_z) > 50:
            c, p = stats.spearmanr(all_z, all_r)
            tag = "✅" if p < 0.05 and c < 0 else ("⚠️" if p < 0.10 else "❌")
            print(f"  {sector:<14} {c:>+10.4f} {p:>10.2e} {tag:>12}")

    results["predictive_power"] = predictive_results

    # ══════════════════════════════════════════════════════════════
    # TEST 3: LEADING INDICATORS
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("  TEST 3: LEADING INDICATORS (¿chiva el modelo antes de cambios?)")
    logger.info("="*70)

    # Detectar cambios de tendencia y ver si el z-score se movió antes
    # Un "cambio de tendencia" = retorno acumulado 20d cambia de signo
    print(f"\n  Análisis de anticipación: ¿los z-scores extremos preceden cambios?")
    print(f"  {'Ticker':<8} {'z extremos':>10} {'Seguidos de reversal':>22} {'Hit rate':>10}")
    print(f"  {'-'*55}")

    leading_results = {}
    for i, ticker in enumerate(gb.tickers):
        z_i = z_scores[:, i]
        ret_i = returns_df.iloc[:, i].values

        # Encontrar días donde |z| > 2 (señal extrema)
        extreme_days = np.where(np.abs(z_i) > 2.0)[0]
        extreme_days = extreme_days[extreme_days < len(ret_i) - 10]  # margen

        if len(extreme_days) < 3:
            continue

        # ¿Cuántos de esos días extremos fueron seguidos por una reversión en 10 días?
        hits = 0
        for d in extreme_days:
            z_sign = np.sign(z_i[d])
            fwd_ret = np.nansum(ret_i[d+1:d+11])  # retorno acumulado 10d
            # Reversión = el retorno va en dirección opuesta al z-score
            if z_sign * fwd_ret < 0:
                hits += 1

        hit_rate = hits / len(extreme_days) if extreme_days.size > 0 else 0
        leading_results[ticker] = {
            "n_extremes": len(extreme_days),
            "n_hits": hits,
            "hit_rate": round(hit_rate, 3),
        }

    # Print top 15 by number of extremes
    sorted_leading = sorted(leading_results.items(),
                           key=lambda x: x[1]["n_extremes"], reverse=True)[:15]
    for ticker, m in sorted_leading:
        hr = m["hit_rate"]
        tag = "🟢" if hr > 0.55 else ("🟡" if hr > 0.45 else "🔴")
        print(f"  {ticker:<8} {m['n_extremes']:>10} {m['n_hits']:>22} {hr:>8.0%}  {tag}")

    # Summary by sector
    print(f"\n  Hit rate por sector (% veces que z extremo precede reversión):")
    for sector, tickers in SECTORS.items():
        sector_hrs = [leading_results[t]["hit_rate"]
                     for t in tickers if t in leading_results]
        if sector_hrs:
            avg_hr = np.mean(sector_hrs)
            tag = "🟢" if avg_hr > 0.55 else ("🟡" if avg_hr > 0.45 else "🔴")
            print(f"  {sector:<14} {avg_hr:.0%}  {tag}")

    results["leading_indicators"] = leading_results

    # ══════════════════════════════════════════════════════════════
    # TEST 4: CORRELACIONES NO OBVIAS
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("  TEST 4: CORRELACIONES NO OBVIAS")
    logger.info("="*70)

    print(f"\n  {'Líder':>8} → {'Seguidor':<8} {'lag':>4} {'corr_real':>10} {'corr_grafo':>11} {'¿Captura?':>10}")
    print(f"  {'-'*65}")

    cross_results = []
    for leader, follower, direction, lag, explanation in EXPECTED_CROSS_CORRELATIONS:
        if leader not in gb.tickers or follower not in gb.tickers:
            continue

        i_lead = gb.tickers.index(leader)
        i_follow = gb.tickers.index(follower)

        # Correlación real con lag
        r_lead = returns_df.iloc[:, i_lead].values
        r_follow = returns_df.iloc[:, i_follow].values

        # Lag correlation: corr(leader[t], follower[t+lag])
        if lag < len(r_lead):
            lead_slice = r_lead[:-lag] if lag > 0 else r_lead
            follow_slice = r_follow[lag:] if lag > 0 else r_follow
            min_len = min(len(lead_slice), len(follow_slice))
            mask = ~(np.isnan(lead_slice[:min_len]) | np.isnan(follow_slice[:min_len]))
            if mask.sum() > 30:
                real_corr, _ = stats.spearmanr(lead_slice[:min_len][mask],
                                               follow_slice[:min_len][mask])
            else:
                real_corr = np.nan
        else:
            real_corr = np.nan

        # ¿El grafo capturó esta relación?
        graph_weight = gb.W[i_lead, i_follow]

        # Evaluar
        expected_sign = -1 if direction == "negative" else 1
        captures_direction = (np.sign(real_corr) == expected_sign) if not np.isnan(real_corr) else False
        graph_sees_it = abs(graph_weight) > 0.01

        tag = "✅" if captures_direction and graph_sees_it else (
              "⚠️" if captures_direction or graph_sees_it else "❌")

        print(f"  {leader:>8} → {follower:<8} {lag:>4}d {real_corr:>+10.4f} "
              f"{graph_weight:>+11.4f} {tag:>10}")
        print(f"           {explanation}")

        cross_results.append({
            "leader": leader, "follower": follower,
            "lag": lag, "real_corr": round(real_corr if not np.isnan(real_corr) else 0, 4),
            "graph_weight": round(float(graph_weight), 4),
            "captures": captures_direction and graph_sees_it,
        })

    n_captured = sum(1 for x in cross_results if x["captures"])
    print(f"\n  Capturadas: {n_captured}/{len(cross_results)} "
          f"({n_captured/len(cross_results)*100:.0f}%)")

    results["cross_correlations"] = cross_results

    # ══════════════════════════════════════════════════════════════
    # TEST 5: ANÁLISIS DE RÉGIMEN MACRO
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("  TEST 5: RÉGIMEN MACRO (inflación, deflación, crisis)")
    logger.info("="*70)

    # Dividir los datos en periodos según indicadores macro
    dates = gb.returns.index

    # Inferir periodos de inflación/deflación/crisis desde yield_2y y VIX
    print(f"\n  Parámetros del modelo por régimen macro:")
    print(f"  {'Régimen':<18} {'Días':>5} {'α':>8} {'s(t)':>8} {'NL':>8} "
          f"{'MAE':>8} {'z_std':>8}")
    print(f"  {'-'*65}")

    # Classify each date into a regime
    regimes = pd.Series("normal", index=dates)

    vix_aligned = gb.vix.reindex(dates).ffill()
    spread_aligned = gb.yield_spread.reindex(dates).ffill()
    y2_aligned = pd.Series(dtype=float)
    if len(gb.inflation_daily) > 0:
        y2_aligned = (gb.inflation_daily * 252 * 100).reindex(dates).ffill()

    if len(vix_aligned) > 0:
        regimes[vix_aligned > 25] = "alta_volatilidad"
        regimes[vix_aligned > 35] = "crisis"

    if len(spread_aligned) > 0:
        regimes[(spread_aligned < -0.5) & (regimes == "normal")] = "curva_invertida"

    if len(y2_aligned) > 0:
        regimes[(y2_aligned > 4) & (regimes == "normal")] = "inflación_alta"
        regimes[(y2_aligned < 1) & (regimes == "normal")] = "tipos_bajos"

    # Per-regime statistics
    for regime_name in regimes.unique():
        mask_idx = regimes[regimes == regime_name].index
        day_indices = [dates.get_loc(d) for d in mask_idx if d in dates]

        if len(day_indices) < 10:
            continue

        # Slice
        u_r_slice = u_real[day_indices]
        u_p_slice = u_pred[day_indices]
        z_slice = z_scores[day_indices]

        mae = np.nanmean(np.abs(u_r_slice - u_p_slice))
        z_std = np.nanstd(z_slice)

        # Recalculate s for this period's last date
        last_date = str(mask_idx[-1].date())
        ref_s = gb.s  # approximate

        print(f"  {regime_name:<18} {len(day_indices):>5} {engine.alpha:>8.4f} "
              f"{ref_s:>8.3f} {'-':>8} {mae:>8.4f} {z_std:>8.3f}")

    # ══════════════════════════════════════════════════════════════
    # TEST 6: SPECTRAL MODE INTERPRETATION
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("  TEST 6: INTERPRETACIÓN DE MODOS ESPECTRALES")
    logger.info("="*70)

    print(f"\n  Los primeros modos del grafo representan los \"temas\" de mercado:")
    print(f"  {'Modo':>5} {'λ':>8} {'λ^s':>8} {'τ (días)':>10} {'Positivos':>35} {'Negativos':>35}")
    print(f"  {'-'*105}")

    lam_s = np.power(gb.eigenvalues, gb.s)
    for k in range(1, min(8, gb.N)):  # skip mode 0 (constant)
        interp = gb.mode_interpretation(k)
        tau_k = 1.0 / (engine.alpha * lam_s[k]) if lam_s[k] > 1e-6 else np.inf

        pos_str = ", ".join(f"{t}({v:+.2f})" for t, v in interp["positive"][:3])
        neg_str = ", ".join(f"{t}({v:+.2f})" for t, v in interp["negative"][:3])

        print(f"  {k:>5} {gb.eigenvalues[k]:>8.3f} {lam_s[k]:>8.3f} "
              f"{tau_k:>10.1f} {pos_str:>35} {neg_str:>35}")

    # ══════════════════════════════════════════════════════════════
    # RESUMEN FINAL
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  DIAGNÓSTICO COMPLETO — {elapsed:.1f}s")
    print(f"{'='*70}")

    # Sector tracking summary
    good_tracking = sum(1 for v in sector_results.values() if v["R²"] > 0.5)
    print(f"\n  Tracking:      {good_tracking}/{len(sector_results)} sectores con R² > 0.5")

    # Predictive power
    pred_horizons = [k for k, v in predictive_results.items() if v["predictive"]]
    print(f"  Predictivo:    {', '.join(pred_horizons) if pred_horizons else 'Ningún horizonte'}")

    # Leading indicators
    good_leads = sum(1 for v in leading_results.values() if v["hit_rate"] > 0.55)
    total_leads = len(leading_results)
    print(f"  Leading:       {good_leads}/{total_leads} activos con hit rate > 55%")

    # Cross correlations
    print(f"  Correlaciones: {n_captured}/{len(cross_results)} no obvias capturadas")

    # Overall verdict
    overall_score = (
        (good_tracking / max(len(sector_results), 1)) * 0.25 +
        (len(pred_horizons) / max(len(horizons), 1)) * 0.30 +
        (good_leads / max(total_leads, 1)) * 0.25 +
        (n_captured / max(len(cross_results), 1)) * 0.20
    )

    print(f"\n  SCORE GLOBAL: {overall_score:.0%}")
    if overall_score > 0.7:
        print(f"  Veredicto: 🟢 Modelo robusto — listo para backtesting")
    elif overall_score > 0.4:
        print(f"  Veredicto: 🟡 Modelo aceptable — necesita refinamiento")
    else:
        print(f"  Veredicto: 🔴 Modelo débil — requiere revisión fundamental")

    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    run_diagnostics()
