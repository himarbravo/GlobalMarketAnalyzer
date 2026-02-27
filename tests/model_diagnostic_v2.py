"""
MODEL DIAGNOSTIC v2 — Tests Avanzados
========================================
Tests específicos para validar las mejoras:

  7. Non-locality: ¿W²/W³ capta propagación indirecta real?
  8. Event backtesting: simular eventos y comparar con datos reales
  9. Inflation islands: ¿activos USD vs EUR tienen inflación coherente?
  10. Signal pipeline final con todas las mejoras

Ejecuta PRIMERO model_diagnostic.py (tests 1-6) y luego este.

Uso:
  python model_diagnostic_v2.py
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
from perturbation_simulator import PerturbationSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_advanced_diagnostics():
    t0 = time.time()
    logger.info("="*70)
    logger.info("  MODEL DIAGNOSTIC v2 — Tests Avanzados")
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

    sim = PerturbationSimulator(gb, engine)

    logger.info(f"  {gb.N} activos, {gb.returns.shape[0]} días")
    logger.info(f"  α={engine.alpha:.4f}, s={gb.s:.3f}")

    returns_df = gb.returns.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float64)

    # ══════════════════════════════════════════════════════════════
    # TEST 7: ¿W²/W³ CAPTA NO-LOCALIDAD REAL?
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("  TEST 7: NON-LOCALITY — ¿W²/W³ captan propagación indirecta?")
    logger.info("="*70)

    # Comparar: pares que NO son vecinos directos (W_direct=0)
    # pero SÍ tienen peso en W (gracias a W², W³).
    # ¿Existe correlación real entre esos pares con lag?

    W_direct = gb.W_direct
    W_full = gb.W

    indirect_pairs = []
    for i in range(gb.N):
        for j in range(i+1, gb.N):
            if abs(W_direct[i,j]) < 0.01 and abs(W_full[i,j]) > 0.01:
                # Par conectado SOLO por vecinos indirectos
                indirect_pairs.append((i, j, W_full[i,j]))

    print(f"\n  Pares conectados SOLO por W²/W³ (no vecinos directos): {len(indirect_pairs)}")
    print(f"  {'Par':>20} {'W_full':>8} {'corr_real':>10} {'p-value':>10} {'¿Real?':>8}")
    print(f"  {'-'*60}")

    n_real = 0
    tested = 0
    for i, j, w in sorted(indirect_pairs, key=lambda x: abs(x[2]), reverse=True)[:20]:
        # Correlación real con lag
        r_i = returns_df.iloc[:, i].values
        r_j = returns_df.iloc[:, j].values

        # Buscar mejor lag [-10, +10]
        best_c, best_lag = 0, 0
        for lag in range(-10, 11):
            if lag >= 0:
                x, y = r_i[:len(r_i)-lag], r_j[lag:]
            else:
                x, y = r_i[-lag:], r_j[:len(r_j)+lag]
            min_l = min(len(x), len(y))
            if min_l > 30:
                c, p = stats.spearmanr(x[:min_l], y[:min_l])
                if abs(c) > abs(best_c):
                    best_c, best_lag = c, lag

        is_real = abs(best_c) > 0.05  # hay correlación real
        if is_real:
            n_real += 1
        tested += 1
        tag = "✅" if is_real else "❌"
        print(f"  {gb.tickers[i]:>8}↔{gb.tickers[j]:<8} {w:>+8.4f} "
              f"{best_c:>+10.4f} lag={best_lag:+d}d   {tag}")

    nl_rate = n_real / max(tested, 1)
    print(f"\n  Non-locality validada: {n_real}/{tested} = {nl_rate:.0%}")
    print(f"  {'🟢' if nl_rate > 0.5 else '🟡' if nl_rate > 0.3 else '🔴'}"
          f" W²/W³ {'captan' if nl_rate > 0.3 else 'no captan'} propagación indirecta real")

    # ══════════════════════════════════════════════════════════════
    # TEST 8: BACKTESTING CON EVENTOS HISTÓRICOS
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("  TEST 8: EVENT BACKTESTING — Eventos reales en nuestros datos")
    logger.info("="*70)

    # Identificar eventos reales en nuestro rango de datos (2024-02 a 2026-02)
    # Buscamos caídas/subidas extremas y verificamos si la perturbación
    # simulada coincide con la propagación real
    dates = gb.returns.index

    print(f"\n  Buscando eventos extremos en {len(dates)} días de datos...")
    print(f"  (eventos = días donde algún activo se mueve > 3σ)")

    # Encontrar eventos extremos
    events_found = []
    vol = returns_df.std()
    for d_idx in range(30, len(dates) - 10):
        for i in range(gb.N):
            ret = returns_df.iloc[d_idx, i]
            if vol.iloc[i] > 0 and abs(ret) > 3 * vol.iloc[i]:
                events_found.append({
                    "date": dates[d_idx],
                    "ticker": gb.tickers[i],
                    "return": ret,
                    "sigma": ret / vol.iloc[i],
                    "d_idx": d_idx,
                })

    # Agrupar por fecha (tomar el más extremo por día)
    events_by_date = {}
    for e in events_found:
        d = str(e["date"].date())
        if d not in events_by_date or abs(e["sigma"]) > abs(events_by_date[d]["sigma"]):
            events_by_date[d] = e

    # Top 10 eventos
    top_events = sorted(events_by_date.values(),
                       key=lambda x: abs(x["sigma"]), reverse=True)[:10]

    print(f"\n  Top 10 eventos extremos:")
    print(f"  {'Fecha':>12} {'Ticker':>8} {'Retorno':>10} {'σ':>6}  Propagación real vs simulada")
    print(f"  {'-'*70}")

    event_scores = []
    for event in top_events:
        ticker = event["ticker"]
        d_idx = event["d_idx"]
        delta = event["return"]

        # Propagación REAL en los 10 días siguientes
        real_impact = returns_df.iloc[d_idx+1:d_idx+11].sum().values  # (N,)

        # Propagación SIMULADA
        sim_df = sim.simulate(ticker, delta=delta, horizon=10)
        sim_impact = sim_df.sum(axis=0).values  # (N,)

        # Correlación entre real y simulada (excluyendo el propio activo)
        i_tick = gb.tickers.index(ticker)
        mask = np.ones(gb.N, dtype=bool)
        mask[i_tick] = False
        real_others = real_impact[mask]
        sim_others = sim_impact[mask]

        corr_val, p_val = stats.spearmanr(real_others, sim_others)
        tag = "✅" if corr_val > 0.15 and p_val < 0.1 else (
              "⚠️" if corr_val > 0 else "❌")

        print(f"  {str(event['date'].date()):>12} {ticker:>8} {delta:>+10.4f} "
              f"{event['sigma']:>+5.1f}σ  corr(real,sim)={corr_val:+.3f} {tag}")

        event_scores.append(corr_val)

    avg_event_corr = np.mean(event_scores)
    print(f"\n  Correlación media real↔simulada: {avg_event_corr:+.3f}")
    print(f"  {'🟢' if avg_event_corr > 0.15 else '🟡' if avg_event_corr > 0 else '🔴'}"
          f" El simulador {'predice' if avg_event_corr > 0.1 else 'no predice'}"
          f" la propagación de eventos reales")

    # ══════════════════════════════════════════════════════════════
    # TEST 9: PERTURBACIONES SINTÉTICAS (crisis simuladas)
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("  TEST 9: CRISIS SIMULADAS — Escenarios hipotéticos")
    logger.info("="*70)

    scenarios = [
        ("Crash tech -10%",
         [("NVDA", -0.10), ("AMD", -0.10), ("AVGO", -0.10), ("ASML", -0.10)]),
        ("Bajada de tipos (bonds+5%)",
         [("TLT", +0.05), ("IEF", +0.03), ("SHY", +0.01)]),
        ("Crisis de crédito (HYG -5%)",
         [("HYG", -0.05), ("LQD", -0.03)]),
        ("Oil shock +15%",
         [("USO", +0.15)]),
        ("Flight to safety",
         [("GLD", +0.08), ("TLT", +0.05), ("SHY", +0.02)]),
        ("Emergentes -8%",
         [("EWZ", -0.08), ("FXI", -0.08), ("INDA", -0.08)]),
    ]

    from model_diagnostic import SECTORS

    for name, events in scenarios:
        impact_df = sim.scenario(events, horizon=10)
        cumulative = impact_df.sum(axis=0)

        print(f"\n  📌 {name}")
        print(f"  {'Sector':<14} {'Impacto medio':>14}")
        print(f"  {'-'*30}")

        for sector, stickers in SECTORS.items():
            sector_impact = [cumulative[t] for t in stickers if t in cumulative.index]
            if sector_impact:
                avg = np.mean(sector_impact)
                if abs(avg) > 0.0005:
                    tag = "🔴" if avg < -0.003 else ("🟡" if avg < 0 else "🟢")
                    print(f"  {tag} {sector:<12} {avg:>+14.4f}")

    # ══════════════════════════════════════════════════════════════
    # TEST 10: INFLACIÓN POR ISLAS DE MONEDA
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("  TEST 10: INFLACIÓN POR MONEDA — ¿Se reflejan por 'islas'?")
    logger.info("="*70)

    # Todos nuestros activos cotizan en USD. Pero algunos son europeos
    # (ASML, SAP, NVO) o emergentes (EWZ, FXI). Verificar:
    # 1. ¿La inflación USD se refleja uniformemente?
    # 2. ¿Activos EUR-denominados responden diferente a yield_2y (proxy inflación)?

    eur_proxy_tickers = ["ASML", "SAP", "NVO", "EWG", "EWU"]
    usd_native_tickers = ["AAPL", "MSFT", "JPM", "WMT", "JNJ"]
    em_proxy_tickers = ["EWZ", "FXI", "INDA", "BABA"]
    commodity_tickers = ["GLD", "USO", "DBA", "SLV"]

    groups = {
        "USD nativo": usd_native_tickers,
        "EUR proxy": eur_proxy_tickers,
        "Emergentes": em_proxy_tickers,
        "Commodities": commodity_tickers,
    }

    # Correlación de retornos con proxy de inflación (yield_2y daily change)
    inflation_proxy = gb.inflation_daily.reindex(dates).ffill().fillna(0)
    inflation_changes = inflation_proxy.diff().fillna(0)

    print(f"\n  ¿Cómo responden las 'islas' a cambios de inflación (yield 2y)?")
    print(f"  {'Grupo':<16} {'corr(r, Δπ)':>12} {'p-value':>10} {'β (sensib.)':>12}")
    print(f"  {'-'*55}")

    for group_name, tickers_list in groups.items():
        indices = [gb.tickers.index(t) for t in tickers_list if t in gb.tickers]
        if not indices:
            continue

        all_r, all_pi = [], []
        for i in indices:
            r = returns_df.iloc[:, i].values.astype(np.float64)
            pi = inflation_changes.values[:len(r)].astype(np.float64)
            min_l = min(len(r), len(pi))
            mask = ~(np.isnan(r[:min_l]) | np.isnan(pi[:min_l]))
            all_r.extend(r[:min_l][mask])
            all_pi.extend(pi[:min_l][mask])

        if len(all_r) > 100 and np.std(all_pi) > 1e-10:
            corr_val, p_val = stats.spearmanr(all_pi, all_r)
            slope, _, _, _, _ = stats.linregress(all_pi, all_r)
            print(f"  {group_name:<16} {corr_val:>+12.4f} {p_val:>10.2e} {slope:>+12.4f}")
        elif len(all_r) > 100:
            print(f"  {group_name:<16}    (inflación constante en ventana — sin señal)")

    # Grafo: ¿forman islas los activos por moneda?
    print(f"\n  ¿Se agrupan en islas en el grafo?")
    print(f"  (Conectividad INTRA-grupo vs INTER-grupo)")
    print(f"  {'Grupo':<16} {'Intra':>8} {'Inter':>8} {'Ratio I/E':>10} {'¿Isla?':>8}")
    print(f"  {'-'*55}")

    for group_name, tickers_list in groups.items():
        indices = [gb.tickers.index(t) for t in tickers_list if t in gb.tickers]
        if len(indices) < 2:
            continue

        # Conectividad intra-grupo
        intra = 0
        n_intra = 0
        for a in indices:
            for b in indices:
                if a != b:
                    intra += abs(gb.W[a, b])
                    n_intra += 1

        # Conectividad inter-grupo
        inter = 0
        n_inter = 0
        other_indices = [i for i in range(gb.N) if i not in indices]
        for a in indices:
            for b in other_indices:
                inter += abs(gb.W[a, b])
                n_inter += 1

        avg_intra = intra / max(n_intra, 1)
        avg_inter = inter / max(n_inter, 1)
        ratio = avg_intra / max(avg_inter, 1e-6)

        tag = "✅ isla" if ratio > 1.5 else ("🟡 parcial" if ratio > 1.0 else "❌ no")
        print(f"  {group_name:<16} {avg_intra:>8.4f} {avg_inter:>8.4f} "
              f"{ratio:>10.2f} {tag:>8}")

    # ══════════════════════════════════════════════════════════════
    # TEST 11: SEÑAL PIPELINE MEJORADO
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("  TEST 11: PIPELINE FINAL — Señales con todas las mejoras")
    logger.info("="*70)

    # Top señales del detector mejorado
    alerts = detector.get_alerts()
    print(f"\n  Top 15 señales del modelo mejorado:")
    print(f"  {'Ticker':>8} {'Score':>8} {'Dir':>6} {'Estado':>12} {'Mass':>6} {'H':>6}")
    print(f"  {'-'*55}")
    for a in alerts[:15]:
        print(f"  {a['ticker']:>8} {a['score']:>+8.3f} {a['direction']:>6} "
              f"{a['state']:>12} {a['mass']:>6.3f} {a['hysteresis']:>6.2f}")

    # Phase report
    print(detector.get_phase_report())

    # ══════════════════════════════════════════════════════════════
    # RESUMEN v2
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  DIAGNÓSTICO v2 COMPLETO — {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"  Non-locality W²/W³:  {n_real}/{tested} "
          f"({nl_rate:.0%}) pares indirectos validados")
    print(f"  Event backtesting:    corr media = {avg_event_corr:+.3f}")
    print(f"  Señales activas:      {len(alerts)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_advanced_diagnostics()
