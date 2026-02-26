"""
SIGNAL GENERATOR — GlobalMarketAnalyzer
=========================================
Orquesta todo el pipeline:
  Datos → Grafo → O-U → Residuos → Inercias → Señales → Supabase

Uso:
  python signal_generator.py                     # analiza hoy
  python signal_generator.py --date 2026-02-10   # fecha específica
  python signal_generator.py --backtest          # backtest completo
"""

import argparse
import logging
import time
from datetime import date

import numpy as np
import pandas as pd

from database_manager import DatabaseManager
from fundamental_filter import FundamentalFilter
from graph_builder import GraphBuilder
from heat_engine import HeatEngine
from inertia_detector import InertiaDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Umbrales (ajustados para sensibilidad razonable) ──
BUY_THRESHOLD   = -0.8    # z-score composite para señal BUY
SELL_THRESHOLD  =  0.8    # z-score composite para señal SELL
WATCH_THRESHOLD =  0.5    # umbral mínimo para WATCH
PROB_MIN        =  0.55   # probabilidad mínima para señal


class SignalGenerator:
    """
    Pipeline completo: datos → señales.

    Uso:
        gen = SignalGenerator()
        signals = gen.run()
        gen.save_signals()
    """

    def __init__(self):
        self.db = DatabaseManager()
        self.ff = FundamentalFilter(self.db)
        self.gb = GraphBuilder(self.db)
        self.engine: HeatEngine = None
        self.detector: InertiaDetector = None

        self.signals: list[dict] = []
        self.diagnostics: dict = {}

    def run(self, reference_date: str = None) -> list[dict]:
        """
        Ejecuta el pipeline completo para una fecha.
        Devuelve lista de señales.
        """
        t0 = time.time()
        ref = reference_date or date.today().strftime("%Y-%m-%d")
        logger.info(f"=== Signal Generator — {ref} ===")

        # ── 1. Fundamentales ──
        logger.info("[1/6] Calculando scores fundamentales...")
        fund_df = self.ff.compute_all()
        fund_summary = self.ff.summary()
        logger.info(f"  Clasificación: {fund_summary}")

        # ── 2. Grafo ──
        logger.info("[2/6] Construyendo grafo de correlaciones...")
        self.gb.load_data()
        self.gb.build(reference_date=ref)
        n_edges_pos = int((self.gb.W > 0).sum() / 2)
        n_edges_neg = int((self.gb.W < 0).sum() / 2)
        logger.info(f"  N = {self.gb.N} activos")
        logger.info(f"  s(t) = {self.gb.s:.3f} (no-localidad)")
        logger.info(f"  Aristas: {n_edges_pos} positivas + {n_edges_neg} negativas")

        # ── 3. O-U solver ──
        logger.info("[3/6] Resolviendo Ornstein-Uhlenbeck...")
        self.engine = HeatEngine(self.gb, self.ff)
        self.engine.solve(calibrate=True)
        logger.info(f"  α global = {self.engine.alpha:.4f}")

        # ── 4. Inercias ──
        logger.info("[4/6] Detectando inercias...")
        self.detector = InertiaDetector(self.engine)
        self.detector.analyze()

        mode_summary = self.detector.get_mode_summary()
        n_reversals = int(mode_summary["reversal"].sum())
        n_anomalies = int(mode_summary["anomaly"].sum())
        logger.info(f"  Modos con reversión: {n_reversals}")
        logger.info(f"  Modos con anomalía: {n_anomalies}")

        # ── 5. Generar señales ──
        logger.info("[5/6] Generando señales...")
        self.signals = self._generate_signals(ref)

        # ── 6. No-localidad ──
        nl = self.gb.nonlocality_ratio(self.engine.alpha)
        logger.info(f"  NL ratio = {nl:.4f}")

        # Diagnósticos
        elapsed = time.time() - t0
        self.diagnostics = {
            "date": ref,
            "n_assets": self.gb.N,
            "s": round(self.gb.s, 4),
            "alpha": round(self.engine.alpha, 4),
            "nonlocality_ratio": round(nl, 4),
            "n_signals": len(self.signals),
            "n_buy": sum(1 for s in self.signals if s["signal"] == "BUY"),
            "n_sell": sum(1 for s in self.signals if s["signal"] == "SELL"),
            "n_watch": sum(1 for s in self.signals if s["signal"] == "WATCH"),
            "elapsed_s": round(elapsed, 2),
        }

        logger.info(f"\n  {self.diagnostics}")
        return self.signals

    def _generate_signals(self, ref_date: str) -> list[dict]:
        """
        Combina scores + probabilidades O-U → señales.
        Usa probabilidades analíticas, no heurísticas.
        """
        signals = []
        asset_scores = self.detector.asset_scores
        z_real = self.engine.z_scores

        for i, ticker in enumerate(self.gb.tickers):
            score = asset_scores.get(ticker, 0.0)
            z = float(z_real[-1, i]) if len(z_real) > 0 else 0.0
            F = self.ff.scores.get(ticker, 0.0)
            classification = self.ff.classifications.get(ticker, "neutral")

            # Precio actual
            price = float(self.gb.prices[ticker].iloc[-1]) \
                if ticker in self.gb.prices.columns else None

            # Probabilidad O-U analítica
            prob = self.engine.compute_probability(i)
            p_rev = prob["p_reversion"]
            exp_ret = prob["expected_return"]
            half_life = prob["half_life_days"]

            # Score compuesto: z-score + inercia + fundamental
            composite = score  # ya incluye spectral + inertia + z

            # Determinar señal usando probabilidad + score
            # F constraint relajado: permite señales en growth stocks
            if composite < BUY_THRESHOLD and p_rev > PROB_MIN and F >= -0.05:
                signal = "BUY"
                confidence = round(p_rev * 100, 1)
            elif composite > SELL_THRESHOLD and p_rev > PROB_MIN and F <= 0.05:
                signal = "SELL"
                confidence = round(p_rev * 100, 1)
            elif abs(composite) > WATCH_THRESHOLD:
                signal = "WATCH"
                confidence = round(p_rev * 100, 1)
            else:
                signal = "HOLD"
                confidence = round((1 - p_rev) * 100, 1)

            # En crisis (s bajo), reducir confianza de señales individuales
            if signal != "HOLD":
                confidence = round(confidence * self.gb.s, 1)

            signals.append({
                "ticker":             ticker,
                "date":               ref_date,
                "signal":             signal,
                "confidence":         confidence,
                "price":              price,
                "strategy":           "OU_fractional_graph",
                "regime":             self._classify_regime(),
                "technical_score":    round(float(composite), 3),
                "fundamental_score":  round(float(F * 100), 3),
                "macro_score":        round(float(self.gb.s * 100), 1),
                "p_reversion":        p_rev,
                "expected_return_5d": exp_ret,
                "half_life_days":     half_life,
                "rationale":          self._build_rationale(
                    ticker, composite, z, F, classification, p_rev, half_life
                ),
            })

        return signals

    def _classify_regime(self) -> str:
        """Clasifica el régimen de mercado actual."""
        s = self.gb.s
        if s > 0.80:
            return "calm_local"
        elif s > 0.60:
            return "normal"
        elif s > 0.40:
            return "stressed"
        elif s > 0.25:
            return "contagion"
        else:
            return "crisis"

    def _build_rationale(self, ticker: str, score: float, z: float,
                         F: float, classification: str,
                         p_rev: float, half_life: float) -> str:
        """Genera explicación con probabilidades reales."""
        parts = []

        if z < -2:
            parts.append(f"z={z:.1f} (muy frío)")
        elif z < -1:
            parts.append(f"z={z:.1f} (frío)")
        elif z > 2:
            parts.append(f"z={z:.1f} (muy caliente)")
        elif z > 1:
            parts.append(f"z={z:.1f} (caliente)")

        if p_rev > 0.6:
            parts.append(f"P(rev)={p_rev:.0%}")
        if half_life < 30:
            parts.append(f"t½={half_life:.0f}d")

        if classification == "value_creator":
            parts.append("crea valor")
        elif classification == "speculative":
            parts.append("especulativo")

        s = self.gb.s
        if s < 0.5:
            parts.append(f"contagio (s={s:.2f})")

        return "; ".join(parts) if parts else "equilibrio"

    def save_signals(self):
        """Guarda señales en la tabla signals de Supabase."""
        if not self.signals:
            logger.warning("No hay señales para guardar")
            return

        # Filtrar solo señales no-HOLD
        to_save = [s for s in self.signals if s["signal"] != "HOLD"]

        if not to_save:
            logger.info("Todas las señales son HOLD, nada que guardar")
            return

        CHUNK = 400
        total = 0
        for i in range(0, len(to_save), CHUNK):
            chunk = to_save[i:i + CHUNK]
            self.db.client.table("signals").insert(chunk).execute()
            total += len(chunk)

        logger.info(f"  ✓ {total} señales guardadas en Supabase")

    def print_summary(self):
        """Imprime resumen legible con probabilidades."""
        if not self.signals:
            return

        buys  = [s for s in self.signals if s["signal"] == "BUY"]
        sells = [s for s in self.signals if s["signal"] == "SELL"]
        watch = [s for s in self.signals if s["signal"] == "WATCH"]

        print(f"\n{'='*75}")
        print(f"  SEÑALES — {self.diagnostics.get('date', 'hoy')}")
        print(f"  Régimen: {self._classify_regime()} | s = {self.gb.s:.3f}")
        print(f"  α = {self.engine.alpha:.4f} | NL = {self.diagnostics.get('nonlocality_ratio', 0):.4f}")
        print(f"{'='*75}")

        if buys:
            buys.sort(key=lambda x: x["p_reversion"], reverse=True)
            print(f"\n  🟢 BUY ({len(buys)}):")
            for s in buys[:15]:
                print(f"    {s['ticker']:<8} P(rev)={s['p_reversion']:.0%}  "
                      f"E[ret]={s['expected_return_5d']:>+.2f}%  "
                      f"t½={s['half_life_days']:>5.1f}d  {s['rationale']}")

        if sells:
            sells.sort(key=lambda x: x["p_reversion"], reverse=True)
            print(f"\n  🔴 SELL ({len(sells)}):")
            for s in sells[:15]:
                print(f"    {s['ticker']:<8} P(rev)={s['p_reversion']:.0%}  "
                      f"E[ret]={s['expected_return_5d']:>+.2f}%  "
                      f"t½={s['half_life_days']:>5.1f}d  {s['rationale']}")

        if watch:
            watch.sort(key=lambda x: abs(x["technical_score"]), reverse=True)
            print(f"\n  🟡 WATCH ({len(watch)}):")
            for s in watch[:10]:
                print(f"    {s['ticker']:<8} P(rev)={s['p_reversion']:.0%}  "
                      f"score={s['technical_score']:>+.3f}  {s['rationale']}")

        n_hold = sum(1 for s in self.signals if s['signal'] == 'HOLD')
        print(f"\n  HOLD: {n_hold}")
        print(f"  Tiempo: {self.diagnostics.get('elapsed_s', 0):.1f}s")
        print(f"{'='*75}\n")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GlobalMarketAnalyzer — Signal Generator"
    )
    parser.add_argument("--date", default=None, help="Fecha (YYYY-MM-DD)")
    parser.add_argument("--save", action="store_true", help="Guardar en Supabase")
    parser.add_argument("--backtest", action="store_true", help="Backtest completo")
    args = parser.parse_args()

    gen = SignalGenerator()
    signals = gen.run(reference_date=args.date)
    gen.print_summary()

    if args.save:
        gen.save_signals()


if __name__ == "__main__":
    main()
